from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import logging
import websockets
from src.data.database import DatabaseManager
from src.data.websocket import WebSocketManager
from src.data.market_data import MarketDataManager
from src.data.trading_signals import TradingSignalManager
from src.data.models import Position, PositionType, AccountBalance
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.main import Hummingbird
from src.data.models import Position
from src.technical.analysis import TechnicalAnalysis
from src.technical.market_structure import MarketStructureAnalyzer
from src.data.market_data import MarketData
from src.llm.analyzer import LLMAnalyzer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hummingbird instance with default settings
try:
    # Use relative path for config as that's what main.py expects
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config", "config.yaml")
    hummingbird = Hummingbird(config_path)
    
    # Set default settings
    hummingbird.symbol = "BTC/USDT"  # Set default symbol
    hummingbird.trading_mode = "scalping"  # Set default mode
    
    # Initialize database and position manager
    db = DatabaseManager()
    hummingbird.db = db
    hummingbird._init_position_manager()
    
    # Initialize market data and technical analysis
    hummingbird.market_data = MarketData(hummingbird.config)
    hummingbird.technical_analysis = TechnicalAnalysis(hummingbird.config)
    hummingbird.market_analyzer = MarketStructureAnalyzer(hummingbird.config)
    
    # Update the LLM analyzer to use Gemini model
    hummingbird.llm_analyzer = LLMAnalyzer(
        model_config=hummingbird.config['llm'],
        model_name="gemini",
        technical_analysis=hummingbird.technical_analysis
    )
    
    # Set required attributes for LLM analyzer
    hummingbird.llm_analyzer.market_data = hummingbird.market_data
    hummingbird.llm_analyzer.market_analyzer = hummingbird.market_analyzer
    hummingbird.llm_analyzer.trading_mode = hummingbird.trading_mode
    
    # Set position manager for LLM analyzer
    hummingbird.llm_analyzer.set_position_manager(hummingbird.position_manager, hummingbird.db)
    
    # Initialize market data with default symbol
    initial_data = hummingbird.market_data.fetch_historical_data(
        hummingbird.symbol,
        hummingbird.config['trading']['modes'][hummingbird.trading_mode]['default_timeframe'],
        hummingbird.config['data']['historical_data_days']
    )
    if initial_data is None or initial_data.empty:
        raise ValueError("Failed to fetch initial market data")
    
    logger.info("Successfully initialized Hummingbird with Gemini model")
except Exception as e:
    logger.error(f"Error initializing Hummingbird: {str(e)}")
    raise

# Initialize managers
db_manager = DatabaseManager()
ws_manager = WebSocketManager()
market_data_manager = MarketDataManager()
signal_manager = TradingSignalManager()

# Initialize signal manager with required components
signal_manager.initialize(
    llm_analyzer=hummingbird.llm_analyzer,
    market_data=hummingbird.market_data,
    technical_analysis=hummingbird.technical_analysis
)

class SettingsUpdate(BaseModel):
    symbol: str
    trading_mode: str

class PositionHistoryResponse(BaseModel):
    symbol: str
    type: str
    entry_price: float
    exit_price: float
    pnl: float
    closed_reason: str
    duration: str
    created_at: datetime
    closed_at: Optional[datetime] = None

@app.get("/api/market-data")
async def get_market_data(symbol: str = Query(..., description="Trading pair symbol")):
    try:
        market_data = hummingbird.market_data
        # Get market data using the correct method
        df = market_data.get_market_data(symbol)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No market data available")
            
        # Calculate indicators
        df_with_indicators = hummingbird.technical_analysis.calculate_indicators(df)
        if df_with_indicators is None or df_with_indicators.empty:
            raise HTTPException(status_code=404, detail="No indicators available")
            
        # Get multi-timeframe volumes
        volumes = {
            '1h': float(df_with_indicators['volume'].tail(24).sum()),  # Last hour volume
            '4h': float(df_with_indicators['volume'].tail(96).sum()),  # Last 4 hours volume
            '1d': float(df_with_indicators['volume'].tail(576).sum()),  # Last day volume (assuming 2.5min candles)
        }
            
        # Get current data with enhanced technical indicators
        current_data = {
            "symbol": symbol,
            "current_price": float(df_with_indicators['close'].iloc[-1]),
            "price_change_24h": float(df_with_indicators['close'].pct_change(periods=24).iloc[-1] * 100),
            "volume_24h": float(df_with_indicators['volume'].sum()),
            "technical_indicators": {
                "RSI": float(df_with_indicators['RSI'].iloc[-1]),
                "MACD": {
                    "MACD": float(df_with_indicators['MACD'].iloc[-1]),
                    "Signal": float(df_with_indicators['MACD_Signal'].iloc[-1]),
                    "Histogram": float(df_with_indicators['MACD_Hist'].iloc[-1])
                },
                "EMA": {
                    "EMA8": float(df_with_indicators['EMA_8'].iloc[-1]),
                    "EMA21": float(df_with_indicators['EMA_21'].iloc[-1]),
                    "EMA50": float(df_with_indicators['EMA_50'].iloc[-1]),
                    "EMA200": float(df_with_indicators['EMA_200'].iloc[-1])
                },
                "BB": {
                    "Upper": float(df_with_indicators['BB_upper'].iloc[-1]),
                    "Middle": float(df_with_indicators['BB_middle'].iloc[-1]),
                    "Lower": float(df_with_indicators['BB_lower'].iloc[-1])
                },
                "Volume": volumes,
                "ATR": float(df_with_indicators['ATR'].iloc[-1])
            },
            "trading_mode": hummingbird.trading_mode
        }
        return current_data
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    try:
        positions = hummingbird.position_manager.get_active_positions()
        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_signals(symbol: str = Query(..., description="Trading pair symbol")):
    try:
        signal = hummingbird.llm_analyzer.generate_signal(symbol)
        return signal
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    try:
        hummingbird.symbol = settings.symbol
        hummingbird.trading_mode = settings.trading_mode
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions/history")
async def get_position_history(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbol: Optional[str] = None,
    type: Optional[str] = None
) -> List[PositionHistoryResponse]:
    try:
        positions = db_manager.get_closed_positions(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            type=type
        )
        return positions
    except Exception as e:
        logger.error(f"Error fetching position history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def serialize_datetime(obj):
    """Helper function to serialize datetime objects to ISO format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def format_position(position: Position) -> dict:
    """Helper function to format a Position object into a JSON-serializable dictionary"""
    try:
        return {
            'id': position.id,
            'symbol': position.symbol,
            'position_type': position.position_type.value,
            'status': position.status.value,
            'entry_price': float(position.entry_price),
            'current_price': float(position.current_price) if position.current_price is not None else None,
            'stop_loss': float(position.stop_loss) if position.stop_loss is not None else None,
            'take_profit': float(position.take_profit) if position.take_profit is not None else None,
            'size': float(position.size),
            'pnl': float(position.pnl) if position.pnl is not None else 0.0,
            'created_at': position.created_at.isoformat() if position.created_at is not None else None,
            'closed_at': position.closed_at.isoformat() if position.closed_at is not None else None
        }
    except Exception as e:
        logger.error(f"Error formatting position {position.id}: {str(e)}")
        return None

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(id(websocket))
    await ws_manager.connect(websocket, client_id)
    logger.info(f"New client {client_id} connected")
    
    try:
        while True:
            try:
                # Function to prepare and send data
                async def send_dashboard_update():
                    # Get current positions and format them
                    positions = hummingbird.position_manager.get_active_positions()
                    formatted_positions = [format_position(pos) for pos in positions if pos is not None]
                    
                    # Get market data and signal
                    market_dataSend = hummingbird.market_data.get_market_data(hummingbird.symbol)
                    market_data = hummingbird.analyze_market(hummingbird.symbol)
                    signal = market_data.get('signal') if market_data is not None else None
                    
                    # Get the current trading mode configuration
                    mode_config = hummingbird.config['trading']['modes'][hummingbird.trading_mode]
                    
                    # Calculate indicators and prepare market data
                    if market_dataSend is not None:
                        df_with_indicators = hummingbird.technical_analysis.calculate_indicators(market_dataSend)
                        
                        # Calculate volumes for different timeframes
                        current_time = pd.Timestamp.now()
                        df_with_indicators['timestamp'] = pd.to_datetime(df_with_indicators.index)
                        
                        # Safely get technical indicators with default values if not available
                        try:
                            rsi = float(df_with_indicators['RSI'].iloc[-1])
                        except (KeyError, IndexError, ValueError):
                            rsi = None
                            
                        try:
                            macd = float(df_with_indicators['MACD'].iloc[-1])
                        except (KeyError, IndexError, ValueError):
                            macd = None
                            
                        try:
                            ema_8 = float(df_with_indicators['EMA_8'].iloc[-1])
                        except (KeyError, IndexError, ValueError):
                            ema_8 = None
                            
                        try:
                            sma_8 = float(df_with_indicators['SMA_8'].iloc[-1])
                        except (KeyError, IndexError, ValueError):
                            sma_8 = None
                        
                        # Calculate volumes with proper error handling
                        try:
                            volume_4h = float(df_with_indicators[
                                df_with_indicators['timestamp'] >= current_time - pd.Timedelta(hours=4)
                            ]['volume'].sum())
                        except:
                            volume_4h = 0.0
                            
                        try:
                            volume_1h = float(df_with_indicators[
                                df_with_indicators['timestamp'] >= current_time - pd.Timedelta(hours=1)
                            ]['volume'].sum())
                        except:
                            volume_1h = 0.0
                            
                        try:
                            volume_15m = float(df_with_indicators[
                                df_with_indicators['timestamp'] >= current_time - pd.Timedelta(minutes=15)
                            ]['volume'].sum())
                        except:
                            volume_15m = 0.0
                        
                        current_data = {
                            "symbol": hummingbird.symbol,
                            "current_price": float(df_with_indicators['close'].iloc[-1]),
                            "price_change_24h": float(df_with_indicators['close'].pct_change(periods=24).iloc[-1] * 100),
                            "volume_24h": float(df_with_indicators['volume'].sum()),
                            "volume_4h": volume_4h,
                            "volume_1h": volume_1h,
                            "volume_15m": volume_15m,
                            "rsi": rsi,
                            "macd": macd,
                            "ema": ema_8,
                            "sma": sma_8,
                            "trading_mode": hummingbird.trading_mode
                        }
                    else:
                        current_data = {
                            "symbol": hummingbird.symbol,
                            "current_price": 0.0,
                            "price_change_24h": 0.0,
                            "volume_24h": 0.0,
                            "volume_4h": 0.0,
                            "volume_1h": 0.0,
                            "volume_15m": 0.0,
                            "rsi": None,
                            "macd": None,
                            "ema": None,
                            "sma": None,
                            "trading_mode": hummingbird.trading_mode
                        }
                    
                    # Format signal data
                    formatted_signal = {
                        "signal": signal.get('signal', 'HOLD') if signal else 'HOLD',
                        "confidence": signal.get('confidence', 0.0) if signal else 0.0,
                        "entry_price": signal.get('entry_price', 0.0) if signal else 0.0,
                        "stop_loss": signal.get('stop_loss', 0.0) if signal else 0.0,
                        "take_profit": signal.get('take_profit', 0.0) if signal else 0.0,
                        "reason": signal.get('reasoning', 'No reasoning provided') if signal else 'No signal available',
                        "symbol": hummingbird.symbol,
                        "timeframe": mode_config['smc_analysis']['primary_timeframe'],
                        "timestamp": datetime.now().isoformat(),
                        "position_management": signal.get('position_management', {
                            "action": "MAINTAIN",
                            "stop_loss_adjustment": "None",
                            "take_profit_adjustment": "None",
                            "risk_reward_ratio": 1.0
                        }) if signal else {
                            "action": "MAINTAIN",
                            "stop_loss_adjustment": "None",
                            "take_profit_adjustment": "None",
                            "risk_reward_ratio": 1.0
                        }
                    }
                    
                    message = {
                        "type": "dashboard_update",
                        "data": {
                            "positions": formatted_positions,
                            "market_data": current_data,
                            "signal": formatted_signal
                        }
                    }
                    
                    await websocket.send_json(message)
                
                # Send initial data immediately upon connection
                await send_dashboard_update()
                
                # Handle incoming messages (including ping)
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                    if data.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
                except asyncio.TimeoutError:
                    # No message received, continue with updates
                    pass
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}")
                    if isinstance(e, WebSocketDisconnect):
                        raise
                
                # Send regular updates
                await send_dashboard_update()
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop for client {client_id}: {str(e)}")
                await asyncio.sleep(10)
                continue
            
    except Exception as e:
        logger.error(f"Fatal WebSocket error for client {client_id}: {str(e)}")
    finally:
        await ws_manager.disconnect(client_id)
        logger.info(f"Client {client_id} connection cleaned up")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 