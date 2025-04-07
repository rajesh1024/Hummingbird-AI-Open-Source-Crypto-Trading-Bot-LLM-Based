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
from src.data.symbol_manager import SymbolManager
import traceback
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Reduce access logs

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.main import Hummingbird
from src.data.models import Position
from src.technical.analysis import TechnicalAnalysis
from src.technical.market_structure import MarketStructureAnalyzer
from src.data.market_data import MarketData
from src.llm.analyzer import LLMAnalyzer
from src.interface.dashboard.routes import router as dashboard_router

app = FastAPI()

# Add CORS middleware - Very permissive for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hummingbird instance
hummingbird = None
symbol_manager = None
signal_manager = None

def init_hummingbird():
    """Initialize Hummingbird and its components"""
    global hummingbird, symbol_manager, signal_manager

    try:
        hummingbird = Hummingbird("../../../config/config.yaml")
        hummingbird.symbol = None
        hummingbird.trading_mode = "scalping"
        hummingbird.market_data = MarketData(hummingbird.config)
        hummingbird.technical_analysis = TechnicalAnalysis(hummingbird.config)
        hummingbird.market_analyzer = MarketStructureAnalyzer(hummingbird.config)
        hummingbird.llm_analyzer = LLMAnalyzer(
            model_config=hummingbird.config['llm'],
            model_name="gemini",
            technical_analysis=hummingbird.technical_analysis
        )
        hummingbird.llm_analyzer.market_data = hummingbird.market_data
        hummingbird.llm_analyzer.market_analyzer = hummingbird.market_analyzer
        hummingbird.llm_analyzer.trading_mode = hummingbird.trading_mode
        symbol_manager = SymbolManager(hummingbird)
        if not symbol_manager.position_manager:
            logger.error("Symbol manager failed to initialize position manager")
            return False
        signal_manager = TradingSignalManager()
        signal_manager.initialize(
            llm_analyzer=hummingbird.llm_analyzer,
            market_data=hummingbird.market_data,
            technical_analysis=hummingbird.technical_analysis
        )
        logger.info("Successfully initialized all Hummingbird components")
        return True
    except Exception as e:
        logger.error(f"Error initializing Hummingbird: {str(e)}")
        return False

# Initialize components
init_success = init_hummingbird()
if not init_success:
    raise ValueError("Failed to initialize Hummingbird components")

# Include dashboard routes with /api prefix
app.include_router(dashboard_router, prefix="/api")

# Initialize managers
db_manager = DatabaseManager()
ws_manager = WebSocketManager()
market_data_manager = MarketDataManager()

class SettingsUpdate(BaseModel):
    symbol: str
    trading_mode: str

class PositionHistoryResponse(BaseModel):
    data: List[Dict]
    total: int
    page: int
    totalPages: int
    stats: Dict[str, float]

class SymbolRequest(BaseModel):
    symbol: str

# Centralized cache
global_cache: Dict[str, Dict[str, Any]] = {}
# Track running symbols with their status
running_symbols: Dict[str, bool] = {}

async def _generate_and_cache_data(symbol: str):
    try:
        # Force refresh market data
        market_data = await get_market_data_for_symbol(symbol)
        if not market_data:
            logger.error(f"No market data for {symbol} for caching.")
            return False
            
        # Ensure symbol is set in market data
        market_data['symbol'] = symbol
            
        # Get fresh positions with current prices
        positions = await symbol_manager._get_active_positions(symbol)
        if positions:
            # Update current prices in positions
            for position in positions:
                position['symbol'] = symbol  # Ensure symbol is set in position
                position['current_price'] = market_data['current_price']
                if position['entry_price'] and position['current_price']:
                    # Calculate PnL as percentage
                    if position['position_type'] == 'LONG':
                        position['pnl'] = ((position['current_price'] - position['entry_price']) / position['entry_price']) * 100
                    else:  # SHORT
                        position['pnl'] = ((position['entry_price'] - position['current_price']) / position['entry_price']) * 100
        
        # Prepare market context with fresh data
        market_context = {
            'symbol': symbol,
            'current_price': market_data.get('current_price'),
            'price_change_24h': market_data.get('price_change_24h'),
            'volume_24h': market_data.get('volume_24h'),
            'technical_indicators': market_data.get('technical_indicators', {}),
            'active_positions': positions,
            'trading_mode': hummingbird.trading_mode,
            'market_data': market_data
        }

        # Validate required fields
        required_fields = ['current_price', 'technical_indicators']
        missing_fields = [field for field in required_fields if market_context.get(field) is None]
        if missing_fields:
            logger.error(f"Missing fields for signal generation: {missing_fields}")
            return False

        # Generate fresh signal
        try:
            if asyncio.iscoroutinefunction(hummingbird.llm_analyzer.generate_signal):
                signal = await hummingbird.llm_analyzer.generate_signal(market_context)
            else:
                signal = hummingbird.llm_analyzer.generate_signal(market_context)

            # Ensure symbol is set in signal
            if signal:
                signal['symbol'] = symbol
        except Exception as e:
            logger.error(f"Error in signal generation for {symbol}: {str(e)}")
            signal = None
            
        # Get previous cache to check if data has changed
        previous_cache = global_cache.get(symbol, {})
        
        # Update cache with fresh data
        timestamp = datetime.now().isoformat()
        new_cache = {
            "market_data": market_data,
            "positions": positions or [],
            "signal": signal,
            "timestamp": timestamp,
            "symbol": symbol  # Ensure symbol is included at the top level
        }
        
        # Check if data has actually changed
        has_changes = False
        if not previous_cache:
            logger.info(f"Initial data for {symbol}")
            has_changes = True
        else:
            # Safely get previous values
            prev_market_data = previous_cache.get('market_data', {}) or {}
            prev_price = prev_market_data.get('current_price')
            new_price = market_data.get('current_price')
            
            prev_signal_data = previous_cache.get('signal') or {}
            prev_signal = prev_signal_data.get('signal') if isinstance(prev_signal_data, dict) else None
            new_signal = signal.get('signal') if signal else None
            
            # Log the comparison
            logger.debug(f"Comparing data for {symbol}:")
            logger.debug(f"Previous price: {prev_price}, New price: {new_price}")
            logger.debug(f"Previous signal: {prev_signal}, New signal: {new_signal}")
            
            # Compare values
            price_changed = prev_price != new_price
            signal_changed = prev_signal != new_signal
            positions_changed = positions != previous_cache.get('positions')
            
            if price_changed or signal_changed or positions_changed:
                changes = []
                if price_changed:
                    changes.append(f"Price: {prev_price} -> {new_price}")
                if signal_changed:
                    changes.append(f"Signal: {prev_signal} -> {new_signal}")
                if positions_changed:
                    changes.append("Positions updated")
                
                logger.info(f"Changes detected for {symbol}: {', '.join(changes)}")
                has_changes = True
        
        if has_changes:
            global_cache[symbol] = new_cache
            logger.info(f"Updated cache for {symbol} with new data")
            return True
        else:
            logger.debug(f"No significant changes for {symbol}")
            return False

    except Exception as e:
        logger.error(f"Error generating/caching data for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return False

async def start_data_generation(symbol: str):
    """Start symbol analysis"""
    global running_symbols
    if symbol not in running_symbols or not running_symbols[symbol]:
        logger.info(f"Starting data generation for {symbol}")
        running_symbols[symbol] = True
        # Generate initial data immediately
        try:
            await _generate_and_cache_data(symbol)
        except Exception as e:
            logger.error(f"Error generating initial data for {symbol}: {e}")

@app.post("/api/symbols/start")
async def start_symbol(request: SymbolRequest):
    symbol = request.symbol
    try:
        if not hummingbird.position_manager:
            logger.error("Position manager not initialized")
            raise HTTPException(status_code=500, detail="Position manager not initialized")
        success = await symbol_manager.start_symbol(symbol)
        if success:
            await start_data_generation(symbol)
            return {"status": "success", "message": f"Started analysis for {symbol}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start analysis for {symbol}")
    except Exception as e:
        logger.error(f"Error starting symbol: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/symbols/stop")
async def stop_symbol(request: SymbolRequest):
    symbol = request.symbol
    global running_symbols, global_cache
    try:
        success = await symbol_manager.stop_symbol(symbol)
        if success:
            if symbol in running_symbols:
                running_symbols[symbol] = False
                logger.info(f"Stopped data generation for {symbol}")
            if symbol in global_cache:
                del global_cache[symbol]
                logger.info(f"Removed cached data for {symbol}")
            return {"status": "success", "message": f"Stopped analysis for {symbol}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to stop analysis for {symbol}")
    except Exception as e:
        logger.error(f"Error stopping symbol: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/symbols/status")
async def get_symbol_status(symbol: Optional[str] = None):
    try:
        if symbol:
            status = symbol_manager.get_symbol_status(symbol)
            status['data_generation'] = running_symbols.get(symbol, False)
            return {symbol: status}
        else:
            all_status = symbol_manager.get_all_symbol_status()
            for sym in all_status:
                all_status[sym]['data_generation'] = running_symbols.get(sym, False)
            return all_status
    except Exception as e:
        logger.error(f"Error getting symbol status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_market_data_for_symbol(symbol: str):
    try:
        market_data = hummingbird.market_data
        df = market_data.get_market_data(symbol)
        if df is None or df.empty:
            logger.warning(f"No market data for symbol {symbol}.")
            return None
        df_with_indicators = hummingbird.technical_analysis.calculate_indicators(df)
        if df_with_indicators is None or df_with_indicators.empty:
            logger.warning(f"No indicators for symbol {symbol}.")
            return None
        latest_data = df_with_indicators.iloc[-1]
        def safe_float(value):
            try:
                return float(value) if pd.notna(value) else 0.0
            except (ValueError, TypeError):
                return 0.0
        current_data = {
            "symbol": symbol,
            "current_price": safe_float(latest_data.get('close')),
            "price_change_24h": safe_float(df_with_indicators['close'].pct_change(periods=24).iloc[-1] * 100),
            "volume_24h": safe_float(df_with_indicators['volume'].sum()),
            "technical_indicators": {
                "RSI": safe_float(latest_data.get('RSI')),
                "MACD": {
                    "MACD": safe_float(latest_data.get('MACD')),
                    "Signal": safe_float(latest_data.get('MACD_Signal')),
                    "Histogram": safe_float(latest_data.get('MACD_Hist'))
                },
                "EMA": {
                    "EMA8": safe_float(latest_data.get('EMA_8')),
                    "EMA21": safe_float(latest_data.get('EMA_21')),
                    "EMA50": safe_float(latest_data.get('EMA_50'))
                },
                "BB": {
                    "Upper": safe_float(latest_data.get('BB_upper')),
                    "Middle": safe_float(latest_data.get('BB_middle')),
                    "Lower": safe_float(latest_data.get('BB_lower'))
                },
                "Volume": {
                    '1h': safe_float(df_with_indicators['volume'].tail(24).sum()),
                    '4h': safe_float(df_with_indicators['volume'].tail(96).sum()),
                    '1d': safe_float(df_with_indicators['volume'].tail(576).sum()),
                },
                "ATR": safe_float(latest_data.get('ATR'))
            },
            "trading_mode": hummingbird.trading_mode,
            "timestamp": datetime.now().isoformat()
        }
        required_fields = ["current_price", "price_change_24h", "volume_24h", "technical_indicators"]
        for field in required_fields:
            if current_data.get(field) is None:
                logger.error(f"Missing field {field} in market data for {symbol}.")
                return None
        return current_data
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return None

@app.get("/api/market-data")
async def get_market_data(symbol: str = Query(..., description="Trading pair symbol")):
    return await get_market_data_for_symbol(symbol)

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
    if symbol in global_cache and "signal" in global_cache[symbol]:
        return global_cache[symbol]["signal"]
    else:
        raise HTTPException(status_code=404, detail=f"No signal for {symbol}.")

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    try:
        hummingbird.symbol = settings.symbol
        hummingbird.trading_mode = settings.trading_mode
        return {"status": "success", "message": "Settings updated."}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions/history", response_model=PositionHistoryResponse)
async def get_position_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbol: Optional[str] = None,
    type: Optional[str] = None
):
    try:
        logger.info(f"Fetching position history with params: page={page}, limit={limit}, start_date={start_date}, end_date={end_date}, symbol={symbol}, type={type}")
        
        # Get all positions with filters
        positions = db_manager.get_closed_positions(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            type=type
        )
        
        logger.info(f"Found {len(positions)} positions")
        
        # Debug log the first few positions
        if positions:
            logger.info(f"Sample position data: {json.dumps(positions[0], default=str)}")
        
        # Calculate statistics from all positions
        total_positions = len(positions)
        logger.info(f"Total positions: {total_positions}")
        
        # Calculate total PnL and winning trades
        total_pnl = 0
        winning_trades = 0
        for pos in positions:
            try:
                pnl = float(pos.get('pnl', 0))
                total_pnl += pnl
                if pnl >= 0:
                    winning_trades += 1
                logger.debug(f"Position PnL: {pnl}, Running total: {total_pnl}, Winning trades: {winning_trades}")
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing PnL for position: {pos.get('pnl')}, Error: {str(e)}")
        
        logger.info(f"Total PnL: {total_pnl}, Winning trades: {winning_trades}")
        
        # Calculate win rate and average PnL
        win_rate = (winning_trades / total_positions * 100) if total_positions > 0 else 0
        avg_pnl_per_trade = total_pnl / total_positions if total_positions > 0 else 0
        
        logger.info(f"Win rate: {win_rate}%, Avg PnL per trade: {avg_pnl_per_trade}")
        
        # Calculate pagination
        total_pages = math.ceil(total_positions / limit)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Get paginated positions
        paginated_positions = positions[start_idx:end_idx]
        
        # Format positions for response
        formatted_positions = []
        for pos in paginated_positions:
            formatted_pos = {
                'id': str(pos.get('id', '')),
                'symbol': pos.get('symbol', ''),
                'type': pos.get('type', ''),
                'entry_price': float(pos.get('entry_price', 0)),
                'exit_price': float(pos.get('exit_price', 0)),
                'pnl': float(pos.get('pnl', 0)),
                'closed_reason': pos.get('closed_reason', ''),
                'duration': pos.get('duration', ''),
                'closed_at': pos.get('closed_at', '').isoformat() if isinstance(pos.get('closed_at'), datetime) else pos.get('closed_at', '')
            }
            formatted_positions.append(formatted_pos)
        
        stats = {
            "totalTrades": total_positions,
            "winRate": round(win_rate, 1),
            "totalPnL": round(total_pnl, 2),
            "avgPnLPerTrade": round(avg_pnl_per_trade, 2)
        }
        
        response = {
            "data": formatted_positions,
            "total": total_positions,
            "page": page,
            "totalPages": total_pages,
            "stats": stats
        }
        
        logger.info(f"Returning response with stats: {json.dumps(stats, default=str)}")
        return response
        
    except Exception as e:
        logger.error(f"Error fetching position history: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.client_symbols: Dict[WebSocket, str] = {}  # Track which symbol each client is connected to

    async def connect(self, websocket: WebSocket, symbol: str):
        logger.info(f"Connecting new WebSocket for symbol {symbol}")
        
        # First disconnect from any existing symbol
        if websocket in self.client_symbols:
            old_symbol = self.client_symbols[websocket]
            logger.info(f"Client already connected to {old_symbol}, disconnecting first")
            await self.disconnect(websocket, old_symbol)
            # Stop data generation for old symbol if no more connections
            if old_symbol not in self.active_connections or not self.active_connections[old_symbol]:
                if old_symbol in running_symbols:
                    running_symbols[old_symbol] = False
                    logger.info(f"Stopped data generation for {old_symbol} - no active connections")
        
        # Accept the new connection
        await websocket.accept()
        
        # Add to active connections for new symbol
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
        self.client_symbols[websocket] = symbol
        
        # Start data generation for new symbol
        if symbol not in running_symbols or not running_symbols[symbol]:
            running_symbols[symbol] = True
            logger.info(f"Started data generation for {symbol}")
        
        logger.info(f"WebSocket connected for symbol {symbol}")
        await self.send_cached_data(websocket, symbol)

    async def disconnect(self, websocket: WebSocket, symbol: str):
        logger.info(f"Disconnecting WebSocket from symbol {symbol}")
        
        # Remove from active connections
        if symbol in self.active_connections and websocket in self.active_connections[symbol]:
            self.active_connections[symbol].remove(websocket)
            logger.info(f"Removed WebSocket from active connections for {symbol}")
            
            # Clean up empty symbol connections
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]
                logger.info(f"Removed empty connection list for {symbol}")
                # Stop data generation when no clients are connected
                if symbol in running_symbols:
                    running_symbols[symbol] = False
                    logger.info(f"Stopped data generation for {symbol} - no active connections")
        
        # Remove from client symbols tracking
        if websocket in self.client_symbols:
            del self.client_symbols[websocket]
            logger.info(f"Removed client symbol tracking for {symbol}")
        
        # Close the websocket if it's still open
        try:
            await websocket.close(code=1000, reason=f"Disconnected from {symbol}")
        except Exception as e:
            logger.error(f"Error closing WebSocket for {symbol}: {e}")

    async def send_cached_data(self, websocket: WebSocket, symbol: str):
        """Send initial cached data for a specific symbol"""
        if symbol in global_cache:
            try:
                cached_data = global_cache[symbol].copy()
                message = {
                    "type": "initial",
                    "symbol": symbol,
                    "data": cached_data
                }
                await websocket.send_json(message)
                logger.info(f"Sent initial data for {symbol} - Price: {cached_data['market_data']['current_price']}")
            except Exception as e:
                logger.error(f"Error sending cached data for {symbol}: {e}")
                await websocket.send_json({"type": "error", "message": "Failed to send initial data."})
                await websocket.close(code=1000, reason="Failed to send initial data.")
        else:
            await websocket.send_json({"type": "info", "message": f"No cached data yet for {symbol}."})

    async def broadcast_update(self, symbol: str, data: Dict):
        """Broadcast updated data only to clients subscribed to the specific symbol"""
        if symbol not in self.active_connections:
            return

        message = {
            "type": "update",
            "symbol": symbol,
            "data": data
        }
        disconnected = []

        # Only broadcast to clients that are currently subscribed to this symbol
        for websocket in self.active_connections[symbol]:
            try:
                current_symbol = self.client_symbols.get(websocket)
                if current_symbol == symbol:  # Double check the client is still subscribed to this symbol
                    await websocket.send_json(message)
                    logger.debug(f"Sent update to client for {symbol}")
                else:
                    logger.debug(f"Skipping update for client - subscribed to {current_symbol}, not {symbol}")
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket for {symbol}: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket, symbol)

# Initialize the connection manager
manager = ConnectionManager()

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket, symbol: str = Query(...)):
    logger.info(f"New WebSocket connection request for symbol: {symbol}")
    try:
        # Validate symbol format
        if not symbol or "/" not in symbol:
            logger.error(f"Invalid symbol format: {symbol}")
            await websocket.close(code=4000, reason="Invalid symbol format")
            return

        # Connect using the connection manager
        await manager.connect(websocket, symbol)

        # Start symbol analysis and data generation
        try:
            success = await symbol_manager.start_symbol(symbol)
            if success:
                await start_data_generation(symbol)
                logger.info(f"Started analysis for {symbol} on WebSocket connect")
            else:
                logger.warning(f"Failed to start symbol analysis for {symbol}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to start symbol analysis for {symbol}"
                })
                await websocket.close(code=1000, reason="Failed to start symbol analysis")
                return
        except Exception as e:
            logger.error(f"Error starting analysis for {symbol}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Error starting analysis: {str(e)}"
            })
            await websocket.close(code=1000, reason="Error starting analysis")
            return

        # Main message loop
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Verify the client is still connected to this symbol
                    current_symbol = manager.client_symbols.get(websocket)
                    if current_symbol != symbol:
                        logger.warning(f"Client symbol mismatch - expected {symbol}, got {current_symbol}")
                        break
                    
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "symbol": symbol
                        })
                        logger.debug(f"Received ping for {symbol}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received for {symbol}: {data}")
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for {symbol}")
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket loop for {symbol}: {e}")
                    break
        finally:
            await manager.disconnect(websocket, symbol)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during setup for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {str(e)}")
    finally:
        await manager.disconnect(websocket, symbol)

# Background task to periodically update cache
async def _periodic_cache_update():
    logger.info("Starting periodic cache update task")
    while True:
        try:
            active_symbols = [symbol for symbol, is_active in running_symbols.items() if is_active]
            logger.info(f"Active symbols for update: {active_symbols}")
            
            for symbol in active_symbols:
                try:
                    logger.info(f"Generating new data for {symbol}")
                    # Generate new data and check if it changed
                    data_changed = await _generate_and_cache_data(symbol)
                    
                    if data_changed and symbol in global_cache:
                        cached_data = global_cache[symbol]
                        
                        # Log the data being broadcast
                        logger.info(f"Broadcasting update for {symbol} - Price: {cached_data['market_data']['current_price']}, "
                                  f"24h Change: {cached_data['market_data']['price_change_24h']}%, "
                                  f"Signal: {cached_data['signal']['signal'] if cached_data['signal'] else 'None'}")
                        
                        # Broadcast the update to all connected clients
                        await manager.broadcast_update(symbol, cached_data)
                    else:
                        logger.debug(f"No updates needed for {symbol} - data unchanged")
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Wait before next update
            logger.debug("Waiting 5 seconds before next update cycle")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in periodic cache update loop: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            await asyncio.sleep(5)  # Still wait before retrying

@app.on_event("startup")
async def startup_event():
    logger.info("Starting FastAPI application")
    try:
        # Start the periodic update task
        update_task = asyncio.create_task(_periodic_cache_update())
        # Store the task to prevent garbage collection
        app.state.update_task = update_task
        logger.info("Successfully started periodic update task")
    except Exception as e:
        logger.error(f"Failed to start periodic update task: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)