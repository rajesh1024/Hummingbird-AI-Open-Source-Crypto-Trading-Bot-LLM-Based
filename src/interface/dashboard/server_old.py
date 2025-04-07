# from fastapi import FastAPI, WebSocket, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.websockets import WebSocketDisconnect
# from pydantic import BaseModel
# import asyncio
# import json
# from typing import Dict, List, Optional, Any
# from datetime import datetime, timedelta
# import os
# import sys
# from pathlib import Path
# import logging
# import websockets
# from src.data.database import DatabaseManager
# from src.data.websocket import WebSocketManager
# from src.data.market_data import MarketDataManager
# from src.data.trading_signals import TradingSignalManager
# from src.data.models import Position, PositionType, AccountBalance
# import pandas as pd
# from src.data.symbol_manager import SymbolManager

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Add the project root to Python path
# project_root = str(Path(__file__).parent.parent.parent)
# sys.path.append(project_root)

# from src.main import Hummingbird
# from src.data.models import Position
# from src.technical.analysis import TechnicalAnalysis
# from src.technical.market_structure import MarketStructureAnalyzer
# from src.data.market_data import MarketData
# from src.llm.analyzer import LLMAnalyzer
# from src.interface.dashboard.routes import router as dashboard_router

# app = FastAPI()

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, replace with specific origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize Hummingbird instance
# hummingbird = None
# symbol_manager = None
# signal_manager = None

# # Add signal cache at the top with other global variables
# signal_cache = {}

# def init_hummingbird():
#     """Initialize Hummingbird and its components"""
#     global hummingbird, symbol_manager, signal_manager
    
#     try:
#         # Initialize Hummingbird with config
#         hummingbird = Hummingbird("../../../config/config.yaml")
        
#         # Set default symbol and trading mode
#         hummingbird.symbol = "BTC/USDT"
#         hummingbird.trading_mode = "scalping"
        
#         # Initialize market data and technical analysis
#         hummingbird.market_data = MarketData(hummingbird.config)
#         hummingbird.technical_analysis = TechnicalAnalysis(hummingbird.config)
#         hummingbird.market_analyzer = MarketStructureAnalyzer(hummingbird.config)
        
#         # Initialize LLM analyzer with all required components
#         hummingbird.llm_analyzer = LLMAnalyzer(
#             model_config=hummingbird.config['llm'],
#             model_name="gemini",
#             technical_analysis=hummingbird.technical_analysis
#         )
        
#         # Set required attributes for LLM analyzer
#         hummingbird.llm_analyzer.market_data = hummingbird.market_data
#         hummingbird.llm_analyzer.market_analyzer = hummingbird.market_analyzer
#         hummingbird.llm_analyzer.trading_mode = hummingbird.trading_mode
        
#         # Initialize symbol manager with Hummingbird instance
#         symbol_manager = SymbolManager(hummingbird)
        
#         # Verify symbol manager initialization
#         if not symbol_manager.position_manager:
#             logger.error("Symbol manager failed to initialize position manager")
#             return False
            
#         # Initialize signal manager with all components
#         signal_manager = TradingSignalManager()
#         signal_manager.initialize(
#             llm_analyzer=hummingbird.llm_analyzer,
#             market_data=hummingbird.market_data,
#             technical_analysis=hummingbird.technical_analysis
#         )
        
#         logger.info("Successfully initialized all Hummingbird components")
#         return True
        
#     except Exception as e:
#         logger.error(f"Error initializing Hummingbird: {str(e)}")
#         return False

# # Initialize components
# init_success = init_hummingbird()
# if not init_success:
#     raise ValueError("Failed to initialize Hummingbird components")

# # Include dashboard routes with /api prefix
# app.include_router(dashboard_router, prefix="/api")

# # Initialize managers
# db_manager = DatabaseManager()
# ws_manager = WebSocketManager()
# market_data_manager = MarketDataManager()

# class SettingsUpdate(BaseModel):
#     symbol: str
#     trading_mode: str

# class PositionHistoryResponse(BaseModel):
#     symbol: str
#     type: str
#     entry_price: float
#     exit_price: float
#     pnl: float
#     closed_reason: str
#     duration: str
#     created_at: datetime
#     closed_at: Optional[datetime] = None

# class SymbolRequest(BaseModel):
#     symbol: str

# @app.post("/api/symbols/start")
# async def start_symbol(request: SymbolRequest):
#     """Start analysis for a symbol"""
#     try:
#         if not hummingbird.position_manager:
#             logger.error("Position manager not initialized")
#             raise HTTPException(status_code=500, detail="Position manager not initialized")
            
#         # Start symbol analysis
#         success = await symbol_manager.start_symbol(request.symbol)
#         if success:
#             return {"status": "success", "message": f"Started analysis for {request.symbol}"}
#         else:
#             raise HTTPException(status_code=500, detail=f"Failed to start analysis for {request.symbol}")
            
#     except Exception as e:
#         logger.error(f"Error starting symbol analysis: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/symbols/stop")
# async def stop_symbol(request: SymbolRequest):
#     """Stop analysis for a symbol"""
#     try:
#         success = await symbol_manager.stop_symbol(request.symbol)
#         if success:
#             return {"status": "success", "message": f"Stopped analysis for {request.symbol}"}
#         else:
#             raise HTTPException(status_code=500, detail=f"Failed to stop analysis for {request.symbol}")
#     except Exception as e:
#         logger.error(f"Error stopping symbol analysis: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/symbols/status")
# async def get_symbol_status(symbol: Optional[str] = None):
#     """Get status of one or all symbols"""
#     try:
#         if symbol:
#             status = symbol_manager.get_symbol_status(symbol)
#             return {symbol: status}
#         else:
#             return symbol_manager.get_all_symbol_status()
#     except Exception as e:
#         logger.error(f"Error getting symbol status: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/market-data")
# async def get_market_data(symbol: str = Query(..., description="Trading pair symbol")):
#     try:
#         market_data = hummingbird.market_data
#         # Get market data using the correct method
#         df = market_data.get_market_data(symbol)
#         if df is None or df.empty:
#             raise HTTPException(status_code=404, detail="No market data available")
            
#         # Calculate indicators
#         df_with_indicators = hummingbird.technical_analysis.calculate_indicators(df)
#         if df_with_indicators is None or df_with_indicators.empty:
#             raise HTTPException(status_code=404, detail="No indicators available")
            
#         # Get multi-timeframe volumes
#         volumes = {
#             '1h': float(df_with_indicators['volume'].tail(24).sum()),  # Last hour volume
#             '4h': float(df_with_indicators['volume'].tail(96).sum()),  # Last 4 hours volume
#             '1d': float(df_with_indicators['volume'].tail(576).sum()),  # Last day volume (assuming 2.5min candles)
#         }
#          # "RSI": float(df_with_indicators['RSI'].iloc[-1]),
#                 # "MACD": {
#                 #     "MACD": float(df_with_indicators['MACD'].iloc[-1]),
#                 #     "Signal": float(df_with_indicators['MACD_Signal'].iloc[-1]),
#                 #     "Histogram": float(df_with_indicators['MACD_Hist'].iloc[-1])
#                 # },
#                 # "EMA": {
#                 #     "EMA8": float(df_with_indicators['EMA_8'].iloc[-1]),
#                 #     "EMA21": float(df_with_indicators['EMA_21'].iloc[-1]),
#                 #     "EMA50": float(df_with_indicators['EMA_50'].iloc[-1])
#                 # },
#                 # "BB": {
#                 #     "Upper": float(df_with_indicators['BB_Upper'].iloc[-1]),
#                 #     "Middle": float(df_with_indicators['BB_Middle'].iloc[-1]),
#                 #     "Lower": float(df_with_indicators['BB_Lower'].iloc[-1])
#                 # },
#                 # "Volume": volumes,
#                 # "ATR": 0.0  # Default value since ATR is not calculated
            
#         # Get current data with enhanced technical indicators
#         current_data = {
#             "symbol": symbol,
#             "current_price": float(df_with_indicators['close'].iloc[-1]),
#             "price_change_24h": float(df_with_indicators['close'].pct_change(periods=24).iloc[-1] * 100),
#             "volume_24h": float(df_with_indicators['volume'].sum()),
#             "technical_indicators": { 
#             },
#             "trading_mode": hummingbird.trading_mode
#         }
#         return current_data
#     except Exception as e:
#         logger.error(f"Error getting market data: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/positions")
# async def get_positions():
#     try:
#         positions = hummingbird.position_manager.get_active_positions()
#         return positions
#     except Exception as e:
#         logger.error(f"Error getting positions: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/signals")
# async def get_signals(symbol: str = Query(..., description="Trading pair symbol")):
#     try:
#         signal = hummingbird.llm_analyzer.generate_signal(symbol)
#         return signal
#     except Exception as e:
#         logger.error(f"Error getting signals: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/settings")
# async def update_settings(settings: SettingsUpdate):
#     try:
#         hummingbird.symbol = settings.symbol
#         hummingbird.trading_mode = settings.trading_mode
#         return {"status": "success", "message": "Settings updated successfully"}
#     except Exception as e:
#         logger.error(f"Error updating settings: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/positions/history")
# async def get_position_history(
#     start_date: Optional[datetime] = None,
#     end_date: Optional[datetime] = None,
#     symbol: Optional[str] = None,
#     type: Optional[str] = None
# ) -> List[PositionHistoryResponse]:
#     try:
#         positions = db_manager.get_closed_positions(
#             start_date=start_date,
#             end_date=end_date,
#             symbol=symbol,
#             type=type
#         )
#         return positions
#     except Exception as e:
#         logger.error(f"Error fetching position history: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# def serialize_datetime(obj):
#     """Helper function to serialize datetime objects to ISO format"""
#     if isinstance(obj, datetime):
#         return obj.isoformat()
#     raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# def format_position(position: Position) -> dict:
#     """Helper function to format a Position object into a JSON-serializable dictionary"""
#     try:
#         return {
#             'id': position.id,
#             'symbol': position.symbol,
#             'position_type': position.position_type.value,
#             'status': position.status.value,
#             'entry_price': float(position.entry_price),
#             'current_price': float(position.current_price) if position.current_price is not None else None,
#             'stop_loss': float(position.stop_loss) if position.stop_loss is not None else None,
#             'take_profit': float(position.take_profit) if position.take_profit is not None else None,
#             'size': float(position.size),
#             'pnl': float(position.pnl) if position.pnl is not None else 0.0,
#             'created_at': position.created_at.isoformat() if position.created_at is not None else None,
#             'closed_at': position.closed_at.isoformat() if position.closed_at is not None else None
#         }
#     except Exception as e:
#         logger.error(f"Error formatting position {position.id}: {str(e)}")
#         return None

# @app.websocket("/ws/dashboard")
# async def websocket_endpoint(websocket: WebSocket, symbol: str = Query(...)):
#     await websocket.accept()
#     logger.info(f"WebSocket connection established for symbol {symbol}")
    
#     try:
#         while True:
#             try:
#                 # Get current market data
#                 market_data = await get_market_data(symbol)
#                 if not market_data:
#                     logger.warning(f"No market data available for symbol {symbol}")
#                     market_data = {}
                
#                 # Get current positions for the specific symbol
#                 positions = await symbol_manager._get_active_positions(symbol)
#                 if not positions:
#                     logger.warning(f"No positions available for symbol {symbol}")
#                     positions = []
                
#                 # Get latest signal and check if it's different from cache
#                 current_signal = await get_signals(symbol)
#                 if not current_signal:
#                     logger.warning(f"No signals available for symbol {symbol}")
#                     current_signal = {}
                
#                 # Only send update if signal has changed or it's the first connection
#                 if symbol not in signal_cache or signal_cache[symbol] != current_signal:
#                     signal_cache[symbol] = current_signal
                    
#                     # Send combined data
#                     await websocket.send_json({
#                         "type": "update",
#                         "data": {
#                             "market_data": market_data,
#                             "positions": positions,
#                             "signal": current_signal
#                         }
#                     })
                
#                 # Wait for 1 second before next update
#                 await asyncio.sleep(1)
                
#             except Exception as e:
#                 logger.error(f"Error processing data for symbol {symbol}: {str(e)}")
#                 await websocket.send_json({
#                     "type": "error",
#                     "message": f"Error processing data: {str(e)}"
#                 })
#                 await asyncio.sleep(1)
                
#     except WebSocketDisconnect:
#         logger.info(f"WebSocket disconnected for symbol {symbol}")
#         # Clean up signal cache for this symbol if no other connections
#         if symbol in signal_cache:
#             del signal_cache[symbol]
#     except Exception as e:
#         logger.error(f"WebSocket error for symbol {symbol}: {str(e)}")
#         try:
#             await websocket.close()
#         except:
#             pass

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000) 