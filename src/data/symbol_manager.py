from typing import Dict, Set, Optional, List
import asyncio
import logging
from enum import Enum
from datetime import datetime
import pandas as pd
from src.main import Hummingbird
from src.data.models import Position
from src.data.database import DatabaseManager
from src.technical.position_manager import PositionManager
from src.technical.market_structure import MarketStructureAnalyzer
import yaml
import os

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file"""
    try:
        # If no config path provided, find it relative to the project root
        if config_path is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to reach the project root
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config/config.yaml')
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

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
        }
    except Exception as e:
        logger.error(f"Error formatting position {position.id}: {str(e)}")
        return None

class SymbolState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"

class SymbolManager:
    def __init__(self, hummingbird, config: Dict = None):
        self.hummingbird = hummingbird
        self.config = config or load_config()
        self.position_manager = None
        self.db = None
        self.symbol_states = {}  # Track state of each symbol
        self.symbol_data = {}    # Store data for each symbol
        self.symbol_tasks = {}   # Store analysis tasks
        self.symbol_subscribers = {}  # Track subscribers for each symbol
        self.update_interval = self.config.get('monitoring', {}).get('interval', 10)
        
        # Initialize components in correct order
        self._init_database()
        self._validate_config()
        self._init_position_manager()
        
    def _validate_config(self):
        """Validate the configuration"""
        try:
            # Validate trading mode settings
            if 'trading' not in self.config:
                raise ValueError("Missing trading configuration")
            
            trading_config = self.config['trading']
            if 'modes' not in trading_config:
                raise ValueError("Missing trading modes configuration")
            
            # Validate scalping mode settings
            if 'scalping' not in trading_config['modes']:
                raise ValueError("Missing scalping mode configuration")
            
            scalping_config = trading_config['modes']['scalping']
            required_scalping_settings = [
                'timeframes',
                'default_timeframe',
                'smc_analysis'
            ]
            for setting in required_scalping_settings:
                if setting not in scalping_config:
                    raise ValueError(f"Missing required scalping setting: {setting}")
            
            # Validate SMC analysis settings
            smc_config = scalping_config['smc_analysis']
            required_smc_settings = [
                'primary_timeframe',
                'secondary_timeframes',
                'lookback_periods'
            ]
            for setting in required_smc_settings:
                if setting not in smc_config:
                    raise ValueError(f"Missing required SMC setting: {setting}")
            
            # Validate LLM settings
            if 'llm' not in self.config:
                raise ValueError("Missing LLM configuration")
            
            llm_config = self.config['llm']
            if 'models' not in llm_config:
                raise ValueError("Missing LLM models configuration")
            
            # Validate Gemini model settings
            if 'gemini' not in llm_config['models']:
                raise ValueError("Missing Gemini model configuration")
            
            gemini_config = llm_config['models']['gemini']
            required_gemini_settings = [
                'type',
                'name',
                'api_key',
                'max_tokens',
                'temperature'
            ]
            for setting in required_gemini_settings:
                if setting not in gemini_config:
                    raise ValueError(f"Missing required Gemini setting: {setting}")
            
            # Store validated configurations
            self.scalping_config = scalping_config
            self.smc_config = smc_config
            self.gemini_config = gemini_config
            
            logger.info("Configuration validation successful")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    def _init_database(self):
        """Initialize the database connection"""
        try:
            if not self.db:
                    self.db = DatabaseManager()
                    logger.info("Created new database connection for symbol manager")
                    return True
            return True
        except Exception as e:
                logger.error(f"Failed to create database connection: {str(e)}")
                self.db = None
                return False
        
    def _init_position_manager(self):
        """Initialize a dedicated position manager for symbol analysis"""
        try:
            if not self.db:
                logger.error("Cannot initialize position manager without database connection")
                self.position_manager = None
                return False
                
            # Create dedicated position manager with database session
            self.position_manager = PositionManager(
                db=self.db.get_session()
            )
            
            # Set the config from Hummingbird
            self.position_manager.config = self.hummingbird.config
            
            # Test the position manager
            active_positions = self.position_manager.get_active_positions()
            if active_positions is None:
                logger.error("Position manager test failed - get_active_positions returned None")
                self.position_manager = None
                return False
                
            logger.info(f"Symbol manager position manager initialized with {len(active_positions)} active positions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize symbol manager position manager: {str(e)}")
            self.position_manager = None
            return False
        
    def _verify_position_manager(self) -> bool:
        """Verify position manager is available and working"""
        if not self.position_manager:
            if hasattr(self.hummingbird, 'position_manager') and self.hummingbird.position_manager:
                self.position_manager = self.hummingbird.position_manager
                logger.info("Restored position manager reference")
                return True
            return False
        return True
        
    async def start_symbol(self, symbol: str) -> bool:
        """Start analysis for a symbol"""
        try:
            # Check if symbol is already running
            if symbol in self.symbol_states and self.symbol_states[symbol] == SymbolState.RUNNING:
                return True
            
            # Set state to starting
            self.symbol_states[symbol] = SymbolState.STARTING
            
            # Initialize symbol data if not exists
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {
                    "symbol": symbol,
                    "data": None,
                    "timestamp": None
                }
            
            # Start analysis task
            self.symbol_tasks[symbol] = asyncio.create_task(self._analyze_symbol(symbol))
            return True
        
        except Exception as e:
            logger.error(f"Error starting symbol {symbol}: {str(e)}")
            self.symbol_states[symbol] = SymbolState.ERROR
            return False

    async def stop_symbol(self, symbol: str) -> bool:
        """Stop analysis for a symbol"""
        if symbol not in self.symbol_states:
            return True

        try:
            # Cancel analysis task if exists
            if symbol in self.symbol_tasks:
                self.symbol_tasks[symbol].cancel()
                try:
                    await self.symbol_tasks[symbol]
                except asyncio.CancelledError:
                    pass
                del self.symbol_tasks[symbol]

            # Clean up state
            self.symbol_states[symbol] = SymbolState.STOPPED
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]
            return True
        except Exception as e:
            logger.error(f"Error stopping symbol {symbol}: {str(e)}")
            return False

    async def subscribe(self, client_id: str, symbol: str) -> bool:
        """Subscribe a client to a symbol's updates"""
        if symbol not in self.symbol_subscribers:
            self.symbol_subscribers[symbol] = set()

        self.symbol_subscribers[symbol].add(client_id)
        
        # Start symbol if not already running
        if symbol not in self.symbol_states or self.symbol_states[symbol] == SymbolState.STOPPED:
            return await self.start_symbol(symbol)
        return True

    async def unsubscribe(self, client_id: str, symbol: str):
        """Unsubscribe a client from a symbol's updates"""
        if symbol in self.symbol_subscribers:
            self.symbol_subscribers[symbol].discard(client_id)
            
            # Stop symbol if no subscribers
            if not self.symbol_subscribers[symbol]:
                await self.stop_symbol(symbol)
                del self.symbol_subscribers[symbol]

    def get_symbol_status(self, symbol: str) -> Dict:
        """Get current status of a symbol"""
        return {
            "symbol": symbol,
            "state": self.symbol_states.get(symbol, SymbolState.STOPPED).value,
            "subscribers": len(self.symbol_subscribers.get(symbol, set())),
            "last_update": self.symbol_data.get(symbol, {}).get("timestamp", None)
        }

    def get_all_symbols_status(self) -> Dict[str, Dict]:
        """Get status of all symbols"""
        return {symbol: self.get_symbol_status(symbol) for symbol in self.symbol_states}

    async def _analyze_symbol(self, symbol: str):
        """Background task to analyze a symbol"""
        try:
            self.symbol_states[symbol] = SymbolState.RUNNING
            
            while True:
                # Get market data
                market_data = await self._fetch_market_data(symbol)
                if market_data:
                    # Ensure all datetime objects in market_data are serialized
                    def serialize_datetime(obj):
                        if isinstance(obj, dict):
                            return {k: serialize_datetime(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [serialize_datetime(item) for item in obj]
                        elif hasattr(obj, 'isoformat'):
                            return obj.isoformat()
                        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                            return obj.isoformat()
                        return obj
                    
                    # Serialize all datetime objects in the market data
                    serialized_data = serialize_datetime(market_data)
                    
                    # Update symbol data with serialized data
                    self.symbol_data[symbol] = {
                        "symbol": symbol,
                        "data": serialized_data,
                        "timestamp": datetime.now().isoformat()
                    }
                
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Analysis task for {symbol} cancelled")
        except Exception as e:
            logger.error(f"Error in analysis task for {symbol}: {str(e)}")
            self.symbol_states[symbol] = SymbolState.ERROR
        finally:
            if symbol in self.symbol_states:
                self.symbol_states[symbol] = SymbolState.STOPPED

    async def _fetch_market_data(self, symbol: str) -> Dict:
        """Fetch market data for a symbol"""
        try:
            # Get current positions
            symbol_positions = await self._get_active_positions(symbol)
            
            # Get market data for all timeframes
            market_data = {}
            technical_indicators = {}
            market_structure = {}
            
            # Get timeframes from config
            primary_tf = self.smc_config['primary_timeframe']
            secondary_tfs = self.smc_config['secondary_timeframes']
            
            # Fetch and process data for each timeframe
            for timeframe in [primary_tf] + secondary_tfs:
                # Get lookback period for this timeframe
                lookback = self.smc_config['lookback_periods'].get(timeframe, 100)
                
                # Fetch market data
                df = self.hummingbird.market_data.get_market_data(symbol, timeframe, limit=lookback)
                if df is None or df.empty:
                    logger.error(f"No market data available for {symbol} on {timeframe}")
                    continue
                
                # Calculate indicators
                df = self.hummingbird.technical_analysis.calculate_indicators(df)
                if df is None or df.empty:
                    logger.error(f"Failed to calculate indicators for {symbol} on {timeframe}")
                    continue
                
                # Store processed data (only for internal use)
                market_data[timeframe] = df
                
                # Calculate technical indicators
                indicators = self._calculate_technical_indicators(df)
                technical_indicators[timeframe] = indicators
                
                # Analyze market structure
                structure = self._analyze_market_structure(df, timeframe)
                # Ensure all datetime objects in structure are serialized
                if 'smc_data' in structure:
                    for key in structure['smc_data']:
                        if isinstance(structure['smc_data'][key], list):
                            for item in structure['smc_data'][key]:
                                if 'timestamp' in item and hasattr(item['timestamp'], 'isoformat'):
                                    item['timestamp'] = item['timestamp'].isoformat()
                market_structure[timeframe] = structure
            
            # Get current price
            current_price = self.hummingbird.market_data.get_current_price(symbol)
            
            # Prepare market context
            market_context = self._prepare_market_context(
                symbol=symbol,
                current_price=current_price,
                market_data=market_data,
                technical_indicators=technical_indicators,
                market_structure=market_structure,
                positions=symbol_positions
            )
            
            # Generate signal using Gemini model
            signal_data = None
            if hasattr(self.hummingbird, 'llm_analyzer') and self.hummingbird.llm_analyzer is not None:
                signal_data = await self._generate_signal(market_context)
            
            # Return only essential data for WebSocket
            result = {
                "positions": symbol_positions,
                "signal": signal_data,
                "market_structure": market_structure,
                "technical_indicators": technical_indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators from dataframe"""
        try:
            indicators = {}
            
            # Map of required indicators to their column names in the dataframe
            indicator_columns = {
                'rsi': 'RSI',
                'macd': 'MACD',
                'macd_signal': 'MACD_Signal',
                'macd_hist': 'MACD_Hist',
                'ema_8': 'EMA_8',
                'ema_21': 'EMA_21',
                'ema_50': 'EMA_50',
                'sma_20': 'SMA_20',
                'bb_upper': 'BB_Upper',
                'bb_middle': 'BB_Middle',
                'bb_lower': 'BB_Lower',
                'volume_ma': 'Volume_MA',
                'volume_ratio': 'Volume_Ratio'
            }
            
            # Get required indicators from config
            required_indicators = self.config.get('technical', {}).get('indicators', {}).keys()
            
            for indicator in required_indicators:
                # Get the corresponding column name
                column_name = indicator_columns.get(indicator.lower())
                if column_name and column_name in df.columns:
                    indicators[indicator] = float(df[column_name].iloc[-1])
                else:
                    logger.warning(f"Missing indicator: {indicator} (column: {column_name})")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _analyze_market_structure(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze market structure and SMC patterns using the existing market structure analyzer"""
        try:
            # Initialize market structure analyzer if not exists
            if not hasattr(self, 'market_analyzer'):
                self.market_analyzer = MarketStructureAnalyzer(config=self.config)
            
            # Get market structure analysis
            structure = self.market_analyzer.analyze_market_structure(
                market_data=df,
                timeframe=timeframe
            )
            
            # If analysis failed, return default structure
            if structure is None:
                logger.warning(f"Market structure analysis failed for {timeframe}")
                return {
                    'market_structure': 'NEUTRAL',
                    'smc_data': {
                        'order_blocks': [],
                        'supply_zones': [],
                        'demand_zones': [],
                        'fair_value_gaps': [],
                        'liquidity_levels': [],
                        'smart_money_traps': []
                    }
                }
            
            # Ensure all required SMC components are present
            if 'smc_data' not in structure:
                structure['smc_data'] = {}
            
            # Add any missing SMC components
            required_components = [
                'order_blocks', 'supply_zones', 'demand_zones',
                'fair_value_gaps', 'liquidity_levels', 'smart_money_traps'
            ]
            
            for component in required_components:
                if component not in structure['smc_data']:
                    structure['smc_data'][component] = []
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {
                'market_structure': 'NEUTRAL',
                'smc_data': {
                    'order_blocks': [],
                    'supply_zones': [],
                    'demand_zones': [],
                    'fair_value_gaps': [],
                    'liquidity_levels': [],
                    'smart_money_traps': []
                }
            }

    def _prepare_market_context(self, symbol: str, current_price: float, market_data: Dict,
                              technical_indicators: Dict, market_structure: Dict, positions: List) -> Dict:
        """Prepare comprehensive market context"""
        try:
            primary_tf = self.config['trading']['modes']['scalping']['smc_analysis']['primary_timeframe']
            
            return {
                'symbol': symbol,
                'timeframe': primary_tf,
                'current_price': current_price,
                'technical_indicators': technical_indicators[primary_tf],
                'market_structure': market_structure[primary_tf]['market_structure'],
                'smc_data': market_structure[primary_tf]['smc_data'],
                'active_positions': positions,
                'position_manager': self.position_manager,
                'config': self.config
            }
        except Exception as e:
            logger.error(f"Error preparing market context: {str(e)}")
            return {}

    async def _generate_signal(self, market_context: Dict) -> Dict:
        """Generate trading signal using LLM analyzer"""
        try:
            if not self.hummingbird.llm_analyzer:
                logger.warning("LLM analyzer not available")
                return None
            
            # Ensure LLM analyzer has access to required components
            self.hummingbird.llm_analyzer.position_manager = self.position_manager
            self.hummingbird.llm_analyzer.db = self.db
            
            # Generate signal - ensure we're not awaiting a dictionary
            signal = self.hummingbird.llm_analyzer.generate_signal(market_context)
            if not signal:
                logger.warning("No signal generated")
                return None
                
            # Ensure all datetime objects are serialized
            if isinstance(signal, dict):
                signal = {
                    k: v.isoformat() if hasattr(v, 'isoformat') else v
                    for k, v in signal.items()
                }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    # SMC Pattern Detection Methods
    def _detect_order_blocks(self, df: pd.DataFrame) -> List:
        """Detect order blocks in price action"""
        # Implementation of order block detection
        pass

    def _detect_supply_zones(self, df: pd.DataFrame) -> List:
        """Detect supply zones in price action"""
        # Implementation of supply zone detection
        pass

    def _detect_demand_zones(self, df: pd.DataFrame) -> List:
        """Detect demand zones in price action"""
        # Implementation of demand zone detection
        pass

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List:
        """Detect fair value gaps in price action"""
        # Implementation of fair value gap detection
        pass

    def _detect_liquidity_levels(self, df: pd.DataFrame) -> List:
        """Detect liquidity levels in price action"""
        # Implementation of liquidity level detection
        pass

    def _detect_smart_money_traps(self, df: pd.DataFrame) -> List:
        """Detect smart money traps in price action"""
        # Implementation of smart money trap detection
        pass

    def _determine_market_structure(self, smc_data: Dict) -> str:
        """Determine overall market structure based on SMC patterns"""
        # Implementation of market structure determination
        return 'NEUTRAL'

    async def _get_active_positions(self, symbol: str) -> List:
        """Get active positions for a symbol"""
        try:
            if not self.position_manager:
                logger.error("No position manager available - reinitializing...")
                if self._init_position_manager():
                    logger.info("Successfully reinitialized position manager")
                else:
                    logger.error("Failed to reinitialize position manager")
            
            if self.position_manager:
                logger.info(f"Fetching active positions for {symbol}...")
                # Get active positions directly from database
                active_positions = self.position_manager.get_active_positions()
                if active_positions is None:
                    logger.error("get_active_positions() returned None")
                else:
                    logger.info(f"Found {len(active_positions)} total active positions")
                    
                    # Filter positions for this specific symbol
                    symbol_positions = [
                        format_position(p) for p in active_positions 
                        if p is not None and p.symbol == symbol and p.status == 'OPEN'
                    ]
                    logger.info(f"Found {len(symbol_positions)} active positions for {symbol}")
            return symbol_positions
        except Exception as e:
            logger.error(f"Error fetching positions for {symbol}: {str(e)}")
            return [] 