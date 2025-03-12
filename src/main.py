import argparse
import logging
import yaml
import time
from typing import Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.market_data import MarketData
from src.data.models import Position, PositionStatus, PositionType, MarketStructureData
from src.technical.analysis import TechnicalAnalysis
from src.technical.position_manager import PositionManager
from src.technical.market_structure import MarketStructureAnalyzer
from src.llm.analyzer import LLMAnalyzer
from src.technical.smc_patterns import SMCPatternDetector
import pandas as pd

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading config: {str(e)}")
        raise

class Hummingbird:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Setup database connection
        engine = create_engine(self.config['database']['url'])
        Session = sessionmaker(bind=engine)
        self.db = Session()
        
        # Initialize components
        self.position_manager = PositionManager(self.db)
        self.market_structure = MarketStructureAnalyzer(self.db, self.config)
        self.market_data = MarketData(self.config)
        self.technical_analyzer = TechnicalAnalysis()
        
        # Initialize LLM analyzer with position management
        self.llm_analyzer = LLMAnalyzer(
            model_config=self.config['llm'],
            model_name=self.config['llm']['default_model'],
            technical_analyzer=self.technical_analyzer
        )
        self.llm_analyzer.set_position_manager(self.position_manager, self.db)
        
        # Initialize trading mode and symbol
        self.trading_mode = "swing"
        self.symbol = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Hummingbird')
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol for Binance API
        """
        if '/' not in symbol:
            return f"{symbol}/USDT"
        return symbol
    
    def _display_position_status(self, position):
        """Display current position status"""
        table = Table(title=f"Position Status - {position.symbol}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Type", position.position_type.value)
        table.add_row("Status", position.status.value)
        table.add_row("Entry Price", f"${position.entry_price:.2f}")
        table.add_row("Current Price", f"${position.current_price:.2f}")
        table.add_row("Stop Loss", f"${position.stop_loss:.2f}")
        table.add_row("Take Profit", f"${position.take_profit:.2f}")
        table.add_row("PnL", f"${position.pnl:.2f}")
        
        console.print(table)
    
    def _display_market_structure(self, market_structure):
        """Display current market structure"""
        table = Table(title="Market Structure Analysis")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")
        
        table.add_row("Structure Type", market_structure.structure_type.value)
        table.add_row("Order Blocks", str(len(market_structure.order_blocks)))
        table.add_row("Fair Value Gaps", str(len(market_structure.fair_value_gaps)))
        table.add_row("Liquidity Levels", str(len(market_structure.liquidity_levels)))
        
        console.print(table)
    
    def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get latest market data for a symbol"""
        try:
            self.logger.info(f"Fetching market data for {symbol}")
            # Get data for the current trading mode's default timeframe
            timeframe = self.config['trading']['modes'][self.trading_mode]['default_timeframe']
            days = self.config['data']['historical_data_days']
            
            df = self.market_data.fetch_historical_data(
                symbol,
                timeframe,
                days
            )
            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def monitor_positions(self):
        """Monitor active positions and market structure"""
        try:
            self.logger.info("Starting position monitoring")
            # Get active positions
            active_positions = self.position_manager.get_active_positions()
            self.logger.info(f"Found {len(active_positions)} active positions")
            
            for position in active_positions:
                self.logger.info(f"Processing position {position.id} for {position.symbol}")
                # Get latest market data
                market_data = self._get_market_data(position.symbol)
                
                # Analyze market structure
                self.logger.info("Analyzing market structure")
                structure = self.market_structure.analyze_market_structure(
                    market_data,
                    position.timeframe,
                    position.id
                )
                
                # Check for exit signals
                self.logger.info("Checking for exit signals")
                if self._check_exit_signals(position, structure):
                    self.position_manager.close_position(position.id)
                    self.logger.info(f"Position {position.id} closed due to exit signal")
                
                # Update position status
                self.logger.info("Updating position status")
                self.position_manager.update_position_status(position.id, structure)
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")
            self.db.rollback()
        finally:
            self.db.commit()
    
    def _check_exit_signals(self, position, structure) -> bool:
        """Check for exit signals based on market structure"""
        # Check structure shift
        if self.market_structure.detect_structure_shift(structure, position.last_structure):
            return True
        
        # Check position strength
        strength = self.market_structure.validate_position_strength(
            position.type,
            structure
        )
        
        # Exit if position strength is too low
        if strength < self.config['position']['min_strength']:
            return True
        
        return False
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Analyze market data and generate trading signals
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                # Get timeframes for current trading mode
                mode_config = self.config['trading']['modes'][self.trading_mode]
                primary_timeframe = mode_config['smc_analysis']['primary_timeframe']
                secondary_timeframes = mode_config['smc_analysis']['secondary_timeframes']
                lookback_periods = mode_config['smc_analysis']['lookback_periods']
                
                # Fetch data for all timeframes
                data_task = progress.add_task("[cyan]Fetching market data...", total=1 + len(secondary_timeframes))
                market_data = {}
                
                # Fetch primary timeframe data
                df = self.market_data.fetch_historical_data(
                    symbol,
                    primary_timeframe,
                    self.config['data']['historical_data_days']
                )
                if df is not None and not df.empty:
                    market_data[primary_timeframe] = df.tail(lookback_periods[primary_timeframe])
                else:
                    self.logger.warning(f"Failed to fetch data for primary timeframe {primary_timeframe}")
                progress.advance(data_task)
                
                # Fetch secondary timeframe data
                for tf in secondary_timeframes:
                    df = self.market_data.fetch_historical_data(
                        symbol,
                        tf,
                        self.config['data']['historical_data_days']
                    )
                    if df is not None and not df.empty:
                        market_data[tf] = df.tail(lookback_periods[tf])
                    else:
                        self.logger.warning(f"Failed to fetch data for secondary timeframe {tf}")
                    progress.advance(data_task)
                
                # Check if we have data for the primary timeframe
                if primary_timeframe not in market_data:
                    raise ValueError(f"No data available for primary timeframe {primary_timeframe}")
                
                # Calculate technical indicators for all timeframes
                indicators_task = progress.add_task("[yellow]Calculating technical indicators...", total=len(market_data))
                technical_indicators = {}
                processed_data = {}  # Store processed dataframes with indicators
                
                for tf, df in market_data.items():
                    try:
                        # Ensure we have enough data points
                        if len(df) < 26:  # Minimum required for MACD
                            self.logger.warning(f"Insufficient data points for {tf}: {len(df)}")
                            continue
                            
                        df_with_indicators = self.technical_analyzer.calculate_indicators(df)
                        if df_with_indicators is not None and not df_with_indicators.empty:
                            processed_data[tf] = df_with_indicators  # Store processed dataframe
                            technical_indicators[tf] = {
                                'RSI': df_with_indicators['RSI'].iloc[-1],
                                'MACD': df_with_indicators['MACD'].iloc[-1],
                                'MACD_Signal': df_with_indicators['MACD_Signal'].iloc[-1],
                                'MACD_Hist': df_with_indicators['MACD_Hist'].iloc[-1],
                                'EMA_8': df_with_indicators['EMA_8'].iloc[-1],
                                'EMA_21': df_with_indicators['EMA_21'].iloc[-1]
                            }
                        else:
                            self.logger.warning(f"Failed to calculate indicators for timeframe {tf}")
                    except Exception as e:
                        self.logger.error(f"Error calculating indicators for {tf}: {str(e)}")
                    progress.advance(indicators_task)
                
                # Analyze market structure for all timeframes
                structure_task = progress.add_task("[green]Analyzing market structure...", total=len(processed_data))
                market_structure = {}
                
                for tf, df in processed_data.items():  # Use processed data with indicators
                    try:
                        # Ensure we have all required columns
                        required_columns = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'EMA_21']
                        if not all(col in df.columns for col in required_columns):
                            self.logger.warning(f"Missing required columns for {tf}")
                            continue
                            
                        market_structure[tf] = self.market_structure.analyze_market_structure(
                            df,
                            tf,
                            symbol
                        )
                    except Exception as e:
                        self.logger.error(f"Error analyzing market structure for {tf}: {str(e)}")
                    progress.advance(structure_task)
                
                # Get current price directly from Binance
                current_price = self.market_data.get_current_price(symbol)
                if current_price is None:
                    raise ValueError("Failed to get current price from Binance")
                
                # Display current market status
                if primary_timeframe in technical_indicators:
                    self._display_market_status(symbol, current_price, technical_indicators[primary_timeframe])
                
                # Display SMC patterns and market structure for all timeframes
                for tf, df in processed_data.items():
                    if df is not None and not df.empty:
                        console.print(f"\n[bold yellow]SMC Analysis for {tf} Timeframe:")
                        # Get the technical indicators for this timeframe
                        indicators = technical_indicators.get(tf, {})
                        
                        # Create market structure data dictionary
                        smc_data = {
                            'Order_Blocks': self.technical_analyzer.indicators.get('Order_Blocks', []),
                            'Supply_Zones': self.technical_analyzer.indicators.get('Supply_Zones', []),
                            'Demand_Zones': self.technical_analyzer.indicators.get('Demand_Zones', []),
                            'Fair_Value_Gaps': self.technical_analyzer.indicators.get('Fair_Value_Gaps', []),
                            'Liquidity_Levels': self.technical_analyzer.indicators.get('Liquidity_Levels', [])
                        }
                        
                        # Display patterns
                        self._display_patterns(smc_data)
                
                # Generate trading signal using all timeframes
                if primary_timeframe in market_structure and market_structure[primary_timeframe] is not None:
                    # Prepare market context
                    market_context = self.llm_analyzer.prepare_market_context(
                        market_data,
                        technical_indicators.get(primary_timeframe, {}),
                        market_structure[primary_timeframe].order_blocks,
                        market_structure[primary_timeframe].fair_value_gaps,
                        market_structure[primary_timeframe].liquidity_levels
                    )
                    
                    # Debug: Print market context
                    console.print("\n[bold cyan]Debug: Market Context being sent to LLM:[/bold cyan]")
                    console.print(market_context)
                    
                    # Get active positions for context
                    active_positions = self.position_manager.get_active_positions()
                    console.print("\n[bold cyan]Debug: Current Active Positions:[/bold cyan]")
                    for pos in active_positions:
                        console.print(f"Position ID: {pos.id}, Type: {pos.position_type}, Status: {pos.status}")
                    
                    # Generate signal
                    signal = self.llm_analyzer.generate_signal(market_context)
                    
                    # Debug: Print LLM response
                    console.print("\n[bold cyan]Debug: LLM Response:[/bold cyan]")
                    console.print(signal)
                    
                else:
                    raise ValueError("Failed to analyze market structure for primary timeframe")
                
                return {
                    'signal': signal,
                    'market_structure': market_structure,
                    'technical_indicators': technical_indicators,
                    'current_price': current_price
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            raise
    
    def _display_market_status(self, symbol: str, current_price: float, indicators: Dict):
        """
        Display current market status in a beautiful format
        """
        table = Table(title=f"Market Status - {symbol}")
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", style="green")
        
        # Add each indicator to the table
        for name, value in indicators.items():
            if isinstance(value, (int, float)):
                table.add_row(
                    name,
                    f"{float(value):.2f}"
                )
            else:
                table.add_row(
                    name,
                    str(value)
                )
        
        console.print(table)
        console.print(Panel(
            f"Current Price: ${current_price:.2f}",
            title="Price Information",
            border_style="blue"
        ))
    
    def _display_patterns(self, market_structure: Dict):
        """
        Display recent market patterns with SMC details
        """
        if not market_structure:
            console.print("\n[bold red]No market structure data available")
            return

        # Display Order Blocks
        order_blocks = market_structure.get('Order_Blocks', [])
        console.print("\n[bold cyan]Order Blocks:")
        if order_blocks:
            for block in sorted(order_blocks, key=lambda x: x['strength'], reverse=True)[:3]:
                console.print(Panel(
                    f"Type: {block['type']}\n"
                    f"Price: {block['price']:.2f}\n"
                    f"Volume: {block['volume']:.2f}\n"
                    f"Strength: {block['strength']:.2f}\n"
                    f"Timeframe: {block['timeframe']}",
                    title="Order Block",
                    border_style="cyan"
                ))
        else:
            console.print("[yellow]No Order Blocks identified")

        # Display Supply and Demand Zones
        supply_zones = market_structure.get('Supply_Zones', [])
        demand_zones = market_structure.get('Demand_Zones', [])
        
        console.print("\n[bold green]Supply and Demand Zones:")
        if supply_zones:
            console.print("\nSupply Zones:")
            for zone in sorted(supply_zones, key=lambda x: x.get('strength', 0), reverse=True)[:3]:
                console.print(Panel(
                    f"Type: {zone['type']}\n"
                    f"Price: {zone['price']:.2f}\n"
                    f"Strength: {zone.get('strength', 0):.2f}\n"
                    f"Timeframe: {zone['timeframe']}",
                    title="Supply Zone",
                    border_style="red"
                ))
        
        if demand_zones:
            console.print("\nDemand Zones:")
            for zone in sorted(demand_zones, key=lambda x: x.get('strength', 0), reverse=True)[:3]:
                console.print(Panel(
                    f"Type: {zone['type']}\n"
                    f"Price: {zone['price']:.2f}\n"
                    f"Strength: {zone.get('strength', 0):.2f}\n"
                    f"Timeframe: {zone['timeframe']}",
                    title="Demand Zone",
                    border_style="green"
                ))

        # Display Fair Value Gaps
        fair_value_gaps = market_structure.get('Fair_Value_Gaps', [])
        console.print("\n[bold yellow]Fair Value Gaps:")
        if fair_value_gaps:
            for gap in fair_value_gaps[:3]:
                console.print(Panel(
                    f"Type: {gap['type']}\n"
                    f"Upper Price: {gap['upper_price']:.2f}\n"
                    f"Lower Price: {gap['lower_price']:.2f}\n"
                    f"Gap Size: {gap['gap_size']:.2f}\n"
                    f"Timeframe: {gap['timeframe']}",
                    title="Fair Value Gap",
                    border_style="yellow"
                ))
        else:
            console.print("[yellow]No Fair Value Gaps identified")

        # Display Liquidity Levels
        liquidity_levels = market_structure.get('Liquidity_Levels', [])
        console.print("\n[bold magenta]Liquidity Levels:")
        if liquidity_levels:
            for level in sorted(liquidity_levels, key=lambda x: x['strength'], reverse=True)[:3]:
                console.print(Panel(
                    f"Type: {level['type']}\n"
                    f"Price: {level['price']:.2f}\n"
                    f"Volume: {level['volume']:.2f}\n"
                    f"Strength: {level['strength']:.2f}\n"
                    f"Timeframe: {level['timeframe']}",
                    title="Liquidity Level",
                    border_style="magenta"
                ))
        else:
            console.print("[yellow]No Liquidity Levels identified")

        console.print()  # Add a blank line for better readability
    
    def run(self):
        """Main loop for the trading system"""
        try:
            self.logger.info(f"Starting Hummingbird trading system in {self.trading_mode} mode for {self.symbol}")
            
            # Get the monitoring interval from config
            interval = self.config['monitoring']['interval']
            self.logger.info(f"Monitoring interval: {interval} seconds")
            
            while True:
                try:
                    # Monitor active positions
                    self.monitor_positions()
                    
                    # Analyze market for new opportunities
                    analysis_result = self.analyze_market(self.symbol)
                    
                    # Log the analysis result
                    if analysis_result and 'signal' in analysis_result:
                        signal_data = analysis_result['signal']
                        if isinstance(signal_data, dict) and 'confidence' in signal_data:
                            self.logger.info(f"Trading signal: {signal_data['signal']} with confidence {signal_data['confidence']:.2%}")
                    
                    # Sleep for the configured interval
                    self.logger.info(f"Waiting {interval} seconds before next analysis...")
                    time.sleep(interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    # Use a shorter delay on error
                    error_delay = self.config['monitoring'].get('error_delay', 5)
                    self.logger.info(f"Waiting {error_delay} seconds before retrying...")
                    time.sleep(error_delay)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down Hummingbird trading system")
        finally:
            self.db.close()

    def run_analysis(self):
        """Run market analysis and position management"""
        try:
            while True:
                # Get market data
                market_data = self._get_market_data(self.symbol)
                
                # Prepare market context
                market_context = {
                    'symbol': self.symbol,
                    'timeframe': self.config['trading']['modes'][self.trading_mode]['default_timeframe'],
                    'current_price': market_data['close'].iloc[-1],
                    'config': self.config,
                    'market_data': market_data.to_dict('records')
                }
                
                # Generate signal and manage positions
                signal = self.llm_analyzer.generate_signal(market_context)
                
                if signal:
                    self.logger.info(f"Trading signal: {signal['signal']} with confidence {signal['confidence']:.2%}")
                    
                    # Display active positions
                    active_positions = self.position_manager.get_active_positions()
                    if active_positions:
                        for position in active_positions:
                            self._display_position_status(position)
                
                # Wait for next analysis
                time.sleep(self.config['monitoring']['interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        except Exception as e:
            self.logger.error(f"Error in analysis loop: {str(e)}")
        finally:
            self.db.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hummingbird Trading System")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--model", type=str, default="local", help="LLM model to use")
    parser.add_argument("--mode", type=str, default="swing", choices=["swing", "scalping"], help="Trading mode")
    args = parser.parse_args()
    
    try:
        hummingbird = Hummingbird("config/config.yaml")
        hummingbird.trading_mode = args.mode
        hummingbird.symbol = args.symbol
        
        # Update the LLM analyzer with the specified model
        console.print(f"[bold cyan]Debug: Setting up {args.model} model[/bold cyan]")
        hummingbird.llm_analyzer = LLMAnalyzer(
            model_config=hummingbird.config['llm'],
            model_name=args.model,  # Use the model specified in command line args
            technical_analyzer=hummingbird.technical_analyzer
        )
        
        hummingbird.run()
    except Exception as e:
        console.print(f"[bold red]Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()