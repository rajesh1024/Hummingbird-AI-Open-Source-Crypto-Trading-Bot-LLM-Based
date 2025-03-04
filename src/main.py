import argparse
import logging
import yaml
import time
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from datetime import datetime

from data.market_data import MarketData
from technical.analysis import TechnicalAnalysis
from llm.analyzer import LLMAnalyzer

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
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.market_data = MarketData()
        self.technical_analyzer = TechnicalAnalysis()
        self.llm_analyzer = LLMAnalyzer(
            model_config=self.config['llm'],
            model_name=self.config['llm']['default_model'],
            technical_analyzer=self.technical_analyzer
        )
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol for Binance API
        """
        if '/' not in symbol:
            return f"{symbol}/USDT"
        return symbol
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Perform comprehensive market analysis
        """
        try:
            # Format symbol for Binance
            symbol = self._format_symbol(symbol)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                # Fetch historical data with adjusted days based on timeframe
                data_task = progress.add_task("[cyan]Fetching market data...", total=len(self.config['trading']['timeframes']))
                historical_data = {}
                for timeframe in self.config['trading']['timeframes']:
                    # Adjust historical data days based on timeframe
                    if timeframe == '15m':
                        days = 1  # Last 24 hours for 15m
                    elif timeframe == '5m':
                        days = 0.5  # Last 12 hours for 5m
                    elif timeframe == '3m':
                        days = 0.25  # Last 6 hours for 3m
                    else:
                        days = self.config['data']['historical_data_days']
                    
                    df = self.market_data.fetch_historical_data(
                        symbol,
                        timeframe,
                        days
                    )
                    # Set the index name to the timeframe
                    df.index.name = timeframe
                    historical_data[timeframe] = df
                    progress.advance(data_task)
                
                # Calculate technical indicators
                indicators_task = progress.add_task("[yellow]Calculating technical indicators...", total=len(historical_data))
                technical_indicators = {}
                for timeframe, df in historical_data.items():
                    # Ensure we're using recent data
                    df = df.tail(100)  # Use last 100 candles for more recent analysis
                    df_with_indicators = self.technical_analyzer.calculate_indicators(df)
                    technical_indicators[timeframe] = {
                        'RSI': df_with_indicators['RSI'].iloc[-1],
                        'MACD': df_with_indicators['MACD'].iloc[-1],
                        'MACD_Signal': df_with_indicators['MACD_Signal'].iloc[-1],
                        'MACD_Hist': df_with_indicators['MACD_Hist'].iloc[-1],
                        'EMA_8': df_with_indicators['EMA_8'].iloc[-1],
                        'EMA_21': df_with_indicators['EMA_21'].iloc[-1]
                    }
                    progress.advance(indicators_task)
                
                # Identify patterns for each timeframe
                patterns_task = progress.add_task("[green]Identifying market patterns...", total=len(historical_data) * 2)
                all_order_blocks = []
                all_supply_zones = []
                all_demand_zones = []
                
                for timeframe, df in historical_data.items():
                    # Use recent data for pattern identification
                    df = df.tail(100)
                    df_with_indicators = self.technical_analyzer.calculate_indicators(df)
                    
                    # Identify order blocks
                    order_blocks = self.technical_analyzer.identify_order_blocks(
                        df_with_indicators,
                        timeframe=timeframe
                    )
                    all_order_blocks.extend(order_blocks)
                    progress.advance(patterns_task)
                    
                    # Identify supply/demand zones
                    supply_zones, demand_zones = self.technical_analyzer.identify_supply_demand_zones(
                        df_with_indicators,
                        timeframe=timeframe
                    )
                    all_supply_zones.extend(supply_zones)
                    all_demand_zones.extend(demand_zones)
                    progress.advance(patterns_task)
                
                # Sort and limit the number of patterns
                all_order_blocks.sort(key=lambda x: x['strength'], reverse=True)
                all_supply_zones.sort(key=lambda x: x['strength'], reverse=True)
                all_demand_zones.sort(key=lambda x: x['strength'], reverse=True)
                
                order_blocks = all_order_blocks[:3]
                supply_zones = all_supply_zones[:2]
                demand_zones = all_demand_zones[:2]
            
            # Get current price directly from Binance
            current_price = self.market_data.get_current_price(symbol)
            
            # Display current market status
            self._display_market_status(symbol, current_price, technical_indicators)
            
            # Display recent patterns
            self._display_patterns(order_blocks, supply_zones, demand_zones)
            
            # Generate trading signal
            signal = self.llm_analyzer.generate_trading_signal(
                self.llm_analyzer.prepare_market_context(
                    historical_data,
                    technical_indicators,
                    order_blocks,
                    supply_zones,
                    demand_zones
                ),
                current_price,
                self.config['llm']['confidence_threshold']
            )
            
            return {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now(),
                'current_price': current_price,
                'signal': signal,
                'technical_indicators': technical_indicators,
                'order_blocks': order_blocks,
                'supply_zones': supply_zones,
                'demand_zones': demand_zones
            }
            
        except Exception as e:
            console.print(f"[bold red]Error analyzing market: {str(e)}")
            raise
    
    def _display_market_status(self, symbol: str, current_price: float, indicators: Dict):
        """
        Display current market status in a beautiful format
        """
        table = Table(title=f"Market Status - {symbol}")
        table.add_column("Timeframe", style="cyan")
        table.add_column("RSI", style="green")
        table.add_column("MACD", style="magenta")
        table.add_column("EMA 8", style="yellow")
        table.add_column("EMA 21", style="yellow")
        
        for timeframe, values in indicators.items():
            table.add_row(
                timeframe,
                f"{values['RSI']:.2f}",
                f"{values['MACD']:.2f}",
                f"{values['EMA_8']:.2f}",
                f"{values['EMA_21']:.2f}"
            )
        
        console.print(table)
        console.print(Panel(
            f"Current Price: ${current_price:.2f}",
            title="Price Information",
            border_style="blue"
        ))
    
    def _display_patterns(self, order_blocks: List[Dict], supply_zones: List[Dict], demand_zones: List[Dict]):
        """
        Display recent market patterns with SMC details
        """
        # Debug print raw pattern data
        console.print("\n[bold yellow]Raw Pattern Data:")
        console.print(f"Order Blocks Count: {len(self.technical_analyzer.order_blocks)}")
        console.print(f"Supply Zones Count: {len(self.technical_analyzer.supply_zones)}")
        console.print(f"Demand Zones Count: {len(self.technical_analyzer.demand_zones)}")
        console.print(f"Fair Value Gaps Count: {len(self.technical_analyzer.fair_value_gaps)}")
        console.print(f"Liquidity Sweeps Count: {len(self.technical_analyzer.liquidity_sweeps)}")
        
        # Display Order Blocks with SMC details
        if self.technical_analyzer.order_blocks:
            console.print("\n[bold cyan]Recent Order Blocks (SMC):")
            for block in sorted(self.technical_analyzer.order_blocks, key=lambda x: x['strength'], reverse=True)[:3]:
                console.print(
                    f"Type: {block['type']}, "
                    f"Price: {block['price']:.2f}, "
                    f"Volume: {block['volume']:.2f}, "
                    f"Strength: {block['strength']:.2f}, "
                    f"Timeframe: {block['timeframe']}"
                )
        else:
            console.print("\n[bold yellow]No Order Blocks identified")
        
        # Display Supply Zones with SMC details
        if self.technical_analyzer.supply_zones:
            console.print("\n[bold red]Supply Zones (SMC):")
            for zone in sorted(self.technical_analyzer.supply_zones, key=lambda x: x['strength'], reverse=True)[:2]:
                console.print(
                    f"Price: {zone['price']:.2f}, "
                    f"Strength: {zone['strength']:.2f}, "
                    f"Timeframe: {zone['timeframe']}"
                )
        else:
            console.print("\n[bold yellow]No Supply Zones identified")
        
        # Display Demand Zones with SMC details
        if self.technical_analyzer.demand_zones:
            console.print("\n[bold green]Demand Zones (SMC):")
            for zone in sorted(self.technical_analyzer.demand_zones, key=lambda x: x['strength'], reverse=True)[:2]:
                console.print(
                    f"Price: {zone['price']:.2f}, "
                    f"Strength: {zone['strength']:.2f}, "
                    f"Timeframe: {zone['timeframe']}"
                )
        else:
            console.print("\n[bold yellow]No Demand Zones identified")
        
        # Display Fair Value Gaps if available
        if self.technical_analyzer.fair_value_gaps:
            console.print("\n[bold yellow]Fair Value Gaps (SMC):")
            for gap in self.technical_analyzer.fair_value_gaps:
                console.print(
                    f"Type: {gap['type']}, "
                    f"Price: {gap['price']:.2f}, "
                    f"Timeframe: {gap['timeframe']}"
                )
        else:
            console.print("\n[bold yellow]No Fair Value Gaps identified")
        
        # Display Liquidity Sweeps if available
        if self.technical_analyzer.liquidity_sweeps:
            console.print("\n[bold magenta]Liquidity Sweeps (SMC):")
            for sweep in self.technical_analyzer.liquidity_sweeps:
                console.print(
                    f"Type: {sweep['type']}, "
                    f"Price: {sweep['price']:.2f}, "
                    f"Timeframe: {sweep['timeframe']}"
                )
        else:
            console.print("\n[bold yellow]No Liquidity Sweeps identified")
        
        console.print()  # Add a blank line for better readability
    
    def run(self, symbol: str):
        """
        Run the trading system continuously
        """
        # Format symbol for display
        display_symbol = self._format_symbol(symbol)
        
        console.print(Panel(
            f"Starting Hummingbird Trading System\nSymbol: {display_symbol}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            title="System Status",
            border_style="green"
        ))
        
        while True:
            try:
                analysis = self.analyze_market(symbol)
                
                # Wait for next update interval
                with console.status(f"[bold blue]Waiting {self.config['data']['update_interval']} seconds until next analysis..."):
                    time.sleep(self.config['data']['update_interval'])
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Stopping Hummingbird trading system")
                break
            except Exception as e:
                console.print(f"[bold red]Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Cryptocurrency Trading System')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTC or BTC/USDT)')
    parser.add_argument('--model', type=str, default='mistral', help='LLM model to use (mistral or gemini)')
    parser.add_argument('--strategy', type=str, default='swing', choices=['swing', 'scalping'], 
                      help='Trading strategy: swing (long-term) or scalping (short-term)')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()
        
        # Override default model if specified
        if args.model:
            if args.model not in config['llm']['models']:
                raise ValueError(f"Invalid model: {args.model}. Available models: {', '.join(config['llm']['models'].keys())}")
            config['llm']['default_model'] = args.model
        
        # Apply strategy-specific settings
        strategy = args.strategy
        config['trading']['timeframes'] = config['trading']['timeframes'][strategy]
        config['trading']['technical_indicators'] = config['trading']['technical_indicators'][strategy]
        config['risk_management'] = config['risk_management'][strategy]
        
        # Adjust update interval based on strategy
        config['data']['update_interval'] = 10 if strategy == 'scalping' else 20
        
        # Initialize components
        market_data = MarketData()
        technical_analyzer = TechnicalAnalysis()
        llm_analyzer = LLMAnalyzer(
            config['llm'],
            config['llm']['default_model'],
            technical_analyzer=technical_analyzer
        )
        
        # Create and run trading system
        hummingbird = Hummingbird()
        hummingbird.config = config  # Update config with strategy-specific settings
        hummingbird.technical_analyzer = technical_analyzer  # Ensure we use the same instance
        hummingbird.llm_analyzer = llm_analyzer  # Ensure we use the same instance
        
        # Format symbol if needed
        symbol = args.symbol.upper()
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"
            
        # Run the system
        hummingbird.run(symbol)
        
    except Exception as e:
        console.print(f"[bold red]Error in main loop: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Hummingbird Trading System")
    main()