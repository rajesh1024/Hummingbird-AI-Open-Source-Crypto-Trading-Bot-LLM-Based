import ccxt
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        load_dotenv()  # Load environment variables
        
        # Get API credentials
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Validate API credentials
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in .env file")
        
        # Clean up API credentials (remove any whitespace or newlines)
        api_key = api_key.strip()
        api_secret = api_secret.strip()
        
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # Enable built-in rate limiter
            'options': {
                'defaultType': 'spot',  # Use spot market
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # Increase receive window
            }
        })
        
        # Test API connection
        try:
            self.exchange.load_markets()
            console.print("[bold green]✓ Successfully connected to Binance API")
        except Exception as e:
            console.print(f"[bold red]Error connecting to Binance API: {str(e)}")
            console.print("[yellow]Please check your API key and secret in the .env file")
            raise
        
        self.timeframes = {
            "1d": "1d",
            "4h": "4h",
            "1h": "1h",
            "15m": "15m",
            "5m": "5m"
        }
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price directly from Binance's ticker endpoint
        """
        try:
            # Ensure symbol is in correct format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']  # Get the last traded price
            
        except Exception as e:
            console.print(f"[bold red]Error fetching current price: {str(e)}")
            raise
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data for a given symbol and timeframe
        """
        try:
            # Ensure symbol is in correct format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes[timeframe],
                since=since
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            console.print(f"[bold red]Error fetching historical data: {str(e)}")
            raise
    
    def fetch_all_timeframes(
        self,
        symbol: str,
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all timeframes
        """
        data = {}
        for timeframe in self.timeframes:
            data[timeframe] = self.fetch_historical_data(
                symbol,
                timeframe,
                days
            )
        return data
    
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get the latest market data for real-time analysis
        """
        try:
            # Ensure symbol is in correct format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
                
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.timeframes[timeframe],
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            console.print(f"[bold red]Error fetching latest data: {str(e)}")
            raise
    
    def get_orderbook(self, symbol: str) -> Dict:
        """
        Get current orderbook data
        """
        try:
            # Ensure symbol is in correct format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            return self.exchange.fetch_order_book(symbol)
        except Exception as e:
            console.print(f"[bold red]Error fetching orderbook: {str(e)}")
            raise 