import ccxt
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from rich.console import Console
import time

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
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
        
        # Initialize exchange
        self.exchange = self._initialize_exchange(api_key, api_secret)
        
        self.timeframes = {
            "1d": "1d",
            "4h": "4h",
            "1h": "1h",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m"
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
    
    def fetch_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch historical market data"""
        try:
            # Format symbol if needed
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
                
            self.logger.info(f"Fetching historical data for {symbol} {timeframe}")
            
            # Calculate the number of candles needed based on timeframe
            if timeframe == '1m':
                limit = 1500  # 1500 minutes = 25 hours (increased from 1000)
            elif timeframe == '5m':
                limit = 1000  # 1000 5-minute candles = ~83.3 hours (increased from 500)
            elif timeframe == '15m':
                limit = 800   # 800 15-minute candles = ~200 hours (increased from 400)
            elif timeframe == '1h':
                limit = 500   # 500 hours = ~20.8 days (increased from 300)
            elif timeframe == '4h':
                limit = 300   # 300 4-hour candles = 50 days (increased from 200)
            elif timeframe == '1d':
                limit = 200   # 200 days (increased from 100)
            
            # Handle 1m timeframe with rate limiting
            if timeframe == '1m':
                all_candles = []
                current_limit = min(limit, 1000)  # Binance limit per request
                remaining_limit = limit
                
                while remaining_limit > 0:
                    try:
                        # If this is not the first request, use the timestamp of the last candle as until
                        until = None
                        if all_candles:
                            until = all_candles[0][0] - 1  # Subtract 1ms to avoid duplicate candle
                        
                        candles = self.exchange.fetch_ohlcv(
                            symbol,
                            timeframe,
                            limit=current_limit,
                            params={'until': until} if until else {}
                        )
                        
                        if not candles:
                            break
                            
                        all_candles = candles + all_candles  # Prepend new candles
                        remaining_limit -= len(candles)
                        
                        # Add small delay to respect rate limits
                        time.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.warning(f"Error fetching 1m candles: {str(e)}")
                        break
                        
                if not all_candles:
                    self.logger.warning(f"No data returned for {symbol} {timeframe}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(
                    all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
            else:
                # For other timeframes, fetch directly
                candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not candles:
                    self.logger.warning(f"No data returned for {symbol} {timeframe}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(
                    candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.index.name = timeframe  # Set the timeframe as the index name
            
            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
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

    def _initialize_exchange(self, api_key: str, api_secret: str) -> ccxt.Exchange:
        """
        Initialize the cryptocurrency exchange connection
        """
        try:
            # Initialize Binance exchange
            exchange = ccxt.binance({
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
            exchange.load_markets()
            self.logger.info("Successfully connected to Binance exchange")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {str(e)}")
            raise 