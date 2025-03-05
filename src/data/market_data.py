import ccxt
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from rich.console import Console
import time
import redis
import json

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange = self._initialize_exchange()
        self.redis_client = self._initialize_redis()
        
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
            "5m": "5m",
            "1m": "1m"
        }
    
    def _initialize_redis(self) -> redis.Redis:
        """Initialize Redis connection"""
        try:
            redis_host = self.config.get('redis', {}).get('host', 'localhost')
            redis_port = self.config.get('redis', {}).get('port', 6379)
            redis_db = self.config.get('redis', {}).get('db', 0)
            return redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {str(e)}")
            return None
            
    def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get market data from Redis cache"""
        if not self.redis_client:
            return None
            
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data_dict = json.loads(cached_data)
                df = pd.DataFrame.from_dict(data_dict)
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            self.logger.error(f"Error getting cached data: {str(e)}")
        return None
        
    def _cache_data(self, symbol: str, timeframe: str, df: pd.DataFrame, ttl: int = 300):
        """Cache market data in Redis"""
        if not self.redis_client:
            return
            
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            data_dict = df.to_dict()
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data_dict)
            )
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")
            
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
        """Fetch historical market data with caching"""
        try:
            # Check cache first
            cached_data = self._get_cached_data(symbol, timeframe)
            if cached_data is not None:
                self.logger.info(f"Using cached data for {symbol} {timeframe}")
                return cached_data
                
            # Format symbol if needed
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
                
            self.logger.info(f"Fetching historical data for {symbol} {timeframe}")
            
            # Handle 1m timeframe with rate limiting
            if timeframe == '1m':
                all_candles = []
                current_limit = min(limit, 1000)  # Binance limit per request
                remaining_limit = limit
                
                while remaining_limit > 0:
                    try:
                        candles = self.exchange.fetch_ohlcv(
                            symbol,
                            timeframe,
                            limit=current_limit
                        )
                        if not candles:
                            break
                            
                        all_candles.extend(candles)
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
            
            # Cache the data
            self._cache_data(symbol, timeframe, df)
            
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