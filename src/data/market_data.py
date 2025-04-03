import ccxt
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from rich.console import Console
import time
import numpy as np

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, config: dict):
        """Initialize market data fetcher with configuration"""
        self.config = config
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000
        })
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch market data for the given symbol and timeframe"""
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()

    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent trades for the given symbol"""
        try:
            # Fetch recent trades
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching recent trades: {str(e)}")
            return pd.DataFrame()

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Fetch order book for the given symbol"""
        try:
            # Fetch order book
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            
            # Convert to DataFrame
            bids_df = pd.DataFrame(orderbook['bids'], columns=['price', 'amount'])
            asks_df = pd.DataFrame(orderbook['asks'], columns=['price', 'amount'])
            
            return {
                'bids': bids_df,
                'asks': asks_df,
                'timestamp': datetime.fromtimestamp(orderbook['timestamp'] / 1000)
            }
            
        except Exception as e:
            print(f"Error fetching order book: {str(e)}")
            return {}

    def get_ticker(self, symbol: str) -> dict:
        """Fetch current ticker data for the given symbol"""
        try:
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'change_24h': ticker['percentage'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
            
        except Exception as e:
            print(f"Error fetching ticker: {str(e)}")
            return {}

    def get_historical_data(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical data for the given symbol and time range"""
        try:
            # Calculate number of candles needed
            timeframe_seconds = self.timeframes[timeframe]
            total_seconds = (end_time - start_time).total_seconds()
            num_candles = int(total_seconds / timeframe_seconds)
            
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=num_candles
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()

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

class MarketDataManager:
    def __init__(self):
        """Initialize market data manager"""
        self.market_data = MarketData({
            'exchange': 'binance',
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
        })
        self.symbol = "BTC/USDT"  # Default symbol

    def get_market_data(self, symbol: str = None) -> Dict:
        """Get current market data for the specified symbol"""
        try:
            if symbol:
                self.symbol = symbol
            
            # Get current price and 24h change
            ticker = self.market_data.get_ticker(self.symbol)
            current_price = float(ticker['last'])
            price_change_24h = float(ticker['change_24h'])

            # Get RSI from 1h timeframe
            df_1h = self.market_data.get_market_data(self.symbol, timeframe='1h', limit=100)
            if not df_1h.empty:
                rsi = float(df_1h['RSI'].iloc[-1]) if 'RSI' in df_1h.columns else 0
            else:
                rsi = 0

            return {
                'symbol': self.symbol,
                'current_price': current_price,
                'price_change_24h': price_change_24h,
                'rsi': rsi,
                'volume_24h': float(ticker['volume'])
            }
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return {
                'symbol': self.symbol,
                'current_price': 0,
                'price_change_24h': 0,
                'rsi': 0,
                'volume_24h': 0
            }

    def set_symbol(self, symbol: str):
        """Set the current trading symbol"""
        self.symbol = symbol 