import pandas as pd
import numpy as np
# import talib

class TechnicalAnalysis:
    def __init__(self, config: dict):
        """Initialize technical analysis with configuration"""
        self.config = config
        self.indicators_config = config.get('technical_indicators', {})

    def calculate_indicators(self, market_data: pd.DataFrame) -> dict:
        """Calculate technical indicators for the given market data"""
        try:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                raise ValueError("Market data missing required columns")

            # Convert column names to lowercase
            market_data.columns = market_data.columns.str.lower()
            
            # Calculate basic indicators
            indicators = {
                'rsi': self._calculate_rsi(market_data),
                'macd': self._calculate_macd(market_data),
                'bollinger_bands': self._calculate_bollinger_bands(market_data),
                'moving_averages': self._calculate_moving_averages(market_data),
                'volume_indicators': self._calculate_volume_indicators(market_data)
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _calculate_rsi(self, market_data: pd.DataFrame) -> dict:
        """Calculate RSI indicator"""
        try:
            rsi = talib.RSI(market_data['close'])
            return {
                'rsi': rsi.iloc[-1],
                'rsi_ma': talib.SMA(rsi, timeperiod=14).iloc[-1],
                'rsi_trend': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral'
            }
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return {}

    def _calculate_macd(self, market_data: pd.DataFrame) -> dict:
        """Calculate MACD indicator"""
        try:
            macd, signal, hist = talib.MACD(market_data['close'])
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': hist.iloc[-1],
                'trend': 'bullish' if hist.iloc[-1] > 0 else 'bearish'
            }
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            return {}

    def _calculate_bollinger_bands(self, market_data: pd.DataFrame) -> dict:
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(market_data['close'])
            current_price = market_data['close'].iloc[-1]
            return {
                'upper': upper.iloc[-1],
                'middle': middle.iloc[-1],
                'lower': lower.iloc[-1],
                'position': 'upper' if current_price > upper.iloc[-1] else 'lower' if current_price < lower.iloc[-1] else 'middle'
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            return {}

    def _calculate_moving_averages(self, market_data: pd.DataFrame) -> dict:
        """Calculate various moving averages"""
        try:
            sma_20 = talib.SMA(market_data['close'], timeperiod=20)
            sma_50 = talib.SMA(market_data['close'], timeperiod=50)
            sma_200 = talib.SMA(market_data['close'], timeperiod=200)
            
            current_price = market_data['close'].iloc[-1]
            
            return {
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'sma_200': sma_200.iloc[-1],
                'trend': self._determine_ma_trend(current_price, sma_20.iloc[-1], sma_50.iloc[-1], sma_200.iloc[-1])
            }
        except Exception as e:
            print(f"Error calculating Moving Averages: {str(e)}")
            return {}

    def _calculate_volume_indicators(self, market_data: pd.DataFrame) -> dict:
        """Calculate volume-based indicators"""
        try:
            obv = talib.OBV(market_data['close'], market_data['volume'])
            ad = talib.AD(market_data['high'], market_data['low'], market_data['close'], market_data['volume'])
            
            return {
                'obv': obv.iloc[-1],
                'obv_trend': 'increasing' if obv.iloc[-1] > obv.iloc[-2] else 'decreasing',
                'ad': ad.iloc[-1],
                'ad_trend': 'increasing' if ad.iloc[-1] > ad.iloc[-2] else 'decreasing'
            }
        except Exception as e:
            print(f"Error calculating Volume Indicators: {str(e)}")
            return {}

    def _determine_ma_trend(self, current_price: float, sma_20: float, sma_50: float, sma_200: float) -> str:
        """Determine trend based on moving averages"""
        if current_price > sma_20 > sma_50 > sma_200:
            return 'strong_bullish'
        elif current_price > sma_20 > sma_50:
            return 'bullish'
        elif current_price < sma_20 < sma_50 < sma_200:
            return 'strong_bearish'
        elif current_price < sma_20 < sma_50:
            return 'bearish'
        else:
            return 'neutral' 