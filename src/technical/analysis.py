import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple
import logging
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
        self.fair_value_gaps = []
        self.liquidity_sweeps = []
        self.order_blocks = []
        self.supply_zones = []
        self.demand_zones = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate essential technical indicators
        """
        try:
            if df is None or df.empty:
                self.logger.warning("Input DataFrame is None or empty")
                return pd.DataFrame()
            
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Ensure we have enough data points
            if len(df) < 50:  # Minimum required for all indicators
                self.logger.warning(f"Insufficient data points: {len(df)}")
                return pd.DataFrame()
            
            # Volume indicators first (since other calculations depend on it)
            try:
                df['Volume_MA'] = ta.sma(df['volume'], length=20)
                df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
            except Exception as e:
                self.logger.error(f"Error calculating volume indicators: {str(e)}")
                return pd.DataFrame()
            
            # RSI with multiple timeframes
            try:
                df['RSI'] = ta.rsi(df['close'])
                df['RSI_MA'] = ta.sma(df['RSI'], length=14)
            except Exception as e:
                self.logger.error(f"Error calculating RSI: {str(e)}")
                return pd.DataFrame()
            
            # MACD with multiple settings
            try:
                macd = ta.macd(df['close'])
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Hist'] = macd['MACDh_12_26_9']
            except Exception as e:
                self.logger.error(f"Error calculating MACD: {str(e)}")
                return pd.DataFrame()
            
            # Moving Averages for different timeframes
            try:
                df['EMA_8'] = ta.ema(df['close'], length=8)    # Short-term trend
                df['EMA_21'] = ta.ema(df['close'], length=21)
                df['EMA_50'] = ta.ema(df['close'], length=50)  # Long-term trend
            except Exception as e:
                self.logger.error(f"Error calculating EMAs: {str(e)}")
                return pd.DataFrame()
            
            # Bollinger Bands
            try:
                bollinger = ta.bbands(df['close'], length=20)
                df['BB_Upper'] = bollinger['BBU_20_2.0']
                df['BB_Middle'] = bollinger['BBM_20_2.0']
                df['BB_Lower'] = bollinger['BBL_20_2.0']
            except Exception as e:
                self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
                return pd.DataFrame()
            
            # Store the indicators for reference
            self.indicators = {
                'RSI': df['RSI'],
                'RSI_MA': df['RSI_MA'],
                'MACD': df['MACD'],
                'MACD_Signal': df['MACD_Signal'],
                'MACD_Hist': df['MACD_Hist'],
                'EMA_8': df['EMA_8'],
                'EMA_21': df['EMA_21'],
                'EMA_50': df['EMA_50'],
                'BB_Upper': df['BB_Upper'],
                'BB_Middle': df['BB_Middle'],
                'BB_Lower': df['BB_Lower'],
                'Volume_MA': df['Volume_MA'],
                'Volume_Ratio': df['Volume_Ratio']
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            self.logger.debug("Debug info:")
            if df is not None:
                self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                self.logger.debug(f"DataFrame shape: {df.shape}")
                self.logger.debug(f"DataFrame head:\n{df.head()}")
            return pd.DataFrame()
    
    def identify_order_blocks(
        self,
        df: pd.DataFrame,
        window: int = 20,
        timeframe: str = '15m'
    ) -> List[Dict]:
        """
        Identify order blocks using SMC principles
        Adjust window size based on timeframe
        """
        # Clear previous order blocks for this timeframe
        self.order_blocks = [ob for ob in self.order_blocks if ob['timeframe'] != timeframe]
        
        # Adjust window size based on timeframe
        if timeframe == '15m':
            window = 10  # Look at last 10 candles for 15m
        elif timeframe == '5m':
            window = 15  # Look at last 15 candles for 5m
        elif timeframe == '3m':
            window = 20  # Look at last 20 candles for 3m
        
        order_blocks = []
        
        # Ensure we have enough data
        if len(df) < window * 2:
            return order_blocks
        
        for i in range(window, len(df) - window):
            # Check for significant volume (adjust threshold based on timeframe)
            volume_threshold = df['Volume_MA'].iloc[i] * (1.5 if timeframe == '15m' else 1.2)
            
            # Bullish Order Block (ICT principles)
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                df['volume'].iloc[i] > volume_threshold and    # High volume
                df['low'].iloc[i] < df['low'].iloc[i-1] and    # Lower low
                df['low'].iloc[i] < df['low'].iloc[i+1] and    # Lower low
                df['close'].iloc[i] > df['EMA_21'].iloc[i] and # Price above EMA21
                df['RSI'].iloc[i] < 40):  # RSI oversold condition
                
                # Validate with surrounding price action
                if self._validate_bullish_ob(df, i, window):
                    ob = {
                        'type': 'bullish',
                        'timestamp': df.index[i],
                        'price': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'strength': self._calculate_ob_strength(df, i, 'bullish'),
                        'timeframe': timeframe
                    }
                    order_blocks.append(ob)
                    self.order_blocks.append(ob)
            
            # Bearish Order Block (ICT principles)
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                df['volume'].iloc[i] > volume_threshold and    # High volume
                df['high'].iloc[i] > df['high'].iloc[i-1] and  # Higher high
                df['high'].iloc[i] > df['high'].iloc[i+1] and  # Higher high
                df['close'].iloc[i] < df['EMA_21'].iloc[i] and # Price below EMA21
                df['RSI'].iloc[i] > 60):  # RSI overbought condition
                
                # Validate with surrounding price action
                if self._validate_bearish_ob(df, i, window):
                    ob = {
                        'type': 'bearish',
                        'timestamp': df.index[i],
                        'price': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'strength': self._calculate_ob_strength(df, i, 'bearish'),
                        'timeframe': timeframe
                    }
                    order_blocks.append(ob)
                    self.order_blocks.append(ob)
        
        # Sort by strength and return only the strongest blocks
        order_blocks.sort(key=lambda x: x['strength'], reverse=True)
        return order_blocks[:3]  # Return only top 3 strongest blocks for scalping
    
    def identify_supply_demand_zones(
        self,
        df: pd.DataFrame,
        window: int = 20,
        timeframe: str = '15m'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify supply and demand zones using SMC principles
        Adjust window size based on timeframe
        """
        # Clear previous zones for this timeframe
        self.supply_zones = [sz for sz in self.supply_zones if sz['timeframe'] != timeframe]
        self.demand_zones = [dz for dz in self.demand_zones if dz['timeframe'] != timeframe]
        
        # Adjust window size based on timeframe
        if timeframe == '15m':
            window = 10
        elif timeframe == '5m':
            window = 15
        elif timeframe == '3m':
            window = 20
        
        supply_zones = []
        demand_zones = []
        
        # Identify CHoCH (Change of Character)
        for i in range(window, len(df) - window):
            # Bullish CHoCH
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['low'].iloc[i] > df['low'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i]):
                ch = {
                    'type': 'CHoCH',
                    'direction': 'bullish',
                    'timestamp': df.index[i],
                    'price': df['high'].iloc[i],
                    'strength': self._calculate_choch_strength(df, i, 'bullish'),
                    'timeframe': timeframe
                }
                self.supply_zones.append(ch)
                supply_zones.append(ch)
            
            # Bearish CHoCH
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['high'].iloc[i] < df['high'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i]):
                ch = {
                    'type': 'CHoCH',
                    'direction': 'bearish',
                    'timestamp': df.index[i],
                    'price': df['low'].iloc[i],
                    'strength': self._calculate_choch_strength(df, i, 'bearish'),
                    'timeframe': timeframe
                }
                self.demand_zones.append(ch)
                demand_zones.append(ch)
        
        # Identify PDL (Previous Day Low) and PDH (Previous Day High)
        for i in range(window, len(df) - window):
            # PDL
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['volume'].iloc[i] > df['Volume_MA'].iloc[i]):
                pdl = {
                    'type': 'PDL',
                    'timestamp': df.index[i],
                    'price': df['low'].iloc[i],
                    'strength': self._calculate_pdl_strength(df, i),
                    'timeframe': timeframe
                }
                self.demand_zones.append(pdl)
                demand_zones.append(pdl)
            
            # PDH
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['volume'].iloc[i] > df['Volume_MA'].iloc[i]):
                pdh = {
                    'type': 'PDH',
                    'timestamp': df.index[i],
                    'price': df['high'].iloc[i],
                    'strength': self._calculate_pdh_strength(df, i),
                    'timeframe': timeframe
                }
                self.supply_zones.append(pdh)
                supply_zones.append(pdh)
        
        # Sort by strength and return only the strongest zones
        supply_zones.sort(key=lambda x: x['strength'], reverse=True)
        demand_zones.sort(key=lambda x: x['strength'], reverse=True)
        return supply_zones[:2], demand_zones[:2]  # Return only top 2 strongest zones for scalping
    
    def _check_fair_value_gap(self, df: pd.DataFrame, index: int, ob_type: str) -> bool:
        """
        Check for fair value gaps (FVG) in ICT principles
        """
        # Check if we have enough future candles
        if index + 2 >= len(df):
            return False
            
        if ob_type == 'bullish':
            # Check for bullish FVG
            if (df['low'].iloc[index+1] > df['high'].iloc[index] and
                df['low'].iloc[index+2] > df['high'].iloc[index]):
                # Store the FVG
                self.fair_value_gaps.append({
                    'type': 'bullish',
                    'price': df['high'].iloc[index],
                    'timeframe': df.index.name
                })
                return True
        else:
            # Check for bearish FVG
            if (df['high'].iloc[index+1] < df['low'].iloc[index] and
                df['high'].iloc[index+2] < df['low'].iloc[index]):
                # Store the FVG
                self.fair_value_gaps.append({
                    'type': 'bearish',
                    'price': df['low'].iloc[index],
                    'timeframe': df.index.name
                })
                return True
        return False
    
    def _check_liquidity(self, df: pd.DataFrame, index: int, zone_type: str) -> bool:
        """
        Check for liquidity sweeps in ICT principles
        """
        # Check if we have enough future candles
        if index + 1 >= len(df):
            return False
            
        if zone_type == 'supply':
            # Check for liquidity sweep above
            if (df['high'].iloc[index+1] > df['high'].iloc[index] and
                df['close'].iloc[index+1] < df['high'].iloc[index]):
                # Store the liquidity sweep
                self.liquidity_sweeps.append({
                    'type': 'supply',
                    'price': df['high'].iloc[index],
                    'timeframe': df.index.name
                })
                return True
        else:
            # Check for liquidity sweep below
            if (df['low'].iloc[index+1] < df['low'].iloc[index] and
                df['close'].iloc[index+1] > df['low'].iloc[index]):
                # Store the liquidity sweep
                self.liquidity_sweeps.append({
                    'type': 'demand',
                    'price': df['low'].iloc[index],
                    'timeframe': df.index.name
                })
                return True
        return False
    
    def _validate_bullish_ob(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """Validate a bullish order block"""
        # Check if price respects the order block
        for i in range(index + 1, min(index + window, len(df))):
            if df['low'].iloc[i] < df['low'].iloc[index]:
                return False
        return True
    
    def _validate_bearish_ob(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """Validate a bearish order block"""
        # Check if price respects the order block
        for i in range(index + 1, min(index + window, len(df))):
            if df['high'].iloc[i] > df['high'].iloc[index]:
                return False
        return True
    
    def _validate_supply_zone(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """Validate a supply zone"""
        # Check if price respects the supply zone
        for i in range(index + 1, min(index + window, len(df))):
            if df['high'].iloc[i] > df['high'].iloc[index]:
                return False
        return True
    
    def _validate_demand_zone(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """Validate a demand zone"""
        # Check if price respects the demand zone
        for i in range(index + 1, min(index + window, len(df))):
            if df['low'].iloc[i] < df['low'].iloc[index]:
                return False
        return True
    
    def _calculate_ob_strength(self, df: pd.DataFrame, index: int, ob_type: str) -> float:
        """Calculate the strength of an order block"""
        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['Volume_MA'].iloc[index]
        
        # Price action strength
        if ob_type == 'bullish':
            body_size = df['close'].iloc[index] - df['open'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        else:  # bearish
            body_size = df['open'].iloc[index] - df['close'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        
        # Combine strengths
        return (volume_strength + price_strength) / 2
    
    def _calculate_zone_strength(self, df: pd.DataFrame, index: int, zone_type: str) -> float:
        """Calculate the strength of a supply/demand zone"""
        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['Volume_MA'].iloc[index]
        
        # Price action strength
        if zone_type == 'supply':
            body_size = df['open'].iloc[index] - df['close'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        else:  # demand
            body_size = df['close'].iloc[index] - df['open'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        
        # Combine strengths
        return (volume_strength + price_strength) / 2
    
    def _calculate_choch_strength(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Calculate the strength of a Change of Character"""
        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['Volume_MA'].iloc[index]
        
        # Price action strength
        if direction == 'bullish':
            body_size = df['close'].iloc[index] - df['open'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        else:  # bearish
            body_size = df['open'].iloc[index] - df['close'].iloc[index]
            total_size = df['high'].iloc[index] - df['low'].iloc[index]
            price_strength = body_size / total_size if total_size > 0 else 0
        
        # Combine strengths
        return (volume_strength + price_strength) / 2
    
    def _calculate_pdl_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a Previous Day Low"""
        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['Volume_MA'].iloc[index]
        
        # Price action strength
        body_size = df['close'].iloc[index] - df['open'].iloc[index]
        total_size = df['high'].iloc[index] - df['low'].iloc[index]
        price_strength = body_size / total_size if total_size > 0 else 0
        
        # Combine strengths
        return (volume_strength + price_strength) / 2
    
    def _calculate_pdh_strength(self, df: pd.DataFrame, index: int) -> float:
        """Calculate the strength of a Previous Day High"""
        # Volume strength
        volume_strength = df['volume'].iloc[index] / df['Volume_MA'].iloc[index]
        
        # Price action strength
        body_size = df['open'].iloc[index] - df['close'].iloc[index]
        total_size = df['high'].iloc[index] - df['low'].iloc[index]
        price_strength = body_size / total_size if total_size > 0 else 0
        
        # Combine strengths
        return (volume_strength + price_strength) / 2 