import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from ..data.models import OrderBlock, FairValueGap, LiquidityLevel

class SMCPatternDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_institutional_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect institutional order blocks with advanced pattern recognition"""
        order_blocks = []
        
        # Validate input data
        if df is None or df.empty:
            return order_blocks
            
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'EMA_21']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing required columns for order block detection: {missing_columns}")
            return order_blocks
        
        # Calculate volume profile and price action
        try:
            df['volume_profile'] = df['volume'] * (df['close'] - df['open'])
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        except Exception as e:
            self.logger.error(f"Error calculating price action metrics: {str(e)}")
            return order_blocks
        
        for i in range(2, len(df) - 2):
            try:
                # Bullish institutional order block
                if self._is_bullish_institutional_block(df, i):
                    block = {
                        'block_type': 'BULLISH',
                        'price': float(df['close'].iloc[i]),
                        'volume': float(df['volume'].iloc[i]),
                        'strength': float(self._calculate_institutional_block_strength(df, i, 'bullish')),
                        'timestamp': str(df.index[i])
                    }
                    order_blocks.append(block)
                
                # Bearish institutional order block
                if self._is_bearish_institutional_block(df, i):
                    block = {
                        'block_type': 'BEARISH',
                        'price': float(df['close'].iloc[i]),
                        'volume': float(df['volume'].iloc[i]),
                        'strength': float(self._calculate_institutional_block_strength(df, i, 'bearish')),
                        'timestamp': str(df.index[i])
                    }
                    order_blocks.append(block)
                    
            except Exception as e:
                self.logger.error(f"Error processing order block at index {i}: {str(e)}")
                continue
        
        return order_blocks
    
    def _is_bullish_institutional_block(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bullish institutional order block"""
        # Check for strong bullish candle
        if not (df['close'].iloc[index] > df['open'].iloc[index] and
                df['body_size'].iloc[index] > df['body_size'].mean() * 0.4):
            return False
        
        # Check for bearish candles after
        if not (df['close'].iloc[index+1] < df['open'].iloc[index+1] and
                df['close'].iloc[index+2] < df['open'].iloc[index+2]):
            return False
        
        # Check for institutional characteristics
        if not (df['volume'].iloc[index] > df['volume'].mean() * 1.02 and
                df['lower_wick'].iloc[index] < df['body_size'].iloc[index] * 0.6):
            return False
        
        return True
    
    def _is_bearish_institutional_block(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bearish institutional order block"""
        # Check for strong bearish candle
        if not (df['close'].iloc[index] < df['open'].iloc[index] and
                df['body_size'].iloc[index] > df['body_size'].mean() * 0.4):
            return False
        
        # Check for bullish candles after
        if not (df['close'].iloc[index+1] > df['open'].iloc[index+1] and
                df['close'].iloc[index+2] > df['open'].iloc[index+2]):
            return False
        
        # Check for institutional characteristics
        if not (df['volume'].iloc[index] > df['volume'].mean() * 1.02 and
                df['upper_wick'].iloc[index] < df['body_size'].iloc[index] * 0.6):
            return False
        
        return True
    
    def detect_smart_money_traps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect smart money traps (liquidity sweeps)"""
        traps = []
        
        for i in range(2, len(df) - 2):
            # Bull trap
            if self._is_bull_trap(df, i):
                traps.append({
                    'type': 'bull_trap',
                    'price': df['high'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': self._calculate_trap_strength(df, i, 'bull')
                })
            
            # Bear trap
            if self._is_bear_trap(df, i):
                traps.append({
                    'type': 'bear_trap',
                    'price': df['low'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': self._calculate_trap_strength(df, i, 'bear')
                })
        
        return traps
    
    def _is_bull_trap(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bull trap"""
        # Check for strong upper wick
        if not (df['upper_wick'].iloc[index] > df['body_size'].iloc[index] * 2):
            return False
        
        # Check for bearish candle after
        if not (df['close'].iloc[index+1] < df['open'].iloc[index+1]):
            return False
        
        # Check for high volume
        if not (df['volume'].iloc[index] > df['volume'].mean() * 1.2):
            return False
        
        return True
    
    def _is_bear_trap(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bear trap"""
        # Check for strong lower wick
        if not (df['lower_wick'].iloc[index] > df['body_size'].iloc[index] * 2):
            return False
        
        # Check for bullish candle after
        if not (df['close'].iloc[index+1] > df['open'].iloc[index+1]):
            return False
        
        # Check for high volume
        if not (df['volume'].iloc[index] > df['volume'].mean() * 1.2):
            return False
        
        return True
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect fair value gaps in price action"""
        fair_value_gaps = []
        
        # Validate input data
        if df is None or df.empty or len(df) < 3:
            return fair_value_gaps
            
        for i in range(1, len(df) - 1):
            try:
                # Bullish FVG
                if self._is_bullish_fvg(df, i):
                    gap = {
                        'gap_type': 'BULLISH',
                        'upper_price': float(df['low'].iloc[i+1]),
                        'lower_price': float(df['high'].iloc[i-1]),
                        'size': float(df['low'].iloc[i+1] - df['high'].iloc[i-1]),
                        'timestamp': str(df.index[i])
                    }
                    fair_value_gaps.append(gap)
                
                # Bearish FVG
                if self._is_bearish_fvg(df, i):
                    gap = {
                        'gap_type': 'BEARISH',
                        'upper_price': float(df['low'].iloc[i-1]),
                        'lower_price': float(df['high'].iloc[i+1]),
                        'size': float(df['low'].iloc[i-1] - df['high'].iloc[i+1]),
                        'timestamp': str(df.index[i])
                    }
                    fair_value_gaps.append(gap)
                    
            except Exception as e:
                self.logger.error(f"Error processing fair value gap at index {i}: {str(e)}")
                continue
        
        return fair_value_gaps
    
    def _is_bullish_fvg(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bullish fair value gap"""
        # Check for gap
        if not (df['low'].iloc[index] > df['high'].iloc[index-1] and
                df['low'].iloc[index] > df['high'].iloc[index+1]):
            return False
        
        # Check gap size (reduced minimum size)
        gap_size = df['low'].iloc[index] - max(df['high'].iloc[index-1], df['high'].iloc[index+1])
        if gap_size < self.config['technical']['smc']['fair_value_gap']['min_size'] * 0.5:  # Reduced from 1.0
            return False
        
        # Check volume (more lenient for scalping)
        if not (df['volume'].iloc[index] > df['volume'].mean() * 0.4):  # Reduced from 0.5
            return False
        
        return True
    
    def _is_bearish_fvg(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a candle forms a bearish fair value gap"""
        # Check for gap
        if not (df['high'].iloc[index] < df['low'].iloc[index-1] and
                df['high'].iloc[index] < df['low'].iloc[index+1]):
            return False
        
        # Check gap size (reduced minimum size)
        gap_size = min(df['low'].iloc[index-1], df['low'].iloc[index+1]) - df['high'].iloc[index]
        if gap_size < self.config['technical']['smc']['fair_value_gap']['min_size'] * 0.5:  # Reduced from 1.0
            return False
        
        # Check volume (more lenient for scalping)
        if not (df['volume'].iloc[index] > df['volume'].mean() * 0.4):  # Reduced from 0.5
            return False
        
        return True
    
    def detect_liquidity_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detect liquidity levels (support/resistance) in price action"""
        liquidity_levels = []
        
        # Validate input data
        if df is None or df.empty:
            return liquidity_levels
            
        for i in range(2, len(df) - 2):
            try:
                # Support level
                if self._is_support_level(df, i):
                    level = {
                        'level_type': 'SUPPORT',
                        'price': float(df['low'].iloc[i]),
                        'strength': float(self._calculate_level_strength(df, i, 'support')),
                        'timestamp': str(df.index[i])
                    }
                    liquidity_levels.append(level)
                
                # Resistance level
                if self._is_resistance_level(df, i):
                    level = {
                        'level_type': 'RESISTANCE',
                        'price': float(df['high'].iloc[i]),
                        'strength': float(self._calculate_level_strength(df, i, 'resistance')),
                        'timestamp': str(df.index[i])
                    }
                    liquidity_levels.append(level)
                    
            except Exception as e:
                self.logger.error(f"Error processing liquidity level at index {i}: {str(e)}")
                continue
        
        return liquidity_levels
    
    def _is_support_level(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a price level forms support"""
        # Check for local minimum
        if not (df['low'].iloc[index] < df['low'].iloc[index-1] and
                df['low'].iloc[index] < df['low'].iloc[index+1]):
            return False
        
        # Check for volume (reduced requirement)
        if not (df['volume'].iloc[index] > self.config['technical']['smc']['liquidity_level']['min_volume'] * 0.6):  # Reduced from 0.8
            return False
        
        # Check for touches with allowed deviation (increased deviation)
        price_level = df['low'].iloc[index]
        max_deviation = self.config['technical']['smc']['liquidity_level']['max_deviation'] * 1.5  # Increased from 1.2
        touches = sum(abs(df['low'] - price_level) <= max_deviation)
        if touches < self.config['technical']['smc']['liquidity_level']['min_touches']:
            return False
        
        return True
    
    def _is_resistance_level(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a price level forms resistance"""
        # Check for local maximum
        if not (df['high'].iloc[index] > df['high'].iloc[index-1] and
                df['high'].iloc[index] > df['high'].iloc[index+1]):
            return False
        
        # Check for volume (reduced requirement)
        if not (df['volume'].iloc[index] > self.config['technical']['smc']['liquidity_level']['min_volume'] * 0.6):  # Reduced from 0.8
            return False
        
        # Check for touches with allowed deviation (increased deviation)
        price_level = df['high'].iloc[index]
        max_deviation = self.config['technical']['smc']['liquidity_level']['max_deviation'] * 1.5  # Increased from 1.2
        touches = sum(abs(df['high'] - price_level) <= max_deviation)
        if touches < self.config['technical']['smc']['liquidity_level']['min_touches']:
            return False
        
        return True
    
    def _calculate_institutional_block_strength(self,
                                              df: pd.DataFrame,
                                              index: int,
                                              block_type: str) -> float:
        """Calculate the strength of an institutional order block"""
        volume = df['volume'].iloc[index]
        body_size = df['body_size'].iloc[index]
        avg_volume = df['volume'].mean()
        avg_body_size = df['body_size'].mean()
        
        # Normalize strength between 0 and 1
        volume_strength = min(volume / avg_volume, 1.0)
        body_strength = min(body_size / avg_body_size, 1.0)
        
        return (volume_strength + body_strength) / 2
    
    def _calculate_trap_strength(self,
                               df: pd.DataFrame,
                               index: int,
                               trap_type: str) -> float:
        """Calculate the strength of a smart money trap"""
        volume = df['volume'].iloc[index]
        avg_volume = df['volume'].mean()
        
        if trap_type == 'bull':
            wick_strength = df['upper_wick'].iloc[index] / df['body_size'].iloc[index]
        else:  # bear
            wick_strength = df['lower_wick'].iloc[index] / df['body_size'].iloc[index]
        
        # Normalize strength between 0 and 1
        volume_strength = min(volume / avg_volume, 1.0)
        wick_strength = min(wick_strength, 1.0)
        
        return (volume_strength + wick_strength) / 2
    
    def _calculate_level_strength(self,
                                df: pd.DataFrame,
                                index: int,
                                level_type: str) -> float:
        """Calculate the strength of a liquidity level"""
        volume = df['volume'].iloc[index]
        avg_volume = df['volume'].mean()
        
        # Count touches of this level
        if level_type == 'support':
            touches = sum(df['low'] == df['low'].iloc[index])
        else:  # resistance
            touches = sum(df['high'] == df['high'].iloc[index])
        
        # Normalize strength between 0 and 1
        volume_strength = min(volume / avg_volume, 1.0)
        touch_strength = min(touches / len(df), 1.0)
        
        return (volume_strength + touch_strength) / 2
    
    def analyze_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze all SMC patterns in the market data"""
        try:
            # Detect institutional order blocks
            order_blocks = self.detect_institutional_order_blocks(df)
            
            # Detect smart money traps
            smart_money_traps = self.detect_smart_money_traps(df)
            
            # Detect fair value gaps
            fair_value_gaps = self.detect_fair_value_gaps(df)
            
            # Detect liquidity levels
            liquidity_levels = self.detect_liquidity_levels(df)
            
            # Categorize liquidity levels into supply/demand zones
            supply_zones = [level for level in liquidity_levels if level['level_type'] == 'RESISTANCE']
            demand_zones = [level for level in liquidity_levels if level['level_type'] == 'SUPPORT']
            
            # Add timeframe information to each pattern
            for pattern in order_blocks + smart_money_traps + fair_value_gaps + supply_zones + demand_zones:
                pattern['timeframe'] = timeframe
            
            return {
                'order_blocks': order_blocks,
                'smart_money_traps': smart_money_traps,
                'fair_value_gaps': fair_value_gaps,
                'liquidity_levels': liquidity_levels,
                'supply_zones': supply_zones,
                'demand_zones': demand_zones
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing SMC patterns: {str(e)}")
            return {
                'order_blocks': [],
                'smart_money_traps': [],
                'fair_value_gaps': [],
                'liquidity_levels': [],
                'supply_zones': [],
                'demand_zones': []
            } 