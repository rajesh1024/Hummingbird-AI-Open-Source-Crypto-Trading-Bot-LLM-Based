import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from ..data.models import MarketStructureData, MarketStructure, OrderBlock, FairValueGap, LiquidityLevel
from ..data.database import DatabaseManager
from .smc_patterns import SMCPatternDetector
from datetime import datetime
import logging
import ta.trend as ta_trend

class MarketStructure(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class MarketStructureAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.smc_detector = SMCPatternDetector(config)
        self.logger = logging.getLogger(__name__)
    
    def analyze_market_structure(self, market_data: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze market structure and identify key levels"""
        try:
            if market_data.empty:
                raise ValueError("Empty market data")

            # Get current price and technical indicators
            current_price = float(market_data['close'].iloc[-1])
            technical_indicators = {
                'RSI': float(market_data['RSI'].iloc[-1]) if 'RSI' in market_data else None,
                'MACD': float(market_data['MACD'].iloc[-1]) if 'MACD' in market_data else None,
                'MACD_Signal': float(market_data['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in market_data else None,
                'MACD_Hist': float(market_data['MACD_Hist'].iloc[-1]) if 'MACD_Hist' in market_data else None,
                'EMA_8': float(market_data['EMA_8'].iloc[-1]) if 'EMA_8' in market_data else None,
                'EMA_21': float(market_data['EMA_21'].iloc[-1]) if 'EMA_21' in market_data else None
            }
            
            # Find SMC patterns
            order_blocks = self._find_order_blocks(market_data)
            liquidity_levels = self._find_liquidity_levels(market_data)
            fair_value_gaps = self._find_fair_value_gaps(market_data)
            
            # Process order blocks
            processed_blocks = []
            for block in order_blocks:
                block_data = {
                    'type': block['type'],
                    'price': float(block['entry']),
                    'volume': float(block['volume']),
                    'strength': self._calculate_block_strength(block, market_data),
                    'timeframe': timeframe
                }
                processed_blocks.append(block_data)
            
            # Process liquidity levels
            processed_levels = []
            for level in liquidity_levels:
                level_data = {
                    'type': level['level_type'],
                    'price': float(level['price']),
                    'volume': float(level['volume']),
                    'strength': self._calculate_level_strength(level['price'], market_data, timeframe),
                    'timeframe': timeframe
                }
                processed_levels.append(level_data)
            
            # Process fair value gaps
            processed_gaps = []
            for gap in fair_value_gaps:
                gap_data = {
                    'type': gap['type'],
                    'upper_price': float(gap['upper']),
                    'lower_price': float(gap['lower']),
                    'gap_size': float(gap['upper'] - gap['lower']),
                    'timeframe': timeframe,
                    'imbalance': self._calculate_gap_imbalance(gap, market_data)
                }
                processed_gaps.append(gap_data)
            
            # Determine market structure type
            structure_type = self._determine_market_structure(market_data['close'])
            
            # Create market context
            market_context = {
                'current_price': current_price,
                'technical_indicators': technical_indicators,
                'market_structure': structure_type.value if isinstance(structure_type, Enum) else structure_type,
                'timeframe': timeframe,
                'smc_data': {
                    'order_blocks': processed_blocks,
                    'liquidity_levels': processed_levels,
                    'fair_value_gaps': processed_gaps,
                    'supply_zones': [],  # Will be populated by SMC detector
                    'demand_zones': [],  # Will be populated by SMC detector
                    'smart_money_traps': []  # Will be populated by SMC detector
                }
            }
            
            # Get additional patterns from SMC detector
            smc_patterns = self.smc_detector.analyze_patterns(market_data, timeframe)
            if smc_patterns:
                market_context['smc_data']['supply_zones'] = smc_patterns.get('supply_zones', [])
                market_context['smc_data']['demand_zones'] = smc_patterns.get('demand_zones', [])
                market_context['smc_data']['smart_money_traps'] = smc_patterns.get('smart_money_traps', [])
            
            return market_context
            
        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {str(e)}")
            return {
                'current_price': current_price if 'current_price' in locals() else 0.0,
                'technical_indicators': technical_indicators if 'technical_indicators' in locals() else {},
                'market_structure': structure_type.value if isinstance(structure_type, Enum) else 'UNKNOWN',
                'timeframe': timeframe,
                'smc_data': {
                    'order_blocks': [],
                    'liquidity_levels': [],
                    'fair_value_gaps': [],
                    'supply_zones': [],
                    'demand_zones': [],
                    'smart_money_traps': []
                }
            }

    def _calculate_level_strength(self, price: float, data: pd.DataFrame, timeframe: str) -> float:
        """Calculate strength of a price level based on multiple factors"""
        strength = 0.0
        
        # Historical significance (more touches = stronger)
        touches = len(data[abs(data['close'] - price) / price < 0.001])
        strength += min(touches * 0.1, 0.5)  # Cap at 0.5
        
        # Recent reaction strength
        recent_reactions = self._analyze_recent_reactions(price, data)
        strength += recent_reactions * 0.3  # Max 0.3
        
        # Volume confirmation
        volume_score = self._analyze_volume_confirmation(price, data)
        strength += volume_score * 0.2  # Max 0.2
        
        return min(strength, 1.0)  # Normalize to 0-1

    def _analyze_volume_confirmation(self, price: float, data: pd.DataFrame) -> float:
        """Analyze volume confirmation at price level"""
        price_range = data[abs(data['close'] - price) / price < 0.002]
        if len(price_range) == 0:
            return 0.0
            
        avg_volume = price_range['volume'].mean()
        total_avg_volume = data['volume'].mean()
        
        return min(avg_volume / total_avg_volume, 1.0)

    def _analyze_recent_reactions(self, price: float, data: pd.DataFrame) -> float:
        """Analyze strength of recent price reactions at level"""
        recent_data = data.tail(100)  # Look at last 100 candles
        reactions = 0
        
        for idx in range(1, len(recent_data) - 1):
            if (abs(recent_data.iloc[idx]['high'] - price) / price < 0.001 or 
                abs(recent_data.iloc[idx]['low'] - price) / price < 0.001):
                # Calculate price movement after touch
                next_candle = recent_data.iloc[idx + 1]
                prev_candle = recent_data.iloc[idx - 1]
                
                if abs(next_candle['close'] - prev_candle['close']) / price > 0.002:
                    reactions += 1
                    
        return min(reactions * 0.1, 0.3)  # Scale and cap at 0.3

    def _find_confluence_factors(self, price: float, data: pd.DataFrame, timeframe: str) -> List[str]:
        """Find confluence factors for a price level"""
        factors = []
        
        # Check round numbers
        if abs(price % 100) < 1:
            factors.append('ROUND_100')
        elif abs(price % 50) < 0.5:
            factors.append('ROUND_50')
        elif abs(price % 10) < 0.1:
            factors.append('ROUND_10')
            
        # Check EMAs
        ema_periods = [20, 50, 200]
        for period in ema_periods:
            ema = ta_trend.ema_indicator(data['close'], window=period)
            if abs(ema.iloc[-1] - price) / price < 0.001:
                factors.append(f'EMA_{period}')
                
        # Check volume profile
        if self._is_high_volume_node(price, data):
            factors.append('HIGH_VOLUME_NODE')
            
        return factors

    def _is_high_volume_node(self, price: float, data: pd.DataFrame) -> bool:
        """Check if price level is a high volume node"""
        price_range = data[abs(data['close'] - price) / price < 0.002]
        if len(price_range) == 0:
            return False
            
        avg_volume = price_range['volume'].mean()
        total_avg_volume = data['volume'].mean()
        
        return avg_volume > total_avg_volume * 1.5

    def _determine_market_structure(self, close_prices: pd.Series) -> MarketStructure:
        """Determine the overall market structure type"""
        try:
            # Calculate EMAs directly on the Series using ta.trend
            ema_8 = ta_trend.ema_indicator(close_prices, window=8)
            ema_21 = ta_trend.ema_indicator(close_prices, window=21)
            
            if ema_8 is None or ema_21 is None:
                return MarketStructure.NEUTRAL
            
            last_ema_8 = ema_8.iloc[-1]
            last_ema_21 = ema_21.iloc[-1]
            last_close = close_prices.iloc[-1]
            
            # Strong trend conditions
            if last_ema_8 > last_ema_21 and last_close > last_ema_8:
                return MarketStructure.BULLISH
            elif last_ema_8 < last_ema_21 and last_close < last_ema_8:
                return MarketStructure.BEARISH
            
            return MarketStructure.NEUTRAL
            
        except Exception as e:
            self.logger.error(f"Error determining structure type: {str(e)}")
            return MarketStructure.NEUTRAL
    
    def detect_structure_shift(self,
                             current_structure: MarketStructureData,
                             previous_structure: MarketStructureData) -> bool:
        """Detect if market structure has shifted"""
        # Check structure type change
        if current_structure.structure_type != previous_structure.structure_type:
            return True
        
        # Check for significant price movement
        price_change = abs(current_structure.high - previous_structure.high)
        avg_price = (current_structure.high + previous_structure.high) / 2
        price_change_percent = price_change / avg_price
        
        # Check for significant volume change
        volume_change = abs(current_structure.volume - previous_structure.volume)
        avg_volume = (current_structure.volume + previous_structure.volume) / 2
        volume_change_percent = volume_change / avg_volume
        
        # Structure shift if either price or volume change is significant
        return (price_change_percent > 0.02 or  # 2% price change
                volume_change_percent > 0.5)    # 50% volume change
    
    def validate_position_strength(self,
                                 position_type: str,
                                 market_structure: MarketStructureData) -> float:
        """Validate the strength of a position based on market structure"""
        strength = 0.0
        
        # Check market structure alignment
        if (position_type == 'LONG' and market_structure.structure_type == MarketStructure.BULLISH) or \
           (position_type == 'SHORT' and market_structure.structure_type == MarketStructure.BEARISH):
            strength += 0.3
        
        # Check order blocks
        for block in market_structure.order_blocks:
            if (position_type == 'LONG' and block.block_type == 'institutional') or \
               (position_type == 'SHORT' and block.block_type == 'institutional'):
                strength += 0.2 * block.strength
        
        # Check fair value gaps
        for fvg in market_structure.fair_value_gaps:
            if (position_type == 'LONG' and fvg.gap_type == 'bullish') or \
               (position_type == 'SHORT' and fvg.gap_type == 'bearish'):
                strength += 0.2
        
        # Check liquidity levels
        for level in market_structure.liquidity_levels:
            if (position_type == 'LONG' and level.level_type == 'support') or \
               (position_type == 'SHORT' and level.level_type == 'resistance'):
                strength += 0.3 * level.strength
        
        # Check smart money traps
        for trap in market_structure.smart_money_traps:
            if (position_type == 'LONG' and trap['type'] == 'bear_trap') or \
               (position_type == 'SHORT' and trap['type'] == 'bull_trap'):
                strength += 0.2 * trap['strength']
        
        return min(strength, 1.0)

    def _find_liquidity_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Find liquidity levels in the market data"""
        levels = []
        window = 5  # Window size for local extremes
        
        for i in range(window, len(data) - window):
            high_prices = data['high'].iloc[i-window:i+window]
            low_prices = data['low'].iloc[i-window:i+window]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            current_volume = data['volume'].iloc[i]
            
            # Check for resistance level
            if current_high == high_prices.max():
                levels.append({
                    'level_type': 'RESISTANCE',
                    'price': float(current_high),
                    'timestamp': data.index[i],
                    'volume': float(current_volume),
                    'strength': 0.0  # Will be calculated later
                })
            
            # Check for support level
            if current_low == low_prices.min():
                levels.append({
                    'level_type': 'SUPPORT',
                    'price': float(current_low),
                    'timestamp': data.index[i],
                    'volume': float(current_volume),
                    'strength': 0.0  # Will be calculated later
                })
        
        return levels

    def _find_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Find order blocks in the market data"""
        blocks = []
        for i in range(2, len(data) - 1):
            current_candle = data.iloc[i]
            prev_candle = data.iloc[i-1]
            next_candle = data.iloc[i+1]
            
            # Bullish order block
            if (current_candle['close'] > current_candle['open'] and  # Bullish candle
                next_candle['high'] > current_candle['high'] and      # Price moves up
                current_candle['volume'] > prev_candle['volume']):    # Higher volume
                blocks.append({
                    'index': data.index[i],
                    'entry': float(current_candle['high']),
                    'exit': float(current_candle['low']),
                    'volume': float(current_candle['volume']),
                    'type': 'BULLISH'
                })
            
            # Bearish order block
            elif (current_candle['close'] < current_candle['open'] and  # Bearish candle
                  next_candle['low'] < current_candle['low'] and        # Price moves down
                  current_candle['volume'] > prev_candle['volume']):    # Higher volume
                blocks.append({
                    'index': data.index[i],
                    'entry': float(current_candle['low']),
                    'exit': float(current_candle['high']),
                    'volume': float(current_candle['volume']),
                    'type': 'BEARISH'
                })
        
        return blocks

    def _find_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict]:
        """Find fair value gaps in the market data"""
        gaps = []
        for i in range(1, len(data) - 1):
            current_candle = data.iloc[i]
            prev_candle = data.iloc[i-1]
            next_candle = data.iloc[i+1]
            
            # Bullish FVG
            if prev_candle['low'] > next_candle['high']:
                gaps.append({
                    'type': 'bullish',
                    'upper': float(prev_candle['low']),
                    'lower': float(next_candle['high']),
                    'timestamp': data.index[i],
                    'volume': float(current_candle['volume'])
                })
            
            # Bearish FVG
            elif prev_candle['high'] < next_candle['low']:
                gaps.append({
                    'type': 'bearish',
                    'upper': float(next_candle['low']),
                    'lower': float(prev_candle['high']),
                    'timestamp': data.index[i],
                    'volume': float(current_candle['volume'])
                })
        
        return gaps

    def _calculate_block_strength(self, block: Dict, data: pd.DataFrame) -> float:
        """Calculate the strength of an order block"""
        try:
            block_idx = data.index.get_loc(block['index'])
            subsequent_data = data.iloc[block_idx+1:]
            
            # Base strength starts at 0.3
            strength = 0.3
            
            # Add strength based on volume
            avg_volume = data['volume'].mean()
            volume_factor = min(block['volume'] / avg_volume, 2.0)  # Cap at 2x average
            strength += 0.2 * volume_factor
            
            # Add strength based on subsequent tests
            tests = 0
            for _, candle in subsequent_data.iterrows():
                if block['type'] == 'BULLISH':
                    if candle['low'] <= block['exit'] <= candle['high']:
                        tests += 1
                else:  # BEARISH
                    if candle['low'] <= block['entry'] <= candle['high']:
                        tests += 1
            
            strength += min(0.1 * tests, 0.3)  # Cap at 0.3 for tests
            
            # Add strength based on recency
            recency = 1.0 - (len(data) - block_idx) / len(data)
            strength += 0.2 * recency
            
            return min(strength, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating block strength: {str(e)}")
            return 0.0

    def _calculate_imbalance(self, block: Dict, data: pd.DataFrame) -> float:
        """Calculate the imbalance of an order block"""
        try:
            block_idx = data.index.get_loc(block['index'])
            block_candle = data.iloc[block_idx]
            
            # Calculate price imbalance
            price_range = abs(block['entry'] - block['exit'])
            avg_range = data['high'].sub(data['low']).mean()
            
            # Calculate volume imbalance
            volume_ratio = block['volume'] / data['volume'].mean()
            
            # Combine both factors
            imbalance = (price_range / avg_range + volume_ratio) / 2
            return min(imbalance, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating imbalance: {str(e)}")
            return 0.0

    def _calculate_volume_ratio(self, block: Dict, data: pd.DataFrame) -> float:
        """Calculate the volume ratio of an order block"""
        try:
            block_idx = data.index.get_loc(block['index'])
            avg_volume = data.iloc[max(0, block_idx-10):block_idx]['volume'].mean()
            
            if avg_volume > 0:
                return min(block['volume'] / avg_volume, 3.0)  # Cap at 3x average
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volume ratio: {str(e)}")
            return 1.0

    def _count_retests(self, block: Dict, data: pd.DataFrame) -> int:
        """Count how many times a block has been retested"""
        try:
            block_idx = data.index.get_loc(block['index'])
            subsequent_data = data.iloc[block_idx+1:]
            
            tests = 0
            for _, candle in subsequent_data.iterrows():
                if block['type'] == 'BULLISH':
                    if candle['low'] <= block['exit'] <= candle['high']:
                        tests += 1
                else:  # BEARISH
                    if candle['low'] <= block['entry'] <= candle['high']:
                        tests += 1
            
            return tests
            
        except Exception as e:
            self.logger.error(f"Error counting retests: {str(e)}")
            return 0

    def _calculate_gap_imbalance(self, gap: Dict, data: pd.DataFrame) -> float:
        """Calculate the imbalance of a fair value gap"""
        try:
            gap_size = abs(gap['upper'] - gap['lower'])
            avg_range = data['high'].sub(data['low']).mean()
            
            return min(gap_size / avg_range, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating gap imbalance: {str(e)}")
            return 0.0

    def _calculate_gap_fill(self, gap: Dict, data: pd.DataFrame) -> float:
        """Calculate how much of a gap has been filled"""
        try:
            gap_idx = data.index.get_loc(gap['timestamp'])
            subsequent_data = data.iloc[gap_idx+1:]
            
            gap_size = abs(gap['upper'] - gap['lower'])
            if gap_size == 0:
                return 1.0
                
            # Check how much of the gap has been filled
            for _, candle in subsequent_data.iterrows():
                if candle['low'] <= gap['lower'] and candle['high'] >= gap['upper']:
                    return 1.0  # Fully filled
                    
                if gap['type'] == 'bullish':
                    if candle['high'] > gap['lower']:
                        filled = (candle['high'] - gap['lower']) / gap_size
                        return min(filled, 1.0)
                else:  # bearish
                    if candle['low'] < gap['upper']:
                        filled = (gap['upper'] - candle['low']) / gap_size
                        return min(filled, 1.0)
            
            return 0.0  # Not filled at all
            
        except Exception as e:
            self.logger.error(f"Error calculating gap fill: {str(e)}")
            return 0.0 

    def _analyze_volume_at_level(self, price: float, data: pd.DataFrame) -> Dict:
        """Analyze volume profile at a given price level"""
        try:
            # Define price range around the level (0.1% range)
            price_range = price * 0.001
            level_data = data[
                (data['high'] >= price - price_range) & 
                (data['low'] <= price + price_range)
            ]
            
            if level_data.empty:
                return {
                    'volume_profile': 'LOW',
                    'avg_volume': 0.0,
                    'volume_ratio': 0.0
                }
            
            level_volume = level_data['volume'].mean()
            total_avg_volume = data['volume'].mean()
            volume_ratio = level_volume / total_avg_volume if total_avg_volume > 0 else 0
            
            # Categorize volume profile
            if volume_ratio > 1.5:
                profile = 'HIGH'
            elif volume_ratio > 0.75:
                profile = 'MEDIUM'
            else:
                profile = 'LOW'
            
            return {
                'volume_profile': profile,
                'avg_volume': float(level_volume),
                'volume_ratio': float(volume_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume at level: {str(e)}")
            return {
                'volume_profile': 'LOW',
                'avg_volume': 0.0,
                'volume_ratio': 0.0
            }