import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from ..data.database import DatabaseManager
from .smc_patterns import SMCPatternDetector
from datetime import datetime
import logging
import ta.trend as ta_trend

class MarketStructureType(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class MarketStructureAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.smc_detector = SMCPatternDetector(config)
        self.logger = logging.getLogger(__name__)
        self.min_strength = config.get('position', {}).get('min_strength', 0.6)
    
    def analyze_market_structure(self, market_data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Analyze market structure and return a dictionary containing the analysis results
        """
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
            
            # Calculate basic market structure
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Determine market structure type
            structure_type = self._determine_structure_type(highs, lows, closes)
            
            # Identify SMC patterns
            smc_data = self._identify_smc_patterns(market_data)
            
            # Create market context
            market_context = {
                'current_price': current_price,
                'technical_indicators': technical_indicators,
                'market_structure': structure_type.value,
                'timeframe': timeframe,
                'smc_data': smc_data
            }
            
            return market_context
            
        except Exception as e:
            self.logger.error(f"Error in market structure analysis: {str(e)}")
            return {
                'current_price': current_price if 'current_price' in locals() else 0.0,
                'technical_indicators': technical_indicators if 'technical_indicators' in locals() else {},
                'market_structure': MarketStructureType.NEUTRAL.value,
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

    def _determine_structure_type(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> MarketStructureType:
        """Determine the overall market structure type"""
        # Calculate higher highs and lower lows
        higher_highs = np.all(highs[1:] > highs[:-1])
        lower_lows = np.all(lows[1:] < lows[:-1])
        
        if higher_highs and not lower_lows:
            return MarketStructureType.BULLISH
        elif lower_lows and not higher_highs:
            return MarketStructureType.BEARISH
        else:
            return MarketStructureType.NEUTRAL
    
    def _identify_smc_patterns(self, market_data: pd.DataFrame) -> Dict:
        """Identify Smart Money Concepts patterns"""
        try:
            # Initialize pattern containers
            patterns = {
                'order_blocks': [],
                'liquidity_levels': [],
                'fair_value_gaps': [],
                'supply_zones': [],
                'demand_zones': [],
                'smart_money_traps': []
            }
            
            # Identify order blocks
            patterns['order_blocks'] = self._find_order_blocks(market_data)
            
            # Identify liquidity levels
            patterns['liquidity_levels'] = self._find_liquidity_levels(market_data)
            
            # Identify fair value gaps
            patterns['fair_value_gaps'] = self._find_fair_value_gaps(market_data)
            
            # Identify supply and demand zones
            patterns['supply_zones'] = self._find_supply_zones(market_data)
            patterns['demand_zones'] = self._find_demand_zones(market_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error identifying SMC patterns: {str(e)}")
            return {
                'order_blocks': [],
                'liquidity_levels': [],
                'fair_value_gaps': [],
                'supply_zones': [],
                'demand_zones': [],
                'smart_money_traps': []
            }
    
    def _find_order_blocks(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find order blocks in the market data"""
        order_blocks = []
        try:
            # Implementation for finding order blocks
            # This is a simplified version - you should implement your own logic
            for i in range(2, len(market_data)):
                if market_data['close'].iloc[i] > market_data['high'].iloc[i-1]:
                    order_blocks.append({
                        'type': 'BULLISH',
                        'price': market_data['close'].iloc[i],
                        'volume': market_data['volume'].iloc[i],
                        'strength': 0.8,
                        'timeframe': market_data.index[i]
                    })
                elif market_data['close'].iloc[i] < market_data['low'].iloc[i-1]:
                    order_blocks.append({
                        'type': 'BEARISH',
                        'price': market_data['close'].iloc[i],
                        'volume': market_data['volume'].iloc[i],
                        'strength': 0.8,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding order blocks: {str(e)}")
        return order_blocks
    
    def _find_liquidity_levels(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find liquidity levels in the market data"""
        liquidity_levels = []
        try:
            # Implementation for finding liquidity levels
            # This is a simplified version - you should implement your own logic
            for i in range(1, len(market_data)):
                if market_data['volume'].iloc[i] > market_data['volume'].iloc[i-1] * 1.5:
                    liquidity_levels.append({
                        'type': 'HIGH',
                        'price': market_data['close'].iloc[i],
                        'volume': market_data['volume'].iloc[i],
                        'strength': 0.7,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding liquidity levels: {str(e)}")
        return liquidity_levels
    
    def _find_fair_value_gaps(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find fair value gaps in the market data"""
        fair_value_gaps = []
        try:
            # Implementation for finding fair value gaps
            # This is a simplified version - you should implement your own logic
            for i in range(1, len(market_data)):
                gap = market_data['high'].iloc[i] - market_data['low'].iloc[i-1]
                if gap > market_data['close'].iloc[i-1] * 0.01:  # 1% gap
                    fair_value_gaps.append({
                        'type': 'BULLISH',
                        'upper_price': market_data['high'].iloc[i],
                        'lower_price': market_data['low'].iloc[i-1],
                        'gap_size': gap,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding fair value gaps: {str(e)}")
        return fair_value_gaps
    
    def _find_supply_zones(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find supply zones in the market data"""
        supply_zones = []
        try:
            # Implementation for finding supply zones
            # This is a simplified version - you should implement your own logic
            for i in range(2, len(market_data)):
                if market_data['high'].iloc[i] > market_data['high'].iloc[i-1]:
                    supply_zones.append({
                        'type': 'STRONG',
                        'price': market_data['high'].iloc[i],
                        'strength': 0.8,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding supply zones: {str(e)}")
        return supply_zones
    
    def _find_demand_zones(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find demand zones in the market data"""
        demand_zones = []
        try:
            # Implementation for finding demand zones
            # This is a simplified version - you should implement your own logic
            for i in range(2, len(market_data)):
                if market_data['low'].iloc[i] < market_data['low'].iloc[i-1]:
                    demand_zones.append({
                        'type': 'STRONG',
                        'price': market_data['low'].iloc[i],
                        'strength': 0.8,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding demand zones: {str(e)}")
        return demand_zones
    
    def detect_structure_shift(self, current_structure: Dict, previous_structure: Dict) -> bool:
        """Detect if there's been a shift in market structure"""
        if not previous_structure:
            return False
            
        current_type = current_structure.get('market_structure', MarketStructureType.NEUTRAL.value)
        previous_type = previous_structure.get('market_structure', MarketStructureType.NEUTRAL.value)
        
        return current_type != previous_type
    
    def validate_position_strength(self, position_type: str, structure: Dict) -> float:
        """Validate the strength of a position based on market structure"""
        if not structure:
            return 0.0
            
        structure_type = structure.get('market_structure', MarketStructureType.NEUTRAL.value)
        
        if position_type == 'LONG' and structure_type == MarketStructureType.BULLISH.value:
            return 0.8
        elif position_type == 'SHORT' and structure_type == MarketStructureType.BEARISH.value:
            return 0.8
        elif structure_type == MarketStructureType.NEUTRAL.value:
            return 0.5
        else:
            return 0.2

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