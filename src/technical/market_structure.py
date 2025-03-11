import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..data.models import MarketStructureData, MarketStructure, OrderBlock, FairValueGap, LiquidityLevel
from ..data.database import DatabaseManager
from .smc_patterns import SMCPatternDetector
from datetime import datetime
import logging

class MarketStructureAnalyzer:
    def __init__(self, db_manager: DatabaseManager, config: Dict):
        self.db = db_manager
        self.config = config
        self.smc_detector = SMCPatternDetector(config)
    
    def analyze_market_structure(self,
                               df: pd.DataFrame,
                               timeframe: str,
                               symbol: str,
                               position_id: Optional[int] = None) -> MarketStructureData:
        """
        Analyze market structure for a given symbol and timeframe
        """
        try:
            # Determine overall market structure
            structure_type = self._determine_structure_type(df)
            
            # Detect SMC patterns
            order_blocks = self.smc_detector.detect_institutional_order_blocks(df)
            fair_value_gaps = self.smc_detector.detect_fair_value_gaps(df)
            liquidity_levels = self.smc_detector.detect_liquidity_levels(df)
            
            # Calculate high, low, and volume for the period
            high = df['high'].max()
            low = df['low'].min()
            volume = df['volume'].sum()
            
            # Create market structure data
            structure_data = MarketStructureData(
                symbol=symbol,
                timeframe=timeframe,
                structure_type=structure_type,
                high=high,
                low=low,
                volume=volume,
                created_at=datetime.utcnow()
            )
            
            # Save to database if position_id is provided
            if position_id:
                self.db.save_market_structure(structure_data, position_id)
            
            return structure_data
            
        except Exception as e:
            logging.error(f"Error analyzing market structure: {str(e)}")
            raise
    
    def _determine_structure_type(self, df: pd.DataFrame) -> MarketStructure:
        """Determine the overall market structure type"""
        # Calculate higher highs and lower lows
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        # Count recent higher highs and lower lows
        recent_higher_highs = df['higher_high'].tail(5).sum()
        recent_lower_lows = df['lower_low'].tail(5).sum()
        
        # Calculate trend strength
        trend_strength = abs(recent_higher_highs - recent_lower_lows)
        
        if trend_strength < 2:  # Weak trend
            return MarketStructure.NEUTRAL
        elif recent_higher_highs > recent_lower_lows:
            return MarketStructure.BULLISH
        elif recent_lower_lows > recent_higher_highs:
            return MarketStructure.BEARISH
        else:
            return MarketStructure.SHIFTING
    
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