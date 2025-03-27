import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

class SMCPatternDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_patterns(self, market_data: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze market data for SMC patterns"""
        try:
            patterns = {
                'supply_zones': self._find_supply_zones(market_data),
                'demand_zones': self._find_demand_zones(market_data),
                'smart_money_traps': self._find_smart_money_traps(market_data)
            }
            return patterns
        except Exception as e:
            self.logger.error(f"Error analyzing SMC patterns: {str(e)}")
            return {
                'supply_zones': [],
                'demand_zones': [],
                'smart_money_traps': []
            }
    
    def _find_supply_zones(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find supply zones in the market data"""
        supply_zones = []
        try:
            for i in range(2, len(market_data)):
                if market_data['high'].iloc[i] > market_data['high'].iloc[i-1]:
                    supply_zones.append({
                        'type': 'STRONG',
                        'price': float(market_data['high'].iloc[i]),
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
            for i in range(2, len(market_data)):
                if market_data['low'].iloc[i] < market_data['low'].iloc[i-1]:
                    demand_zones.append({
                        'type': 'STRONG',
                        'price': float(market_data['low'].iloc[i]),
                        'strength': 0.8,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding demand zones: {str(e)}")
        return demand_zones
    
    def _find_smart_money_traps(self, market_data: pd.DataFrame) -> List[Dict]:
        """Find smart money traps in the market data"""
        traps = []
        try:
            for i in range(2, len(market_data)):
                # Bull trap
                if (market_data['high'].iloc[i] > market_data['high'].iloc[i-1] and
                    market_data['close'].iloc[i] < market_data['open'].iloc[i]):
                    traps.append({
                        'type': 'BULL_TRAP',
                        'price': float(market_data['high'].iloc[i]),
                        'strength': 0.7,
                        'timeframe': market_data.index[i]
                    })
                # Bear trap
                elif (market_data['low'].iloc[i] < market_data['low'].iloc[i-1] and
                      market_data['close'].iloc[i] > market_data['open'].iloc[i]):
                    traps.append({
                        'type': 'BEAR_TRAP',
                        'price': float(market_data['low'].iloc[i]),
                        'strength': 0.7,
                        'timeframe': market_data.index[i]
                    })
        except Exception as e:
            self.logger.error(f"Error finding smart money traps: {str(e)}")
        return traps
    
    def _calculate_zone_strength(self, zone: Dict, market_data: pd.DataFrame) -> float:
        """Calculate the strength of a supply or demand zone"""
        try:
            price = zone['price']
            price_range = market_data[abs(market_data['close'] - price) / price < 0.002]
            
            if len(price_range) == 0:
                return 0.0
            
            # Volume strength
            volume_ratio = price_range['volume'].mean() / market_data['volume'].mean()
            volume_strength = min(volume_ratio, 1.0)
            
            # Price action strength
            touches = len(price_range)
            touch_strength = min(touches * 0.1, 0.5)
            
            # Combine strengths
            total_strength = (volume_strength * 0.6 + touch_strength * 0.4)
            return min(total_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating zone strength: {str(e)}")
            return 0.0 