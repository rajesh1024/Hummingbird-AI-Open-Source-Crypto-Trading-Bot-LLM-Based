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
        self.logger = logging.getLogger(__name__)
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
            
            # Set minimum required points based on timeframe
            timeframe = df.index.name if hasattr(df.index, 'name') else 'unknown'
            if timeframe == '1h':
                min_required_points = 24  # 1 day of hourly data
            elif timeframe == '4h':
                min_required_points = 20  # ~3 days of 4h data
            elif timeframe == '1d':
                min_required_points = 20  # 20 days
            elif timeframe == '15m':
                min_required_points = 30  # Reduced from 50 for scalping
            elif timeframe == '5m':
                min_required_points = 25  # Reduced for scalping
            elif timeframe == '1m':
                min_required_points = 20  # Reduced for scalping
            else:
                min_required_points = 26  # Default value

            if len(df) < min_required_points:
                self.logger.warning(f"Insufficient data points for {timeframe}: {len(df)} (minimum required: {min_required_points})")
                return pd.DataFrame()
            
            # Log data points for debugging
            self.logger.debug(f"Processing {len(df)} data points for {timeframe}")
            
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
                # Clean the close price data before MACD calculation
                df['close'] = df['close'].astype(float)  # Ensure close price is float
                df['close'] = df['close'].replace([np.inf, -np.inf, None], np.nan)
                df['close'] = df['close'].ffill().bfill()
                
                # Log the state of close prices before MACD calculation
                self.logger.debug(f"Close prices before MACD: {df['close'].head()}")
                self.logger.debug(f"Close prices shape: {df['close'].shape}")
                self.logger.debug(f"Close prices null count: {df['close'].isna().sum()}")
                
                if df['close'].isna().any():
                    self.logger.warning("Close price contains NaN values after cleaning")
                    return pd.DataFrame()
                
                if len(df) < 26:
                    self.logger.warning(f"Insufficient data points for MACD: {len(df)}")
                    return pd.DataFrame()
                
                # Calculate MACD using pandas_ta with explicit parameters
                try:
                    # Calculate MACD components manually first
                    exp1 = df['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['close'].ewm(span=26, adjust=False).mean()
                    macd_line = exp1 - exp2
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    histogram = macd_line - signal_line
                    
                    # Assign the calculated values
                    df['MACD'] = macd_line
                    df['MACD_Signal'] = signal_line
                    df['MACD_Hist'] = histogram
                    
                    # Clean any remaining invalid values
                    df['MACD'] = df['MACD'].replace([np.inf, -np.inf, None], 0)
                    df['MACD_Signal'] = df['MACD_Signal'].replace([np.inf, -np.inf, None], 0)
                    df['MACD_Hist'] = df['MACD_Hist'].replace([np.inf, -np.inf, None], 0)
                    
                except Exception as e:
                    self.logger.error(f"Error in manual MACD calculation: {str(e)}")
                    return pd.DataFrame()
                
                # Final validation
                if df['MACD'].isna().any() or df['MACD_Signal'].isna().any() or df['MACD_Hist'].isna().any():
                    self.logger.warning("MACD values contain NaN after calculation")
                    return pd.DataFrame()
                
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
            
            # Calculate SMC indicators
            try:
                timeframe = df.index.name if hasattr(df.index, 'name') else 'unknown'
                self.logger.debug(f"Starting SMC analysis for {timeframe}")
                
                # Identify order blocks
                order_blocks = self.identify_order_blocks(df, timeframe=timeframe)
                if order_blocks:
                    self.logger.debug(f"Found {len(order_blocks)} order blocks for {timeframe}")
                
                # Identify supply and demand zones
                supply_zones, demand_zones = self.identify_supply_demand_zones(df, timeframe=timeframe)
                if supply_zones or demand_zones:
                    self.logger.debug(f"Found {len(supply_zones)} supply zones and {len(demand_zones)} demand zones for {timeframe}")
                
                # Identify Fair Value Gaps
                fair_value_gaps = self.identify_fair_value_gaps(df, timeframe=timeframe)
                if fair_value_gaps:
                    self.logger.debug(f"Found {len(fair_value_gaps)} fair value gaps for {timeframe}")
                
                # Identify Liquidity Levels
                liquidity_levels = self.identify_liquidity_levels(df, timeframe=timeframe)
                if liquidity_levels:
                    self.logger.debug(f"Found {len(liquidity_levels)} liquidity levels for {timeframe}")
                
                # Store SMC data in indicators
                self.indicators.update({
                    'Order_Blocks': order_blocks,
                    'Supply_Zones': supply_zones,
                    'Demand_Zones': demand_zones,
                    'Fair_Value_Gaps': fair_value_gaps,
                    'Liquidity_Levels': liquidity_levels
                })
                
                # Log SMC analysis summary
                self.logger.info(f"\nSMC Analysis Summary for {timeframe}:")
                self.logger.info(f"Order Blocks: {len(order_blocks)}")
                self.logger.info(f"Supply Zones: {len(supply_zones)}")
                self.logger.info(f"Demand Zones: {len(demand_zones)}")
                self.logger.info(f"Fair Value Gaps: {len(fair_value_gaps)}")
                self.logger.info(f"Liquidity Levels: {len(liquidity_levels)}")
                
            except Exception as e:
                self.logger.error(f"Error calculating SMC indicators: {str(e)}")
                self.logger.error(f"Traceback: ", exc_info=True)
            
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
            window = 15
        elif timeframe == '5m':
            window = 10  # Reduced from 20 for faster scalping signals
        elif timeframe == '3m':
            window = 15
        elif timeframe == '1h':
            window = 12
        elif timeframe == '4h':
            window = 10
        
        order_blocks = []
        
        # Ensure we have enough data
        if len(df) < window * 1.5:  # Reduced from window * 2
            self.logger.debug(f"Insufficient data for order blocks in {timeframe}: {len(df)} < {window * 1.5}")
            return order_blocks
        
        self.logger.debug(f"Analyzing order blocks for {timeframe} with window size {window}")
        
        for i in range(window, len(df) - window):
            # Check for significant volume (even more lenient threshold for 5m)
            volume_threshold = df['Volume_MA'].iloc[i] * (0.8 if timeframe == '5m' else 1.2)
            
            # Bullish Order Block (ICT principles with more lenient conditions for 5m)
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                df['volume'].iloc[i] > volume_threshold * (0.6 if timeframe == '5m' else 0.8) and    # Even lower volume threshold for 5m
                df['close'].iloc[i] > df['EMA_21'].iloc[i] * 0.998 and # Slightly more lenient EMA condition for 5m
                df['RSI'].iloc[i] < 60):  # Even more lenient RSI condition for scalping
                
                # Validate with very short window for 5m
                validation_window = 2 if timeframe == '5m' else min(window, 5)
                if self._validate_bullish_ob(df, i, validation_window):
                    ob = {
                        'type': 'bullish',
                        'timestamp': df.index[i],
                        'price': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'strength': self._calculate_ob_strength(df, i, 'bullish') * (1.2 if timeframe == '5m' else 1.0),  # Boost 5m strength
                        'timeframe': timeframe
                    }
                    order_blocks.append(ob)
                    self.order_blocks.append(ob)
                    self.logger.info(f"Confirmed bullish order block at {df.index[i]} in {timeframe}")
            
            # Bearish Order Block (ICT principles with more lenient conditions for 5m)
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                df['volume'].iloc[i] > volume_threshold * (0.6 if timeframe == '5m' else 0.8) and    # Even lower volume threshold for 5m
                df['close'].iloc[i] < df['EMA_21'].iloc[i] * 1.002 and # Slightly more lenient EMA condition for 5m
                df['RSI'].iloc[i] > 40):  # Even more lenient RSI condition for scalping
                
                # Validate with very short window for 5m
                validation_window = 2 if timeframe == '5m' else min(window, 5)
                if self._validate_bearish_ob(df, i, validation_window):
                    ob = {
                        'type': 'bearish',
                        'timestamp': df.index[i],
                        'price': df['close'].iloc[i],
                        'volume': df['volume'].iloc[i],
                        'strength': self._calculate_ob_strength(df, i, 'bearish') * (1.2 if timeframe == '5m' else 1.0),  # Boost 5m strength
                        'timeframe': timeframe
                    }
                    order_blocks.append(ob)
                    self.order_blocks.append(ob)
                    self.logger.info(f"Confirmed bearish order block at {df.index[i]} in {timeframe}")
        
        # Sort by strength and return more blocks for 5m
        order_blocks.sort(key=lambda x: x['strength'], reverse=True)
        max_blocks = 5 if timeframe == '5m' else 3  # Return more blocks for 5m timeframe
        return order_blocks[:max_blocks]
    
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
            window = 8  # Reduced for faster scalping signals
        elif timeframe == '3m':
            window = 12
        elif timeframe == '1h':
            window = 8
        elif timeframe == '4h':
            window = 6
        
        supply_zones = []
        demand_zones = []
        
        self.logger.debug(f"Analyzing supply/demand zones for {timeframe} with window size {window}")
        
        # Identify CHoCH (Change of Character) with more lenient conditions for 5m
        for i in range(window, len(df) - window):
            # Bullish CHoCH
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                (timeframe != '5m' or df['high'].iloc[i] > df['high'].iloc[i-2]) and  # Optional condition for 5m
                df['low'].iloc[i] > df['low'].iloc[i-1] and
                df['close'].iloc[i] > df['open'].iloc[i]):
                
                strength = self._calculate_choch_strength(df, i, 'bullish')
                if timeframe == '5m':
                    strength *= 1.2  # Boost strength for 5m timeframe
                
                ch = {
                    'type': 'CHoCH',
                    'direction': 'bullish',
                    'timestamp': df.index[i],
                    'price': df['high'].iloc[i],
                    'strength': strength,
                    'timeframe': timeframe
                }
                self.supply_zones.append(ch)
                supply_zones.append(ch)
                self.logger.debug(f"Found bullish CHoCH at {df.index[i]} in {timeframe}")
            
            # Bearish CHoCH
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                (timeframe != '5m' or df['low'].iloc[i] < df['low'].iloc[i-2]) and  # Optional condition for 5m
                df['high'].iloc[i] < df['high'].iloc[i-1] and
                df['close'].iloc[i] < df['open'].iloc[i]):
                
                strength = self._calculate_choch_strength(df, i, 'bearish')
                if timeframe == '5m':
                    strength *= 1.2  # Boost strength for 5m timeframe
                
                ch = {
                    'type': 'CHoCH',
                    'direction': 'bearish',
                    'timestamp': df.index[i],
                    'price': df['low'].iloc[i],
                    'strength': strength,
                    'timeframe': timeframe
                }
                self.demand_zones.append(ch)
                demand_zones.append(ch)
                self.logger.debug(f"Found bearish CHoCH at {df.index[i]} in {timeframe}")
        
        # Return more zones for 5m timeframe
        supply_zones.sort(key=lambda x: x['strength'], reverse=True)
        demand_zones.sort(key=lambda x: x['strength'], reverse=True)
        max_zones = 3 if timeframe == '5m' else 2
        return supply_zones[:max_zones], demand_zones[:max_zones]
    
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
        # More lenient validation for 5m timeframe
        timeframe = df.index.name if hasattr(df.index, 'name') else 'unknown'
        if timeframe == '5m':
            window = min(window, 3)  # Shorter validation window for 5m
            violation_threshold = 0.001  # Allow small violations
            violations = 0
            max_violations = 1  # Allow one violation
            
            for i in range(index + 1, min(index + window, len(df))):
                if df['high'].iloc[i] > df['high'].iloc[index] * (1 + violation_threshold):
                    violations += 1
                    if violations > max_violations:
                        return False
            return True
        else:
            # Original validation for other timeframes
            for i in range(index + 1, min(index + window, len(df))):
                if df['high'].iloc[i] > df['high'].iloc[index]:
                    return False
            return True
    
    def _validate_demand_zone(self, df: pd.DataFrame, index: int, window: int) -> bool:
        """Validate a demand zone"""
        # More lenient validation for 5m timeframe
        timeframe = df.index.name if hasattr(df.index, 'name') else 'unknown'
        if timeframe == '5m':
            window = min(window, 3)  # Shorter validation window for 5m
            violation_threshold = 0.001  # Allow small violations
            violations = 0
            max_violations = 1  # Allow one violation
            
            for i in range(index + 1, min(index + window, len(df))):
                if df['low'].iloc[i] < df['low'].iloc[index] * (1 - violation_threshold):
                    violations += 1
                    if violations > max_violations:
                        return False
            return True
        else:
            # Original validation for other timeframes
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
    
    def _calculate_choch_strength(self, df: pd.DataFrame, i: int, direction: str) -> float:
        """Calculate Change of Character (CHoCH) strength"""
        try:
            # Volume component
            volume_strength = df['volume'].iloc[i] / df['Volume_MA'].iloc[i]
            
            # Price movement component
            if direction == 'bullish':
                price_movement = (df['high'].iloc[i] - df['low'].iloc[i-1]) / df['low'].iloc[i-1]
            else:  # bearish
                price_movement = (df['high'].iloc[i-1] - df['low'].iloc[i]) / df['high'].iloc[i-1]
            
            # RSI component
            rsi_strength = 0
            if direction == 'bullish' and df['RSI'].iloc[i] < 30:
                rsi_strength = (30 - df['RSI'].iloc[i]) / 30
            elif direction == 'bearish' and df['RSI'].iloc[i] > 70:
                rsi_strength = (df['RSI'].iloc[i] - 70) / 30
            
            # Combine components
            strength = (volume_strength * 0.4 + price_movement * 0.4 + rsi_strength * 0.2)
            return min(max(strength, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating CHoCH strength: {str(e)}")
            return 0.0

    def _calculate_pdl_strength(self, df: pd.DataFrame, i: int) -> float:
        """Calculate Previous Day Low (PDL) strength"""
        try:
            # Volume component
            volume_strength = df['volume'].iloc[i] / df['Volume_MA'].iloc[i]
            
            # Price bounce component
            price_bounce = (df['close'].iloc[i] - df['low'].iloc[i]) / df['low'].iloc[i]
            
            # Previous candles confirmation
            prev_candles_strength = 0
            if i >= 3:
                lower_lows = sum(1 for j in range(i-3, i) if df['low'].iloc[j] > df['low'].iloc[i])
                prev_candles_strength = lower_lows / 3
            
            # Combine components
            strength = (volume_strength * 0.4 + price_bounce * 0.3 + prev_candles_strength * 0.3)
            return min(max(strength, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating PDL strength: {str(e)}")
            return 0.0

    def _calculate_pdh_strength(self, df: pd.DataFrame, i: int) -> float:
        """Calculate Previous Day High (PDH) strength"""
        try:
            # Volume component
            volume_strength = df['volume'].iloc[i] / df['Volume_MA'].iloc[i]
            
            # Price rejection component
            price_rejection = (df['high'].iloc[i] - df['close'].iloc[i]) / df['high'].iloc[i]
            
            # Previous candles confirmation
            prev_candles_strength = 0
            if i >= 3:
                higher_highs = sum(1 for j in range(i-3, i) if df['high'].iloc[j] < df['high'].iloc[i])
                prev_candles_strength = higher_highs / 3
            
            # Combine components
            strength = (volume_strength * 0.4 + price_rejection * 0.3 + prev_candles_strength * 0.3)
            return min(max(strength, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating PDH strength: {str(e)}")
            return 0.0

    def identify_fair_value_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Identify Fair Value Gaps (FVG) in the market structure"""
        fvgs = []
        try:
            # More sensitive gap size threshold for 5m timeframe
            gap_threshold = 0.0001 if timeframe == '5m' else 0.0002  # Even smaller threshold for 5m
            
            for i in range(1, len(df)-1):
                # Bullish FVG
                if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                    gap_size = df['low'].iloc[i+1] - df['high'].iloc[i-1]
                    if gap_size > df['close'].iloc[i] * gap_threshold:
                        strength = gap_size / (df['close'].iloc[i] * (0.0005 if timeframe == '5m' else 0.001))  # More sensitive strength calc for 5m
                        if timeframe == '5m':
                            strength *= 1.5  # Boost strength even more for 5m timeframe
                        
                        fvg = {
                            'type': 'bullish',
                            'timestamp': df.index[i],
                            'upper_price': df['low'].iloc[i+1],
                            'lower_price': df['high'].iloc[i-1],
                            'gap_size': gap_size,
                            'strength': min(strength, 1.0),  # Normalize strength
                            'timeframe': timeframe
                        }
                        fvgs.append(fvg)
                        self.logger.debug(f"Found bullish FVG at {df.index[i]} in {timeframe}")
                
                # Bearish FVG
                if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                    gap_size = df['low'].iloc[i-1] - df['high'].iloc[i+1]
                    if gap_size > df['close'].iloc[i] * gap_threshold:
                        strength = gap_size / (df['close'].iloc[i] * (0.0005 if timeframe == '5m' else 0.001))  # More sensitive strength calc for 5m
                        if timeframe == '5m':
                            strength *= 1.5  # Boost strength even more for 5m timeframe
                        
                        fvg = {
                            'type': 'bearish',
                            'timestamp': df.index[i],
                            'upper_price': df['low'].iloc[i-1],
                            'lower_price': df['high'].iloc[i+1],
                            'gap_size': gap_size,
                            'strength': min(strength, 1.0),  # Normalize strength
                            'timeframe': timeframe
                        }
                        fvgs.append(fvg)
                        self.logger.debug(f"Found bearish FVG at {df.index[i]} in {timeframe}")
            
            # Sort by strength and return more gaps for 5m
            fvgs.sort(key=lambda x: x['strength'], reverse=True)
            max_gaps = 5 if timeframe == '5m' else 3
            return fvgs[:max_gaps]
            
        except Exception as e:
            self.logger.error(f"Error identifying FVGs: {str(e)}")
            return []

    def identify_liquidity_levels(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Identify liquidity levels in the market structure"""
        liquidity_levels = []
        try:
            # Calculate swing highs and lows
            for i in range(2, len(df)-2):
                # Swing high
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    
                    # Check volume confirmation
                    if df['volume'].iloc[i] > df['Volume_MA'].iloc[i] * 1.2:
                        level = {
                            'type': 'resistance',
                            'timestamp': df.index[i],
                            'price': df['high'].iloc[i],
                            'volume': df['volume'].iloc[i],
                            'strength': self._calculate_liquidity_strength(df, i, 'high'),
                            'timeframe': timeframe
                        }
                        liquidity_levels.append(level)
                        self.logger.debug(f"Found resistance liquidity level at {df.index[i]} in {timeframe}")
                
                # Swing low
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    
                    # Check volume confirmation
                    if df['volume'].iloc[i] > df['Volume_MA'].iloc[i] * 1.2:
                        level = {
                            'type': 'support',
                            'timestamp': df.index[i],
                            'price': df['low'].iloc[i],
                            'volume': df['volume'].iloc[i],
                            'strength': self._calculate_liquidity_strength(df, i, 'low'),
                            'timeframe': timeframe
                        }
                        liquidity_levels.append(level)
                        self.logger.debug(f"Found support liquidity level at {df.index[i]} in {timeframe}")
            
            return liquidity_levels
            
        except Exception as e:
            self.logger.error(f"Error identifying liquidity levels: {str(e)}")
            return []

    def _calculate_liquidity_strength(self, df: pd.DataFrame, i: int, level_type: str) -> float:
        """Calculate the strength of a liquidity level"""
        try:
            # Volume component
            volume_strength = df['volume'].iloc[i] / df['Volume_MA'].iloc[i]
            
            # Price rejection component
            if level_type == 'high':
                rejection = (df['high'].iloc[i] - df['close'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i])
            else:  # low
                rejection = (df['close'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i])
            
            # Previous touches component
            touches = 0
            look_back = 10
            start_idx = max(0, i-look_back)
            price_level = df['high'].iloc[i] if level_type == 'high' else df['low'].iloc[i]
            price_threshold = price_level * 0.0005  # 0.05% threshold
            
            for j in range(start_idx, i):
                if level_type == 'high':
                    if abs(df['high'].iloc[j] - price_level) <= price_threshold:
                        touches += 1
                else:
                    if abs(df['low'].iloc[j] - price_level) <= price_threshold:
                        touches += 1
            
            touches_strength = min(touches / 3, 1)  # Normalize touches (max 3 touches for full strength)
            
            # Combine components
            strength = (volume_strength * 0.4 + rejection * 0.3 + touches_strength * 0.3)
            return min(max(strength, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity level strength: {str(e)}")
            return 0.0 