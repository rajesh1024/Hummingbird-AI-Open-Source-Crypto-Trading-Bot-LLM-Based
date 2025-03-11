from ctransformers import AutoModelForCausalLM
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import json
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from datetime import datetime
import os
from dotenv import load_dotenv

from .gemini_model import GeminiModel

console = Console()
logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(
        self,
        model_config: Dict,
        model_name: str = "mistral",
        technical_analyzer: Optional['TechnicalAnalysis'] = None
    ):
        self.model_config = model_config
        self.model_name = model_name
        self.technical_analyzer = technical_analyzer
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
        self.confidence_threshold = model_config.get('confidence_threshold', 0.3)  # Get from config
        
    def _load_model(self):
        """
        Load the appropriate model based on configuration
        """
        try:
            model_settings = self.model_config['models'][self.model_name]
            
            if model_settings['type'] == 'local':
                # Load local Mistral model
                with console.status("[bold green]Loading Mistral model...") as status:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_settings['model_path'],
                        model_type="mistral",
                        context_length=model_settings['context_length'],
                        max_new_tokens=model_settings['max_tokens'],
                        temperature=model_settings['temperature'],
                        top_p=model_settings['top_p']
                    )
                console.print("[bold green]✓ Mistral model loaded successfully!")
                return model
                
            elif model_settings['type'] == 'api':
                # Load Gemini API model
                load_dotenv()  # Load environment variables
                api_key = os.getenv('GEMINI_API_KEY')
                console.print("[bold cyan]Debug: Checking Gemini API key...[/bold cyan]")
                if not api_key:
                    console.print("[bold red]Error: GEMINI_API_KEY not found in environment variables[/bold red]")
                    raise ValueError("Gemini API key not found in .env file")
                else:
                    console.print("[bold green]✓ Found Gemini API key[/bold green]")
                
                try:
                    console.print("[bold cyan]Debug: Initializing Gemini model...[/bold cyan]")
                    model = GeminiModel(
                        api_key=api_key,
                        model_name=model_settings['name'],
                        max_tokens=model_settings['max_tokens'],
                        temperature=model_settings['temperature'],
                        system_prompt=model_settings['system_prompt']
                    )
                    console.print("[bold green]✓ Successfully initialized Gemini model[/bold green]")
                    return model
                except Exception as e:
                    console.print(f"[bold red]Error initializing Gemini model: {str(e)}[/bold red]")
                    raise
            
            else:
                raise ValueError(f"Unsupported model type: {model_settings['type']}")
                
        except Exception as e:
            console.print(f"[bold red]Error loading model: {str(e)}")
            raise
    
    def prepare_market_context(
        self,
        market_data: Dict[str, pd.DataFrame],
        technical_indicators: Dict[str, float],
        order_blocks: List[Dict],
        fair_value_gaps: List[Dict],
        liquidity_levels: List[Dict]
    ) -> str:
        """Prepare market context for LLM analysis"""
        context = []
        
        # Add recent price action first with more detail
        context.append("Recent Price Action (Last 5 Candles):")
        for timeframe, df in market_data.items():
            last_5_candles = df.tail(5)
            context.append(f"\n{timeframe} Timeframe:")
            for i in range(len(last_5_candles)-1, -1, -1):
                candle = last_5_candles.iloc[i]
                body_size = abs(candle['close'] - candle['open'])
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                candle_type = "Bullish" if candle['close'] > candle['open'] else "Bearish"
                context.append(
                    f"- {candle.name}: "
                    f"Type: {candle_type}, "
                    f"Open: {candle['open']:.2f}, "
                    f"High: {candle['high']:.2f}, "
                    f"Low: {candle['low']:.2f}, "
                    f"Close: {candle['close']:.2f}, "
                    f"Volume: {candle['volume']:.2f}, "
                    f"Body: {body_size:.2f}, "
                    f"Upper Wick: {upper_wick:.2f}, "
                    f"Lower Wick: {lower_wick:.2f}"
                )
        
        # Add technical indicators with trend analysis
        context.append("\nTechnical Indicators and Trend Analysis:")
        for name, value in technical_indicators.items():
            if isinstance(value, (int, float)):
                context.append(f"{name}: {float(value):.2f}")
                # Add trend analysis for key indicators
                if name == 'RSI':
                    if value > 70:
                        context.append("RSI indicates overbought conditions - Potential SELL signal")
                    elif value < 30:
                        context.append("RSI indicates oversold conditions - Potential BUY signal")
                    elif value > 60:
                        context.append("RSI showing bullish momentum")
                    elif value < 40:
                        context.append("RSI showing bearish momentum")
                elif name == 'MACD_Hist':
                    if value > 0:
                        context.append("MACD histogram indicates bullish momentum - Potential BUY signal")
                    else:
                        context.append("MACD histogram indicates bearish momentum - Potential SELL signal")
                elif name in ['EMA_8', 'EMA_21']:
                    if name == 'EMA_8' and value > technical_indicators.get('EMA_21', 0):
                        context.append("Price above both EMAs - Bullish trend")
                    elif name == 'EMA_8' and value < technical_indicators.get('EMA_21', 0):
                        context.append("Price below both EMAs - Bearish trend")
        
        # Add SMC analysis with emphasis on recent patterns
        context.append("\nSmart Money Concepts (SMC) Analysis:")
        
        # Add Order Blocks (most recent first)
        if order_blocks:
            context.append("\nRecent Order Blocks (Sorted by Strength):")
            for block in sorted(order_blocks, key=lambda x: x.strength, reverse=True)[:3]:
                context.append(
                    f"- Type: {block.block_type}, "
                    f"Price: {block.price:.2f}, "
                    f"Volume: {block.volume:.2f}, "
                    f"Strength: {block.strength:.2f}"
                )
                # Add distance from current price
                current_price = market_data[list(market_data.keys())[0]]['close'].iloc[-1]
                distance = abs(block.price - current_price) / current_price * 100
                context.append(f"  Distance from current price: {distance:.2f}%")
                if distance < 0.5:
                    context.append(f"  ⚠️ Strong signal: Price near order block!")
        
        # Add Fair Value Gaps (most recent first)
        if fair_value_gaps:
            context.append("\nRecent Fair Value Gaps:")
            for gap in fair_value_gaps:
                context.append(
                    f"- Type: {gap.gap_type}, "
                    f"Upper Price: {gap.upper_price:.2f}, "
                    f"Lower Price: {gap.lower_price:.2f}, "
                    f"Volume: {gap.volume:.2f}"
                )
                # Add gap size percentage
                gap_size = (gap.upper_price - gap.lower_price) / gap.lower_price * 100
                context.append(f"  Gap Size: {gap_size:.2f}%")
                if gap_size > 0.1:
                    context.append(f"  ⚠️ Strong signal: Large FVG detected!")
        
        # Add Liquidity Levels (most recent first)
        if liquidity_levels:
            context.append("\nRecent Liquidity Levels (Sorted by Strength):")
            for level in sorted(liquidity_levels, key=lambda x: x.strength, reverse=True)[:3]:
                context.append(
                    f"- Type: {level.level_type}, "
                    f"Price: {level.price:.2f}, "
                    f"Volume: {level.volume:.2f}, "
                    f"Strength: {level.strength:.2f}"
                )
                # Add distance from current price
                current_price = market_data[list(market_data.keys())[0]]['close'].iloc[-1]
                distance = abs(level.price - current_price) / current_price * 100
                context.append(f"  Distance from current price: {distance:.2f}%")
                if distance < 0.3:
                    context.append(f"  ⚠️ Strong signal: Price near liquidity level!")
        
        # Add market structure data with trend analysis
        context.append("\nMarket Structure Analysis:")
        for timeframe, df in market_data.items():
            context.append(f"\n{timeframe} Timeframe:")
            recent_high = df['high'].tail(5).max()
            recent_low = df['low'].tail(5).min()
            current_price = df['close'].iloc[-1]
            trend = self._determine_trend(df)
            context.append(f"Current Trend: {trend}")
            context.append(f"Recent High: {recent_high:.2f}")
            context.append(f"Recent Low: {recent_low:.2f}")
            context.append(f"Current Price: {current_price:.2f}")
            
            # Add price position relative to recent range
            range_size = recent_high - recent_low
            price_position = (current_price - recent_low) / range_size * 100
            context.append(f"Price Position in Range: {price_position:.2f}%")
            
            # Add trend strength indicators
            if trend in ["STRONG UPTREND", "STRONG DOWNTREND"]:
                context.append("⚠️ Strong trend detected!")
            elif trend in ["MODERATE UPTREND", "MODERATE DOWNTREND"]:
                context.append("Moderate trend strength")
            elif trend == "RANGE-BOUND":
                context.append("⚠️ Range-bound market - Look for breakout opportunities")
        
        # Add risk-reward analysis with specific levels
        context.append("\nRisk-Reward Analysis:")
        for timeframe, df in market_data.items():
            recent_high = df['high'].tail(5).max()
            recent_low = df['low'].tail(5).min()
            current_price = df['close'].iloc[-1]
            risk_reward_ratio = (recent_high - current_price) / (current_price - recent_low)
            context.append(f"\n{timeframe} Timeframe:")
            context.append(f"Risk-Reward Ratio: {risk_reward_ratio:.2f}")
            context.append(f"Potential Risk: {(current_price - recent_low):.2f}")
            context.append(f"Potential Reward: {(recent_high - current_price):.2f}")
            if risk_reward_ratio >= 2:
                context.append("⚠️ Good risk-reward opportunity!")
        
        return "\n".join(context)
    
    def generate_signal(
        self,
        market_context: str,
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """Generate a trading signal based on market data"""
        try:
            # Skip confidence threshold check
            threshold = 0.0  # Always allow signals through
            
            # Add debug logging
            console.print("[bold cyan]Debug: Preparing to call Gemini model[/bold cyan]")
            console.print(f"[bold cyan]Debug: Market Context Length: {len(market_context)}[/bold cyan]")
            console.print("[bold cyan]Debug: First 500 chars of context:[/bold cyan]\n" + market_context[:500])
            
            # Get response from model using the prepared market context
            console.print("[bold cyan]Debug: Calling Gemini model generate_response[/bold cyan]")
            response = self.model.generate_response(market_context)
            console.print(f"[bold cyan]Debug: Got response from Gemini: {response}[/bold cyan]")
            
            # Validate response
            if not response or not isinstance(response, dict):
                self.logger.error("Invalid response from model")
                return self._get_default_signal()
            
            # Validate required fields
            required_fields = ["signal", "confidence", "reasoning", "entry_price", "stop_loss", "take_profit"]
            if not all(field in response for field in required_fields):
                self.logger.error(f"Missing required fields in response: {response}")
                return self._get_default_signal()
            
            # No need to validate confidence - let it pass through
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            console.print(f"[bold red]Debug: Exception in generate_signal: {str(e)}[/bold red]")
            return self._get_default_signal()
    
    def _get_default_signal(self) -> Dict:
        """Return a default signal in case of errors"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,  # Changed from 0.5 to 0.0
            "reasoning": "Error generating signal",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0
        }
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured data
        """
        try:
            # If response is already a dictionary, convert it to the expected format
            if isinstance(response, dict):
                # Convert take_profit to take_profits if needed
                if 'take_profit' in response and 'take_profits' not in response:
                    response['take_profits'] = [response['take_profit']]
                    del response['take_profit']
                
                # Standardize signal types
                if 'signal' in response:
                    response['signal'] = self._standardize_signal(response['signal'])
                return response
                
            # If response is a string, extract JSON
            if isinstance(response, str):
                # Extract JSON from response
                json_str = response.split("{")[1].split("}")[0]
                json_str = "{" + json_str + "}"
                parsed = json.loads(json_str)
                
                # Convert take_profit to take_profits if needed
                if 'take_profit' in parsed and 'take_profits' not in parsed:
                    parsed['take_profits'] = [parsed['take_profit']]
                    del parsed['take_profit']
                
                # Standardize signal types
                if 'signal' in parsed:
                    parsed['signal'] = self._standardize_signal(parsed['signal'])
                    
                return parsed
                
            raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            console.print(f"[bold red]Error parsing LLM response: {str(e)}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profits": [0.0],
                "reasoning": "Error parsing response"
            }
    
    def _standardize_signal(self, signal: str) -> str:
        """
        Standardize signal types to LONG/SHORT/NEUTRAL
        """
        signal = signal.upper()
        if signal in ['BUY', 'LONG']:
            return 'LONG'
        elif signal in ['SELL', 'SHORT']:
            return 'SHORT'
        elif signal in ['HOLD', 'NEUTRAL']:
            return 'NEUTRAL'
        else:
            return 'NEUTRAL'  # Default to NEUTRAL for unknown signals
    
    def _display_signal(self, signal_data: Dict[str, Any]) -> None:
        """Display the trading signal in a formatted table"""
        table = Table(title="Trading Signal Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add signal information
        table.add_row("Signal", signal_data["signal"])
        table.add_row("Confidence", f"{signal_data['confidence']:.2%}")
        
        # Add price levels
        table.add_row("Entry Price", f"${signal_data['entry_price']:.2f}")
        table.add_row("Stop Loss", f"${signal_data['stop_loss']:.2f}")
        table.add_row("Take Profit", f"${signal_data['take_profit']:.2f}")
        
        # Calculate risk:reward ratio
        if signal_data["signal"] == "BUY":
            risk = signal_data["entry_price"] - signal_data["stop_loss"]
            reward = signal_data["take_profit"] - signal_data["entry_price"]
            rr_ratio = reward / risk if risk > 0 else 0
        elif signal_data["signal"] == "SELL":
            risk = signal_data["stop_loss"] - signal_data["entry_price"]
            reward = signal_data["entry_price"] - signal_data["take_profit"]
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0
        
        table.add_row("Risk:Reward Ratio", f"{rr_ratio:.2f}")
        
        # Display the table
        console.print(table)
        
        # Display reasoning in a panel
        console.print(Panel(
            signal_data["reasoning"],
            title="Signal Reasoning",
            border_style="blue"
        ))
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """
        Determine the current market trend based on price action and EMAs
        """
        try:
            # Get the last 20 candles
            recent_data = df.tail(20)
            
            # Calculate EMAs if not already present
            if 'EMA_8' not in df.columns:
                df['EMA_8'] = df['close'].ewm(span=8, adjust=False).mean()
            if 'EMA_21' not in df.columns:
                df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Get current price and EMAs
            current_price = df['close'].iloc[-1]
            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            
            # Get recent highs and lows
            recent_highs = recent_data['high'].max()
            recent_lows = recent_data['low'].min()
            
            # Determine trend based on multiple factors
            if current_price > ema_8 > ema_21:
                # Strong uptrend
                return "STRONG UPTREND"
            elif current_price > ema_8 and ema_8 > ema_21:
                # Moderate uptrend
                return "MODERATE UPTREND"
            elif current_price > ema_8:
                # Weak uptrend
                return "WEAK UPTREND"
            elif current_price < ema_8 < ema_21:
                # Strong downtrend
                return "STRONG DOWNTREND"
            elif current_price < ema_8 and ema_8 < ema_21:
                # Moderate downtrend
                return "MODERATE DOWNTREND"
            elif current_price < ema_8:
                # Weak downtrend
                return "WEAK DOWNTREND"
            else:
                # Check for range-bound market
                price_range = recent_highs - recent_lows
                if price_range < (recent_highs * 0.02):  # Less than 2% range
                    return "RANGE-BOUND"
                else:
                    return "NEUTRAL"
                    
        except Exception as e:
            self.logger.error(f"Error determining trend: {str(e)}")
            return "UNKNOWN" 