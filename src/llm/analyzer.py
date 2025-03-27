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
        technical_analysis: Optional['TechnicalAnalysis'] = None
    ):
        self.model_config = model_config
        self.model_name = model_name
        self.technical_analysis = technical_analysis
        self.position_manager = None  # Will be set after initialization
        self.db = None  # Will be set after initialization
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
        self.confidence_threshold = model_config.get('confidence_threshold', 0.3)  # Get from config
        self.default_settings = {
            'min_risk_reward_ratio': 1.0,  # Changed from 2.0 to match config
            'min_position_strength': 0.7,
            'max_positions': 3,
            'risk_per_trade': 0.01
        }
        
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
    
    def prepare_market_context(self, market_data: pd.DataFrame, technical_indicators: Dict, market_structure: Dict) -> Dict:
        """
        Prepare market context for LLM analysis
        """
        try:
            if market_data is None or technical_indicators is None or market_structure is None:
                raise ValueError("Invalid input data")

            # Create enhanced market context
            context = {
                'current_price': market_structure.get('current_price', market_data['close'].iloc[-1]),
                'technical_indicators': technical_indicators,
                'market_structure': market_structure.get('market_structure', 'NEUTRAL'),
                'timeframe': market_structure.get('timeframe', ''),
                'smc_data': {
                    'order_blocks': market_structure.get('order_blocks', []),
                    'fair_value_gaps': market_structure.get('fair_value_gaps', []),
                    'liquidity_levels': market_structure.get('liquidity_levels', [])
                },
                'trend_analysis': market_structure.get('trend_analysis', {}),
                'market_context': market_structure.get('market_context', {}),
                'volatility': market_structure.get('volatility', 0),
                'active_positions': self._get_active_positions() if self.position_manager else []
            }

            # Add key level summary
            context['key_levels'] = self._summarize_key_levels(context['smc_data'])
            
            # Add trading session context
            context['session_info'] = {
                'type': market_structure.get('market_context', {}).get('session_type', 'UNKNOWN'),
                'volume_profile': market_structure.get('market_context', {}).get('volume_analysis', {}),
                'price_action': market_structure.get('market_context', {}).get('price_action', {})
            }

            return context

        except Exception as e:
            self.logger.error(f"Error preparing market context: {str(e)}")
            return {
                'current_price': 0,
                'technical_indicators': {},
                'market_structure': 'NEUTRAL',
                'smc_data': {'order_blocks': [], 'fair_value_gaps': [], 'liquidity_levels': []},
                'active_positions': []
            }

    def _summarize_key_levels(self, smc_data: Dict) -> Dict:
        """Summarize key price levels for quick reference"""
        key_levels = {
            'support': [],
            'resistance': [],
            'high_probability_zones': []
        }

        # Process liquidity levels
        for level in smc_data.get('liquidity_levels', []):
            if level['level_type'] == 'SUPPORT':
                key_levels['support'].append({
                    'price': level['price'],
                    'strength': level['strength'],
                    'confluence': len(level.get('confluence_factors', [])),
                    'test_count': level.get('test_count', 0)
                })
            else:
                key_levels['resistance'].append({
                    'price': level['price'],
                    'strength': level['strength'],
                    'confluence': len(level.get('confluence_factors', [])),
                    'test_count': level.get('test_count', 0)
                })

        # Identify high probability zones (areas with multiple confluences)
        for level_type in ['support', 'resistance']:
            levels = key_levels[level_type]
            for i, level in enumerate(levels):
                nearby_levels = [l for l in levels if abs(l['price'] - level['price']) / level['price'] < 0.001]
                if len(nearby_levels) > 1:
                    zone = {
                        'price': sum(l['price'] for l in nearby_levels) / len(nearby_levels),
                        'strength': max(l['strength'] for l in nearby_levels),
                        'type': level_type.upper(),
                        'confluence_count': sum(l['confluence'] for l in nearby_levels)
                    }
                    if zone not in key_levels['high_probability_zones']:
                        key_levels['high_probability_zones'].append(zone)

        return key_levels

    def set_position_manager(self, position_manager, db):
        """Set the position manager and database connection"""
        self.position_manager = position_manager
        self.db = db

    def generate_signal(self, market_context: dict, confidence_threshold: float = None) -> dict:
        """Generate trading signal and manage positions based on model's recommendations"""
        try:
            console.print("\n[bold cyan]Debug: Starting signal generation...[/bold cyan]")
            # console.print("\n[bold cyan]Debug: Market context:[/bold cyan]")
            # console.print(market_context)
            
            # Get current active positions if position manager is initialized
            active_positions = []
            if self.position_manager is not None:
                try:
                    active_positions = self.position_manager.get_active_positions()
                    console.print("\n[bold cyan]Debug: Found active positions:[/bold cyan]")
                    for pos in active_positions:
                        self._display_position(pos, market_context.get('current_price', 0))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not fetch active positions: {str(e)}[/yellow]")
            
            # Add positions to market context if they exist
            if active_positions:
                market_context['active_positions'] = active_positions
                console.print("\n[bold cyan]Debug: Added active positions to market context[/bold cyan]")
            else:
                market_context['active_positions'] = []
                console.print("\n[bold cyan]Debug: No active positions found[/bold cyan]")
            
            # Generate signal from model with position context
            console.print("\n[bold cyan]Debug: Generating signal from model...[/bold cyan]")
            signal = self._generate_model_signal(market_context)
            
            # Debug: Print signal details
            console.print("\n[bold cyan]Debug: Generated Signal Details:[/bold cyan]")
            if signal:
                # Create a table for better visualization
                table = Table(title="Signal Analysis")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Current Price", str(market_context["current_price"]))
                table.add_row("Signal Type", str(signal.get('signal', 'N/A')))
                table.add_row("Confidence", f"{signal.get('confidence', 0):.2f}")
                table.add_row("Entry Price", f"{signal.get('entry_price', 0):.2f}")
                table.add_row("Stop Loss", f"{signal.get('stop_loss', 0):.2f}")
                table.add_row("Take Profit", f"{signal.get('take_profit', 0):.2f}")
                table.add_row("Reasoning", str(signal.get('reasoning', 'N/A')))
                table.add_row("Active Positions", str(len(active_positions)))
                
                if 'position_management' in signal:
                    pm = signal['position_management']
                    table.add_row("Position Action", str(pm.get('action', 'N/A')))
                    table.add_row("Risk:Reward", f"{pm.get('risk_reward_ratio', 0):.2f}")
                
                console.print(table)
                
                # Handle position management based on signal
                current_price = market_context.get('current_price', 0)
                
                if active_positions:
                    # If we have active positions, manage them based on the signal
                    self._handle_position_management(active_positions, signal, current_price)
                else:
                    # If no active positions, check if we should create a new one
                    # Use provided confidence threshold or fall back to default
                    threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
                    
                    console.print("\n[bold cyan]Debug: Position Creation Check:[/bold cyan]")
                    console.print(f"Signal Confidence: {signal.get('confidence', 0):.2f}")
                    console.print(f"Confidence Threshold: {threshold}")
                    console.print(f"Position Manager Initialized: {self.position_manager is not None}")
                    
                    if signal.get('confidence', 0) >= threshold:
                        console.print(f"\n[bold cyan]Debug: Signal confidence {signal.get('confidence', 0):.2f} meets threshold {threshold}[/bold cyan]")
                        # Create new position if confidence is high enough
                        if self.position_manager is not None:
                            console.print("\n[bold cyan]Debug: Attempting to create new position...[/bold cyan]")
                            new_position = self._create_new_position(signal, market_context)
                            if new_position:
                                console.print(f"\n[bold green]Created new position based on {signal['signal']} signal[/bold green]")
                                self._display_position(new_position, current_price)
                            else:
                                console.print("\n[yellow]Warning: Failed to create new position[/yellow]")
                        else:
                            console.print("\n[yellow]Warning: Position manager not initialized, skipping position creation[/yellow]")
                    else:
                        console.print(f"\n[yellow]Warning: Signal confidence {signal.get('confidence', 0):.2f} below threshold {threshold}, skipping position creation[/yellow]")
            else:
                console.print("[bold red]No signal generated[/bold red]")
                return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            console.print(f"[bold red]Error generating signal: {str(e)}[/bold red]")
            import traceback
            console.print(f"[bold red]Traceback:\n{traceback.format_exc()}[/bold red]")
            return None

    def _handle_position_management(self, active_positions, signal, current_price):
        """Handle position management based on model's recommendations"""
        try:
            position_management = signal.get('position_management', {})
            action = position_management.get('action', 'MAINTAIN')
            
            for position in active_positions:
                # Update current price and PnL
                position.current_price = current_price
                if position.position_type == "LONG":
                    pnl = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    pnl = (position.entry_price - current_price) / position.entry_price * 100
                
                # Log position status
                self.logger.info(f"Managing position {position.id}:")
                self.logger.info(f"- Type: {position.position_type}")
                self.logger.info(f"- Entry: ${position.entry_price:.2f}")
                self.logger.info(f"- Current: ${current_price:.2f}")
                self.logger.info(f"- PnL: {pnl:.2f}%")
                self.logger.info(f"- Action: {action}")
                
                if action == 'CLOSE':
                    # Close position if recommended
                    self.logger.info(f"Closing position {position.id} based on model recommendation")
                    self.position_manager.close_position(position.id)
                    
                    # # Create exit signal - commented out as not needed
                    # self.position_manager.create_exit_signal(
                    #     position_id=position.id,
                    #     signal_type='model_recommendation',
                    #     price=current_price,
                    #     reason=signal.get('reasoning', 'Model recommended position closure'),
                    #     confidence=signal.get('confidence', 0.0),
                    #     model_analysis=signal.get('reasoning', '')
                    # )
                else:
                    # Update position with model's analysis
                    self.position_manager.update_position_with_analysis(
                        position_id=position.id,
                        current_price=current_price,
                        signal=signal,
                        market_structure=signal.get('market_structure', None)
                    )
                    
                    # Update position strength
                    position.position_strength = self.position_manager._calculate_position_strength(
                        position,
                        signal.get('market_structure', None)
                    )
                
                # Commit changes
                self.db.commit()
                
        except Exception as e:
            self.logger.error(f"Error in position management: {str(e)}")
            self.db.rollback()
            raise

    def _create_new_position(self, signal, market_context):
        """Create a new position based on signal"""
        try:
            # Check if there are any active positions first
            active_positions = self.position_manager.get_active_positions()
            if active_positions:
                self.logger.info(f"Skipping new position creation - {len(active_positions)} active positions exist")
                return None

            if signal['signal'] != "HOLD":
                position_type = "LONG" if signal['signal'] == "BUY" else "SHORT"
                
                # Ensure required fields are present
                if not all(key in market_context for key in ['symbol', 'timeframe']):
                    console.print("[red]Error: Missing required fields in market_context[/red]")
                    console.print(f"Required: symbol, timeframe")
                    console.print(f"Present: {list(market_context.keys())}")
                    return None

                # Add default configuration if not present
                if 'config' not in market_context:
                    console.print("[yellow]Warning: No config found in market_context, using defaults[/yellow]")
                    market_context['config'] = {
                        'position': {
                            'risk_per_trade': 100,  # Default $100 risk per trade
                            'min_risk_reward_ratio': 2.0,  # Default minimum risk:reward ratio
                        }
                    }
                
                # Validate required signal fields
                required_signal_fields = ['entry_price', 'stop_loss', 'take_profit', 'confidence']
                if not all(key in signal for key in required_signal_fields):
                    console.print("[red]Error: Missing required fields in signal[/red]")
                    console.print(f"Required: {required_signal_fields}")
                    console.print(f"Present: {list(signal.keys())}")
                    return None
                
                # Get confidence threshold from config
                confidence_threshold = self.model_config.get('confidence_threshold', 0.3)
                
                # Validate signal confidence
                if signal.get('confidence', 0) < confidence_threshold:
                    console.print("[red]Position creation failed: Confidence below threshold[/red]")
                    self.logger.warning(f"Skipping position creation: Confidence {signal.get('confidence', 0):.2f} below threshold {confidence_threshold}")
                    return None
                
                # Calculate risk-reward ratio
                risk_reward_ratio = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss'])
                min_risk_reward = market_context['config'].get('risk_management', {}).get('scalping', {}).get('min_risk_reward', 1.0)
                
                if risk_reward_ratio >= min_risk_reward:
                    # Calculate position size based on risk management
                    risk_per_trade = market_context['config']['position'].get('risk_per_trade', 100)
                    stop_loss_distance = abs(signal['entry_price'] - signal['stop_loss'])
                    position_size = risk_per_trade / stop_loss_distance if stop_loss_distance > 0 else 0
                    
                    if position_size <= 0:
                        console.print("[red]Error: Invalid position size calculated[/red]")
                        return None
                    
                    try:
                        if self.position_manager is None:
                            console.print("[red]Error: Position manager is not initialized[/red]")
                            return None
                        
                        # Start a new transaction
                        self.db.rollback()  # Clear any existing transaction
                        
                        # Create new position with model's recommended levels
                        position = self.position_manager.create_position(
                            symbol=market_context.get('symbol', 'BTC/USDT'),
                            position_type=position_type,
                            entry_price=signal['entry_price'],
                            stop_loss=signal['stop_loss'],
                            take_profit=signal['take_profit'],
                            size=position_size,
                            timeframe=market_context.get('timeframe', '5m')
                        )
                        
                        if position:
                            # Update position with initial analysis
                            self.position_manager.update_position_with_analysis(
                                position_id=position.id,
                                current_price=signal['entry_price'],  # Use entry price as current price for initial analysis
                                signal=signal,
                                market_structure=signal.get('market_structure', None)
                            )
                            
                            # Commit the transaction
                            self.db.commit()
                            
                            console.print("[green]Position created and persisted successfully![/green]")
                            console.print(f"Position ID: {position.id}")
                            console.print(f"Type: {position.position_type}")
                            console.print(f"Entry Price: ${position.entry_price:.2f}")
                            console.print(f"Stop Loss: ${position.stop_loss:.2f}")
                            console.print(f"Take Profit: ${position.take_profit:.2f}")
                            console.print(f"Position Size: {position.size:.4f}")
                            console.print(f"Risk:Reward Ratio: {risk_reward_ratio:.2f}")
                            
                            return position
                        else:
                            console.print("[red]Failed to create position: Position manager returned None[/red]")
                            self.db.rollback()
                            return None
                            
                    except Exception as e:
                        console.print(f"[red]Error during position creation: {str(e)}[/red]")
                        self.db.rollback()
                        return None
                else:
                    console.print(f"[yellow]Warning: Risk-reward ratio {risk_reward_ratio:.2f} below minimum {min_risk_reward}[/yellow]")
                    return None
            else:
                console.print("[yellow]Warning: Signal type is HOLD, skipping position creation[/yellow]")
                return None
                
        except Exception as e:
            console.print(f"[red]Error creating position: {str(e)}[/red]")
            self.db.rollback()
            return None

    def _generate_model_signal(self, market_context: dict) -> dict:
        """Generate raw signal from the model"""
        try:
            console.print("\n[bold cyan]Debug: Entering _generate_model_signal[/bold cyan]")
            console.print(f"[cyan]Model name: {self.model_name}[/cyan]")
            
            if self.model_name == "gemini":
                console.print("[cyan]Using Gemini model for generation...[/cyan]")
                response = self.model.generate_response(market_context)
                # console.print("\n[bold cyan]Debug: Response from Gemini model:[/bold cyan]")
                # console.print(response)
                return response
            else:
                console.print("[cyan]Using local model for generation...[/cyan]")
                response = self.model.generate_response(market_context)
                console.print("\n[bold cyan]Debug: Response from local model:[/bold cyan]")
                console.print(response)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating model signal: {str(e)}")
            console.print(f"[bold red]Error in _generate_model_signal: {str(e)}[/bold red]")
            import traceback
            console.print(f"[bold red]Traceback:\n{traceback.format_exc()}[/bold red]")
            return None
    
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
        """Display the trading signal and position management details"""
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
        
        # Add position management details
        if 'position_management' in signal_data:
            pm = signal_data['position_management']
            table.add_row("Position Action", pm['action'])
            table.add_row("Risk:Reward Ratio", f"{pm['risk_reward_ratio']:.2f}")
        
        # Display the table
        console.print(table)
        
        # Display reasoning in a panel
        console.print(Panel(
            signal_data["reasoning"],
            title="Signal Reasoning",
            border_style="blue"
        ))
        
        # Display position management details in a separate panel
        if 'position_management' in signal_data:
            pm = signal_data['position_management']
            position_details = f"""
            Stop Loss Adjustment: {pm['stop_loss_adjustment']}
            Take Profit Adjustment: {pm['take_profit_adjustment']}
            """
            console.print(Panel(
                position_details,
                title="Position Management Details",
                border_style="magenta"
            ))
    
    def _display_position(self, position, current_price):
        """Display position details in a formatted table"""
        table = Table(title=f"Position {position.id}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Calculate PnL
        if position.position_type == "LONG":
            pnl = (current_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - current_price) / position.entry_price * 100
        
        table.add_row("Type", position.position_type)
        table.add_row("Status", position.status)
        table.add_row("Entry Price", f"${position.entry_price:.2f}")
        table.add_row("Current Price", f"${current_price:.2f}")
        table.add_row("Stop Loss", f"${position.stop_loss:.2f}")
        table.add_row("Take Profit", f"${position.take_profit:.2f}")
        table.add_row("PnL", f"{pnl:.2f}%")
        
        console.print(table)
    
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