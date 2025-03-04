from ctransformers import AutoModelForCausalLM
import pandas as pd
from typing import Dict, List, Optional
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
        self.model = self._load_model()
        
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
                if not api_key:
                    raise ValueError("Gemini API key not found in .env file")
                
                return GeminiModel(
                    api_key=api_key,
                    model_name=model_settings['model_name'],
                    max_tokens=model_settings['max_tokens'],
                    temperature=model_settings['temperature'],
                    system_prompt=model_settings['system_prompt']
                )
            
            else:
                raise ValueError(f"Unsupported model type: {model_settings['type']}")
                
        except Exception as e:
            console.print(f"[bold red]Error loading model: {str(e)}")
            raise
    
    def prepare_market_context(
        self,
        historical_data: Dict[str, pd.DataFrame],
        technical_indicators: Dict[str, Dict],
        order_blocks: List[Dict],
        supply_zones: List[Dict],
        demand_zones: List[Dict]
    ) -> str:
        """
        Prepare market context for LLM analysis
        """
        context = []
        
        # Add technical indicators
        context.append("Technical Indicators:")
        for timeframe, indicators in technical_indicators.items():
            context.append(f"\n{timeframe} Timeframe:")
            for name, value in indicators.items():
                context.append(f"{name}: {value:.2f}")
        
        # Add SMC Analysis
        context.append("\nSmart Money Concepts (SMC) Analysis:")
        
        # Order Blocks
        if order_blocks:
            context.append("\nOrder Blocks:")
            for block in sorted(order_blocks, key=lambda x: x['strength'], reverse=True)[:3]:
                context.append(
                    f"- {block['type'].title()} OB at {block['price']:.2f} "
                    f"(Strength: {block['strength']:.2f}, TF: {block['timeframe']})"
                )
        
        # Supply/Demand Zones
        if supply_zones:
            context.append("\nSupply Zones:")
            for zone in sorted(supply_zones, key=lambda x: x['strength'], reverse=True)[:2]:
                context.append(
                    f"- Supply at {zone['price']:.2f} "
                    f"(Strength: {zone['strength']:.2f}, TF: {zone['timeframe']})"
                )
        
        if demand_zones:
            context.append("\nDemand Zones:")
            for zone in sorted(demand_zones, key=lambda x: x['strength'], reverse=True)[:2]:
                context.append(
                    f"- Demand at {zone['price']:.2f} "
                    f"(Strength: {zone['strength']:.2f}, TF: {zone['timeframe']})"
                )
        
        # Fair Value Gaps
        if self.technical_analyzer and self.technical_analyzer.fair_value_gaps:
            context.append("\nFair Value Gaps (FVG):")
            for gap in self.technical_analyzer.fair_value_gaps:
                context.append(
                    f"- {gap['type'].title()} FVG at {gap['price']:.2f} "
                    f"(TF: {gap['timeframe']})"
                )
        
        # Liquidity Sweeps
        if self.technical_analyzer and self.technical_analyzer.liquidity_sweeps:
            context.append("\nLiquidity Sweeps:")
            for sweep in self.technical_analyzer.liquidity_sweeps:
                context.append(
                    f"- {sweep['type'].title()} sweep at {sweep['price']:.2f} "
                    f"(TF: {sweep['timeframe']})"
                )
        
        # Add market structure
        context.append("\nMarket Structure:")
        for timeframe, df in historical_data.items():
            if len(df) >= 20:
                recent_highs = df['high'].tail(20)
                recent_lows = df['low'].tail(20)
                context.append(f"\n{timeframe} Timeframe:")
                context.append(f"Recent Highs: {recent_highs.max():.2f}")
                context.append(f"Recent Lows: {recent_lows.min():.2f}")
        
        # Debug print to verify context
        console.print("\n[bold yellow]Market Context Debug:")
        console.print("\n".join(context))
        
        return "\n".join(context)
    
    def generate_trading_signal(
        self,
        market_context: str,
        current_price: float,
        confidence_threshold: float = 0.95
    ) -> Dict:
        """
        Generate trading signal using LLM
        """
        # Format prompt based on model type
        model_settings = self.model_config['models'][self.model_name]
        
        if model_settings['type'] == 'local':
            # Mistral format
            prompt = f"""<s>[INST] {model_settings['system_prompt']}

Market Context:
{market_context}

Current Price: ${current_price:.2f}

Please analyze the market and provide a trading signal in JSON format with the following structure:
{{
    "signal": "LONG" or "SHORT" or "NEUTRAL",
    "confidence": float between 0 and 1,
    "entry_price": float,
    "stop_loss": float,
    "take_profits": [float],
    "reasoning": "detailed explanation"
}}

Important Risk Management Rules:
1. For LONG signals:
   - Stop Loss should be placed at the nearest significant SMC level below (PDL, Order Block, or Demand Zone)
   - Take Profit targets should be placed at the next significant SMC levels above
   Example: If current price is $87,000 with PDL at $85,000 and next resistance at $90,000:
   - Entry: $87,000
   - Stop Loss: $85,000
   - Take Profit: $90,000

2. For SHORT signals:
   - Stop Loss should be placed at the nearest significant SMC level above (PDH, Order Block, or Supply Zone)
   - Take Profit targets should be placed at the next significant SMC levels below
   Example: If current price is $87,000 with PDH at $89,000 and next support at $85,000:
   - Entry: $87,000
   - Stop Loss: $89,000
   - Take Profit: $85,000

3. General Rules:
   - Risk:Reward ratio should be at least 1:2
   - Maximum stop loss distance should not exceed 2% of entry price
   - Take profit targets should be realistic and based on market structure
   - Only provide entry, stop loss, and take profits for LONG or SHORT signals
   - For NEUTRAL signals, set all price levels to current price

[/INST]</s>"""
        else:
            # Gemini format
            prompt = f"""Market Context:
{market_context}

Current Price: ${current_price:.2f}

Please analyze the market and provide a trading signal in JSON format with the following structure:
{{
    "signal": "LONG" or "SHORT" or "NEUTRAL",
    "confidence": float between 0 and 1,
    "entry_price": float,
    "stop_loss": float,
    "take_profits": [float],
    "reasoning": "detailed explanation"
}}

Important Risk Management Rules:
1. For LONG signals:
   - Stop Loss should be placed at the nearest significant SMC level below (PDL, Order Block, or Demand Zone)
   - Take Profit targets should be placed at the next significant SMC levels above
   Example: If current price is $87,000 with PDL at $85,000 and next resistance at $90,000:
   - Entry: $87,000
   - Stop Loss: $85,000
   - Take Profit: $90,000

2. For SHORT signals:
   - Stop Loss should be placed at the nearest significant SMC level above (PDH, Order Block, or Supply Zone)
   - Take Profit targets should be placed at the next significant SMC levels below
   Example: If current price is $87,000 with PDH at $89,000 and next support at $85,000:
   - Entry: $87,000
   - Stop Loss: $89,000
   - Take Profit: $85,000

3. General Rules:
   - Risk:Reward ratio should be at least 1:2
   - Maximum stop loss distance should not exceed 2% of entry price
   - Take profit targets should be realistic and based on market structure
   - Only provide entry, stop loss, and take profits for LONG or SHORT signals
   - For NEUTRAL signals, set all price levels to current price"""
        
        # Print only the market context for debugging
        console.print("\n[bold cyan]Market Context:[/bold cyan]")
        console.print(Panel(
            f"System Prompt:\n{model_settings['system_prompt']}\n\n"
            f"Market Data:\n{market_context}\n\n"
            f"Current Price: ${current_price:.2f}",
            title="Analysis Context",
            border_style="cyan"
        ))
        
        try:
            with console.status("[bold yellow]Generating trading signal...") as status:
                if model_settings['type'] == 'local':
                    response = self.model(prompt)
                else:
                    response = self.model.generate_response(prompt)
                signal_data = self._parse_llm_response(response)
            
            if signal_data["confidence"] >= confidence_threshold:
                self._display_signal(signal_data, current_price)
                return signal_data
            else:
                console.print("[bold yellow]Signal confidence below threshold")
                return {
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "entry_price": current_price,
                    "stop_loss": current_price,
                    "take_profits": [current_price],
                    "reasoning": "Confidence below threshold"
                }
                
        except Exception as e:
            console.print(f"[bold red]Error generating trading signal: {str(e)}")
            raise
    
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
    
    def _display_signal(self, signal: Dict, current_price: float):
        """
        Display trading signal in a beautiful format
        """
        table = Table(title="Trading Signal Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Always show signal and confidence
        table.add_row("Signal", f"[bold {'green' if signal['signal'] == 'LONG' else 'red' if signal['signal'] == 'SHORT' else 'yellow'}]{signal['signal']}")
        table.add_row("Confidence", f"{signal['confidence']:.2%}")
        
        # Only show entry, stop loss, and take profits for active signals
        if signal['signal'] in ['LONG', 'SHORT']:
            table.add_row("Entry Price", f"${signal['entry_price']:.2f}")
            table.add_row("Stop Loss", f"${signal['stop_loss']:.2f}")
            table.add_row("Take Profits", ", ".join([f"${tp:.2f}" for tp in signal['take_profits']]))
        
        console.print(table)
        
        # Display reasoning in a panel
        console.print(Panel(
            signal['reasoning'],
            title="Signal Reasoning",
            border_style="blue"
        )) 