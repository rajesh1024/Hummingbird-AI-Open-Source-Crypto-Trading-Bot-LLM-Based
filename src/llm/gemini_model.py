import google.generativeai as genai
from typing import Dict, Any
import json
import logging
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

class GeminiModel:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = None
    ):
        console.print("[bold cyan]Debug: Initializing Gemini model[/bold cyan]")
        
        if not api_key:
            raise ValueError("API key is required")
            
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Configure the Gemini API
        console.print("[bold cyan]Debug: Configuring Gemini API[/bold cyan]")
        genai.configure(api_key=self.api_key)
        
        try:
            # List available models and verify model name
            # console.print("[bold cyan]Debug: Listing available models...[/bold cyan]")
            # available_models = [m.name for m in genai.list_models()]
            # console.print(f"[cyan]Available models: {available_models}[/cyan]")
            
            # Check if model_name is available, if not use gemini-pro
            # if model_name not in available_models:
            #     console.print(f"[yellow]Warning: Model {model_name} not found. Falling back to gemini-pro[/yellow]")
            #     model_name = "gemini-1.5-flash"
            model_name = "gemini-1.5-flash"
            self.model_name = model_name
            console.print(f"[bold cyan]Debug: Using model: {self.model_name}[/bold cyan]")
            
            # Create model with generation config
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.9,
                "top_k": 40,
                "candidate_count": 1,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            # Test with a simple prompt
            test_prompt = "Respond with 'OK' if you can understand this message."
            console.print(f"[bold cyan]Debug: Testing model with prompt: {test_prompt}[/bold cyan]")
            
            try:
                test_response = self.model.generate_content(test_prompt)
                if test_response and hasattr(test_response, 'text'):
                    console.print(f"[bold cyan]Debug: Test response: {test_response.text}[/bold cyan]")
                    console.print("[green]✓ Successfully connected to Gemini API[/green]")
                else:
                    raise ValueError("Test response was empty or invalid")
            except Exception as test_error:
                console.print(f"[red]✗ Model test failed: {str(test_error)}[/red]")
                raise
                
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize Gemini model: {str(e)}[/red]")
            console.print(f"[red]Error type: {type(e)}[/red]")
            raise

    def generate_response(self, market_context: dict) -> dict:
        """Generate a response from the Gemini model"""
        try:
            # Format the prompt with market context
            prompt = self._format_prompt(market_context)
            
            # Debug: Print the formatted prompt
            # console.print("\n[bold cyan]Debug: Formatted Prompt:[/bold cyan]")
            # console.print(prompt)
            
            # Simple generation config
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.9,
            }
            
            # Generate response
            console.print("\n[bold cyan]Debug: Generating response...[/bold cyan]")
            response = self.model.generate_content(prompt)
            
            # Debug: Print raw response object
            # console.print("\n[bold cyan]Debug: Raw response object:[/bold cyan]")
            # console.print(f"Response type: {type(response)}")
            # console.print(f"Response attributes: {dir(response)}")
            
            if not response:
                console.print("[bold red]Error: No response generated[/bold red]")
                return self._get_default_response()
            
            # Debug: Print raw response text
            # console.print("\n[bold cyan]Debug: Raw response text:[/bold cyan]")
            # console.print(response.text)
            
            try:
                # Try to parse JSON directly
                parsed = json.loads(response.text)
                console.print("\n[bold green]Successfully parsed JSON response[/bold green]")
                return self._validate_response(parsed)
            except json.JSONDecodeError as e:
                console.print(f"\n[yellow]JSON parse error: {str(e)}[/yellow]")
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        # console.print(f"\n[cyan]Extracted JSON string:[/cyan]\n{json_str}")
                        parsed = json.loads(json_str)
                        # console.print("\n[bold yellow]Successfully extracted and parsed JSON[/bold yellow]")
                        return self._validate_response(parsed)
                    except Exception as extract_error:
                        console.print(f"\n[red]Error extracting JSON: {str(extract_error)}[/red]")
                
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                return self._get_default_response()
                
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
            import traceback
            console.print(f"[bold red]Traceback:\n{traceback.format_exc()}[/bold red]")
            return self._get_default_response()

    def _get_default_response(self) -> Dict:
        """Return a default response in case of errors"""
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reasoning": "Error generating signal",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0
        }

    def _validate_response(self, response: Dict[str, Any]) -> Dict:
        """Validate the LLM response format and content."""
        try:
            # Check required fields
            required_fields = ['signal', 'confidence', 'reasoning', 'entry_price', 'stop_loss', 'take_profit']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                console.print(f"[yellow]Warning: Missing required fields: {missing_fields}[/yellow]")
                return self._get_default_response()

            # Validate signal
            if response['signal'] not in ['BUY', 'SELL', 'HOLD']:
                console.print(f"[yellow]Warning: Invalid signal: {response['signal']}[/yellow]")
                response['signal'] = 'HOLD'

            # For HOLD signals, set all price levels to 0
            # if response['signal'] == 'HOLD':
            #     # response['entry_price'] = 0
            #     # response['stop_loss'] = 0
            #     # response['take_profit'] = 0
            #     # Don't modify confidence - let the model's confidence value stand
            #     if 'position_management' not in response:
            #         response['position_management'] = {
            #             'action': 'MAINTAIN',
            #             'stop_loss_adjustment': 'None',
            #             'take_profit_adjustment': 'None',
            #             'risk_reward_ratio': 0.0
            #         }
            #     return response

            # Validate confidence
            if not isinstance(response['confidence'], (int, float)) or not 0 <= response['confidence'] <= 1:
                console.print(f"[yellow]Warning: Invalid confidence value: {response['confidence']}[/yellow]")
                response['confidence'] = 0.0

            # Validate price levels
            for field in ['entry_price', 'stop_loss', 'take_profit']:
                if not isinstance(response[field], (int, float)):
                    try:
                        response[field] = float(response[field])
                    except (ValueError, TypeError):
                        console.print(f"[yellow]Warning: Invalid {field}: {response[field]}[/yellow]")
                        response[field] = 0.0

            # Validate entry price relative to current price
            current_price = response.get('current_price', 0)
            if current_price > 0:
                if response['signal'] == 'BUY' and response['entry_price'] > current_price:
                    console.print(f"[yellow]Warning: Invalid entry price for LONG position: {response['entry_price']} > {current_price}[/yellow]")
                    response['entry_price'] = current_price
                elif response['signal'] == 'SELL' and response['entry_price'] < current_price:
                    console.print(f"[yellow]Warning: Invalid entry price for SHORT position: {response['entry_price']} < {current_price}[/yellow]")
                    response['entry_price'] = current_price

            # Validate reasoning
            if not isinstance(response['reasoning'], str) or len(response['reasoning']) < 50:
                console.print("[yellow]Warning: Invalid or too short reasoning[/yellow]")
                response['reasoning'] = "Error generating signal"

            # Validate SMC patterns in reasoning
            smc_keywords = ['order block', 'fair value gap', 'liquidity', 'smart money', 'market structure']
            if not any(keyword in response['reasoning'].lower() for keyword in smc_keywords):
                console.print("[yellow]Warning: Reasoning does not mention SMC patterns[/yellow]")

            # Validate risk-reward ratio
            entry = response['entry_price']
            sl = response['stop_loss']
            tp = response['take_profit']
            
            if response['signal'] == 'BUY':
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                if risk == 0:
                    console.print("[yellow]Warning: Invalid risk calculation (zero risk)[/yellow]")
                    return self._get_default_response()
                rr_ratio = reward / risk
                if rr_ratio < 1.0:  # Minimum 1:1 for scalping
                    console.print(f"[yellow]Warning: Risk-reward ratio {rr_ratio:.2f} below minimum 1:1[/yellow]")
            elif response['signal'] == 'SELL':
                risk = abs(entry - sl)
                reward = abs(entry - tp)
                if risk == 0:
                    console.print("[yellow]Warning: Invalid risk calculation (zero risk)[/yellow]")
                    return self._get_default_response()
                rr_ratio = reward / risk
                if rr_ratio < 1.0:  # Minimum 1:1 for scalping
                    console.print(f"[yellow]Warning: Risk-reward ratio {rr_ratio:.2f} below minimum 1:1[/yellow]")

            # Add position management details if not present
            if 'position_management' not in response:
                response['position_management'] = {
                    'action': 'MAINTAIN',
                    'stop_loss_adjustment': 'None',
                    'take_profit_adjustment': 'None',
                    'risk_reward_ratio': rr_ratio
                }

            console.print("[bold green]Response validation successful[/bold green]")
            return response

        except Exception as e:
            console.print(f"[bold red]Error validating response: {str(e)}[/bold red]")
            return self._get_default_response()

    def _format_prompt(self, market_context: dict) -> str:
        """Format the prompt with market context"""
        try:
            # Get current price and basic market data
            current_price = market_context.get('current_price', 0)
            active_positions = market_context.get('active_positions', [])
            smc_data = market_context.get('smc_data', {})
            
            # Create base market context
            prompt = f"""You are an expert cryptocurrency scalping analyst specializing in Smart Money Concepts (SMC) and technical analysis. Your goal is to identify high-probability trading opportunities in short-term price movements.

            Key Requirements:
            1. Focus on short-term price movements (5m to 15m timeframes)
            2. Identify clear entry points with good risk-reward ratios (minimum 1:2 for scalping)
            3. Use this strategy to scalp the market for quick profits in 5 mins to 15 mins timeframes
            4. Be proactive in identifying opportunities while maintaining strict risk management
            5. PRIMARY FOCUS: Smart Money Concepts (SMC) patterns:
            - Order Blocks: Look for recent price action forming blocks
            - Fair Value Gaps: Identify gaps between price levels
            - Liquidity Levels: Find areas of high liquidity (high volume nodes)
            - Smart Money Traps: Watch for false breakouts and traps

            Risk Management Rules:
            - Maximum risk per trade: 1% of portfolio
            - Minimum risk-reward ratio: 1:2 for scalping
            - Maximum daily loss: 3%
            - Maximum concurrent positions: 3

            Confidence Calculation Guidelines:
            - High confidence (0.7-1.0): Strong SMC pattern with technical confirmation
            - Medium confidence (0.5-0.7): Clear SMC pattern with partial technical confirmation
            - Low confidence (0.3-0.5): Weak SMC pattern or technical only
            - Below 0.3: No clear setup

            CURRENT MARKET CONDITIONS:
            Current Price: ${current_price}

            SMC PATTERNS:
            Order Blocks:
            {chr(10).join([f"- {block}" for block in smc_data.get('order_blocks', [])])}

            Fair Value Gaps:
            {chr(10).join([f"- {gap}" for gap in smc_data.get('fair_value_gaps', [])])}

            Liquidity Levels:
            {chr(10).join([f"- {level}" for level in smc_data.get('liquidity_levels', [])])}
            """
            # Add active positions if they exist
            if active_positions:
                prompt += "\nACTIVE POSITIONS:\n"
                for pos in active_positions:
                    pnl = ((current_price - pos.entry_price) / pos.entry_price * 100) if pos.position_type == "LONG" else ((pos.entry_price - current_price) / pos.entry_price * 100)
                    prompt += f"""Position ID: {pos.id}
                                    Type: {pos.position_type}
                                    Entry: ${pos.entry_price:.2f}
                                    Current: ${current_price:.2f}
                                    Stop Loss: ${pos.stop_loss:.2f}
                                    Take Profit: ${pos.take_profit:.2f}
                                    PnL: {pnl:.2f}%
                                    """
                prompt += """
                IMPORTANT: For active positions, focus on position management. Your response should:
                1. Monitor for stop loss or take profit hits
                2. Consider trailing stop loss adjustments if in profit
                3. Consider trailing take profit adjustments if close to target
                4. Provide clear reasoning for any position management decisions
                """
            else:
                prompt += """
                IMPORTANT: No active positions. Focus on identifying new trading opportunities. Your response should:
                1. Look for clear SMC patterns:
                - Recent order blocks near current price
                - Fair value gaps that need to be filled
                - Liquidity levels that could attract price
                2. Ensure entry price is valid (<= current price for LONG, >= current price for SHORT)
                3. Set appropriate stop loss and take profit levels
                4. Maintain minimum 1:1 risk-reward ratio
                5. If no clear SMC patterns are present, explain why in the reasoning
                """

            # Add response requirements
            prompt += """
            IMPORTANT: Respond ONLY with a JSON object in the following format. DO NOT include any other text or explanations:

            {
                "signal": "BUY",  // Must be exactly "BUY", "SELL", or "HOLD"
                "confidence": 0.75,  // Number between 0 and 1
                "entry_price": 42000.00,  // For new positions only, use current price for active positions
                "stop_loss": 41500.00,  // Stop loss price level
                "take_profit": 43000.00,  // Take profit price level
                "reasoning": "Brief analysis explanation",
                "position_management": {
                    "action": "MAINTAIN",  // Must be "UPDATE", "CLOSE", or "MAINTAIN"
                    "stop_loss_adjustment": "None",  // Adjustment explanation or "None"
                    "take_profit_adjustment": "None",  // Adjustment explanation or "None"
                    "risk_reward_ratio": 2.5  // Must be >= 1.0 for scalping
                }
            }

            RULES:
            1. Response MUST be valid JSON
            2. All fields are required
            3. For active positions, use the existing entry price
            4. Price levels must be realistic:
            - For LONG positions: entry_price must be <= current_price
            - For SHORT positions: entry_price must be >= current_price
            5. Risk:reward ratio must be >= 1.0 for scalping
            6. Confidence must be between 0 and 1
            7. Signal must be exactly "BUY", "SELL", or "HOLD"
            8. Position action must be exactly "UPDATE", "CLOSE", or "MAINTAIN"
            9. Reasoning must explain why SMC patterns are or are not present

            Analyze the market data and provide your signal now:"""
            
            return prompt
            
        except Exception as e:
            console.print(f"[bold red]Error formatting prompt: {str(e)}[/bold red]")
            return ""

    # def _parse_response(self, response_text: str) -> dict:
    #     """Parse and validate the response from the model"""
    #     try:
    #         signal_data = json.loads(response_text)
            
    #         # Validate required fields
    #         required_fields = ['signal', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'reasoning', 'position_management']
    #         if not all(field in signal_data for field in required_fields):
    #             raise ValueError("Missing required fields in response")
            
    #         # Validate confidence range
    #         if not 0 <= signal_data['confidence'] <= 1:
    #             raise ValueError("Confidence must be between 0 and 1")
            
    #         # Validate price levels
    #         if signal_data['entry_price'] <= 0 or signal_data['stop_loss'] <= 0 or signal_data['take_profit'] <= 0:
    #             raise ValueError("Price levels must be positive")
            
    #         return signal_data
            
    #     except json.JSONDecodeError as e:
    #         console.print(f"[bold red]Error parsing JSON response: {str(e)}[/bold red]")
    #         return self._get_default_response()
    #     except Exception as e:
    #         console.print(f"[bold red]Error validating response: {str(e)}[/bold red]")
    #         return self._get_default_response() 