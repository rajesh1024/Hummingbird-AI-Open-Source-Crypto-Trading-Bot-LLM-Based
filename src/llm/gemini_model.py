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
        
        # Configure the Gemini API
        console.print("[bold cyan]Debug: Configuring Gemini API[/bold cyan]")
        genai.configure(api_key=self.api_key)
        
        try:
            # List available models and verify model name
            console.print("[bold cyan]Debug: Listing available models...[/bold cyan]")
            available_models = [m.name for m in genai.list_models()]
            console.print(f"[cyan]Available models: {available_models}[/cyan]")
            
            # Check if model_name is available, if not use gemini-pro
            if model_name not in available_models:
                console.print(f"[yellow]Warning: Model {model_name} not found. Falling back to gemini-pro[/yellow]")
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
            console.print("\n[bold cyan]Debug: Formatted Prompt:[/bold cyan]")
            console.print(prompt)
            
            # Simple generation config
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.9,
            }
            
            # Generate response
            console.print("\n[bold cyan]Debug: Generating response...[/bold cyan]")
            response = self.model.generate_content(prompt)
            
            if not response:
                console.print("[bold red]Error: No response generated[/bold red]")
                return self._get_default_response()
                
            console.print("\n[bold cyan]Debug: Raw response:[/bold cyan]")
            console.print(response.text)
            
            try:
                # Try to parse JSON directly
                parsed = json.loads(response.text)
                console.print("\n[bold green]Successfully parsed JSON response[/bold green]")
                return self._validate_response(parsed)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        console.print("\n[bold yellow]Successfully extracted and parsed JSON[/bold yellow]")
                        return self._validate_response(parsed)
                    except:
                        pass
                
                console.print("[bold red]Failed to parse response as JSON[/bold red]")
                return self._get_default_response()
                
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
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

    def _validate_response(self, response: Dict) -> Dict:
        """Validate the response from the model"""
        try:
            # Basic structure validation
            required_fields = ['signal', 'confidence', 'reasoning', 'entry_price', 'stop_loss', 'take_profit']
            if not all(field in response for field in required_fields):
                raise ValueError(f"Missing required fields: {[f for f in required_fields if f not in response]}")
            
            # Signal validation
            if response['signal'] not in ['BUY', 'SELL', 'HOLD']:
                raise ValueError(f"Invalid signal: {response['signal']}")
            
            # Let confidence pass through without validation
            
            # Price validations
            for field in ['entry_price', 'stop_loss', 'take_profit']:
                if not isinstance(response[field], (int, float)):
                    response[field] = float(response[field])
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error validating response: {str(e)}")
            return self._get_default_response()

    def _format_prompt(self, market_context: dict) -> str:
        """Format the prompt with market context"""
        try:
            # Get current price and basic market data
            current_price = market_context.get('current_price', 0)
            technical_indicators = market_context.get('technical_indicators', {})
            active_positions = market_context.get('active_positions', [])
            
            # Create base market context
            prompt = f"""You are a cryptocurrency trading analyst. Analyze the market data and provide a trading signal in STRICT JSON format.

                    CURRENT MARKET CONDITIONS:
                    Current Price: ${current_price}

                    TECHNICAL INDICATORS:
                    RSI: {technical_indicators.get('RSI', 'N/A')}
                    MACD: {technical_indicators.get('MACD', 'N/A')}
                    MACD Signal: {technical_indicators.get('MACD_Signal', 'N/A')}
                    EMA_8: {technical_indicators.get('EMA_8', 'N/A')}
                    EMA_21: {technical_indicators.get('EMA_21', 'N/A')}
                    """

            # Add active positions if they exist
            if active_positions:
                prompt += "\nACTIVE POSITIONS:\n"
                for pos in active_positions:
                    pnl = ((current_price - pos.entry_price) / pos.entry_price * 100) if pos.position_type == "LONG" else ((pos.entry_price - current_price) / pos.entry_price * 100)
                    prompt += f"""Type: {pos.position_type}
                                Entry: ${pos.entry_price:.2f}
                                Current: ${current_price:.2f}
                                Stop Loss: ${pos.stop_loss:.2f}
                                Take Profit: ${pos.take_profit:.2f}
                                PnL: {pnl:.2f}%

                                """

            # Add response requirements
            prompt += """
            IMPORTANT: Respond ONLY with a JSON object in the following format. DO NOT include any other text or explanations:

            {
                "signal": "BUY",  // Must be exactly "BUY", "SELL", or "HOLD"
                "confidence": 0.75,  // Number between 0 and 1
                "entry_price": 42000.00,  // Current price for new positions
                "stop_loss": 41500.00,  // Stop loss price level
                "take_profit": 43000.00,  // Take profit price level
                "reasoning": "Brief analysis explanation",
                "position_management": {
                    "action": "MAINTAIN",  // Must be "UPDATE", "CLOSE", or "MAINTAIN"
                    "stop_loss_adjustment": "None",  // Adjustment explanation or "None"
                    "take_profit_adjustment": "None",  // Adjustment explanation or "None"
                    "risk_reward_ratio": 2.5  // Must be >= 2.0
                }
            }

            RULES:
            1. Response MUST be valid JSON
            2. All fields are required
            3. Price levels must be realistic
            4. Risk:reward ratio must be >= 2.0
            5. Confidence must be between 0 and 1
            6. Signal must be exactly "BUY", "SELL", or "HOLD"
            7. Position action must be exactly "UPDATE", "CLOSE", or "MAINTAIN"

            Analyze the market data and provide your signal now:"""
            
            return prompt
            
        except Exception as e:
            console.print(f"[bold red]Error formatting prompt: {str(e)}[/bold red]")
            return ""

    def _parse_response(self, response_text: str) -> dict:
        """Parse and validate the response from the model"""
        try:
            signal_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'reasoning', 'position_management']
            if not all(field in signal_data for field in required_fields):
                raise ValueError("Missing required fields in response")
            
            # Validate confidence range
            if not 0 <= signal_data['confidence'] <= 1:
                raise ValueError("Confidence must be between 0 and 1")
            
            # Validate price levels
            if signal_data['entry_price'] <= 0 or signal_data['stop_loss'] <= 0 or signal_data['take_profit'] <= 0:
                raise ValueError("Price levels must be positive")
            
            return signal_data
            
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error parsing JSON response: {str(e)}[/bold red]")
            return self._get_default_response()
        except Exception as e:
            console.print(f"[bold red]Error validating response: {str(e)}[/bold red]")
            return self._get_default_response() 