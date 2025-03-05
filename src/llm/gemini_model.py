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
        model_name: str = "gemini-2.0-flash",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = None
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        try:
            # Test the connection
            self.model = genai.GenerativeModel(self.model_name)
            console.print("[green]✓ Successfully connected to Gemini API[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to Gemini API: {str(e)}[/red]")
            raise

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response from the Gemini model."""
        try:
            # Add explicit JSON format instructions to the prompt
            json_format_instructions = """
            You MUST respond with a valid JSON object containing ALL of the following fields:
            {
                "signal": "BUY",  // Must be exactly "BUY", "SELL", or "HOLD"
                "confidence": 0.85,  // Must be a number between 0 and 1
                "reasoning": "Detailed explanation of your analysis",
                "entry_price": 87768.61,  // Must be a number
                "stop_loss": 85000.00,  // Must be a number
                "take_profit": 90000.00  // Must be a number
            }
            
            Important:
            1. The response must be ONLY the JSON object, no other text
            2. All fields are required
            3. Numbers must be actual numbers, not strings
            4. The signal must be exactly "BUY", "SELL", or "HOLD"
            5. Confidence must be a number between 0 and 1
            """
            
            # Combine system prompt, format instructions, and user prompt
            full_prompt = f"{self.system_prompt}\n\n{json_format_instructions}\n\n{prompt}" if self.system_prompt else f"{json_format_instructions}\n\n{prompt}"
            
            # Generate response with timeout
            try:
                # Set a timeout of 30 seconds for the API call
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        "max_output_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    safety_settings={
                        "HARASSMENT": "block_none",
                        "HATE_SPEECH": "block_none",
                        "SEXUALLY_EXPLICIT": "block_none",
                        "DANGEROUS_CONTENT": "block_none",
                    }
                )
                
                # Get the response text
                response_text = response.text
                console.print(f"[bold yellow]Raw response from Gemini:[/bold yellow]\n{response_text}")
                
                # Clean the response text to ensure valid JSON
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Remove any control characters and normalize whitespace
                response_text = ''.join(char for char in response_text if char.isprintable() or char in '\n\r\t')
                response_text = response_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                
                # Parse the cleaned JSON
                try:
                    parsed_response = json.loads(response_text)
                    console.print(f"[bold green]Parsed JSON response:[/bold green]\n{json.dumps(parsed_response, indent=2)}")
                except json.JSONDecodeError as e:
                    console.print(f"[bold red]Error parsing JSON: {str(e)}")
                    raise
                
                # Ensure the response has the required fields
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response must be a JSON object")
                
                required_fields = ['signal', 'confidence', 'reasoning', 'entry_price', 'stop_loss', 'take_profit']
                missing_fields = [field for field in required_fields if field not in parsed_response]
                
                if missing_fields:
                    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
                
                # Validate field types
                if not isinstance(parsed_response['signal'], str) or parsed_response['signal'] not in ['BUY', 'SELL', 'HOLD']:
                    raise ValueError("Signal must be one of: BUY, SELL, HOLD")
                
                # Ensure confidence is a number
                try:
                    parsed_response['confidence'] = float(parsed_response['confidence'])
                    if not 0 <= parsed_response['confidence'] <= 1:
                        raise ValueError("Confidence must be between 0 and 1")
                except (ValueError, TypeError):
                    raise ValueError("Confidence must be a valid number between 0 and 1")
                
                # Ensure price fields are numbers
                for price_field in ['entry_price', 'stop_loss', 'take_profit']:
                    try:
                        parsed_response[price_field] = float(parsed_response[price_field])
                    except (ValueError, TypeError):
                        raise ValueError(f"{price_field} must be a valid number")
                
                return parsed_response
                
            except Exception as e:
                console.print(f"[bold red]Error generating response: {str(e)}")
                # Return a neutral response if there's an error
                return {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Error generating response",
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0
                }
                
        except Exception as e:
            console.print(f"[bold red]Error in generate_response: {str(e)}")
            raise 