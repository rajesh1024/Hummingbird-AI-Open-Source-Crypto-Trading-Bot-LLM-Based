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
            """
            
            # Combine system prompt, format instructions, and user prompt
            full_prompt = f"{self.system_prompt}\n\n{json_format_instructions}\n\n{prompt}" if self.system_prompt else f"{json_format_instructions}\n\n{prompt}"
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            )
            
            # Get the response text
            response_text = response.text
            
            # Debug: Print raw response
            # console.print("[yellow]Raw model response:[/yellow]")
            # console.print(response_text)
            
            # Clean the response text to ensure valid JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Remove any control characters and normalize whitespace
            response_text = ''.join(char for char in response_text if char.isprintable() or char in '\n\r\t')
            response_text = response_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            
            # Debug: Print cleaned response
            # console.print("[yellow]Cleaned response:[/yellow]")
            # console.print(response_text)
            
            # Parse the cleaned JSON
            try:
                parsed_response = json.loads(response_text)
                
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
                if not isinstance(parsed_response['confidence'], (int, float)) or not 0 <= parsed_response['confidence'] <= 1:
                    raise ValueError("Confidence must be a float between 0 and 1")
                if not isinstance(parsed_response['reasoning'], str):
                    raise ValueError("Reasoning must be a string")
                if not isinstance(parsed_response['entry_price'], (int, float)):
                    raise ValueError("Entry price must be a number")
                if not isinstance(parsed_response['stop_loss'], (int, float)):
                    raise ValueError("Stop loss must be a number")
                if not isinstance(parsed_response['take_profit'], (int, float)):
                    raise ValueError("Take profit must be a number")
                
                return parsed_response
                
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON response: {str(e)}[/red]")
                console.print(f"[yellow]Raw response: {response_text}[/yellow]")
                raise
            
        except Exception as e:
            console.print(f"[red]Error generating response: {str(e)}[/red]")
            raise 