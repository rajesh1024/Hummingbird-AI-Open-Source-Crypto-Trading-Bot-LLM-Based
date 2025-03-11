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
        console.print("[bold cyan]Debug: Initializing Gemini model[/bold cyan]")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        # Configure the Gemini API
        console.print("[bold cyan]Debug: Configuring Gemini API[/bold cyan]")
        genai.configure(api_key=self.api_key)
        
        try:
            # Test the connection
            console.print("[bold cyan]Debug: Testing Gemini connection[/bold cyan]")
            self.model = genai.GenerativeModel(self.model_name)
            console.print("[green]✓ Successfully connected to Gemini API[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to connect to Gemini API: {str(e)}[/red]")
            raise

    def generate_response(self, market_context: str) -> Dict:
        """Generate a response from the Gemini model"""
        try:
            console.print("[bold cyan]Debug: Starting generate_response in Gemini model[/bold cyan]")
            
            # Format the prompt more neutrally
            prompt = f"""
            Market Analysis Request:
            
            Please analyze the following market data and provide a trading recommendation.
            Follow the analysis guidelines and provide your response in JSON format.
            
            Guidelines:
            - Analyze multiple timeframes
            - Consider technical indicators
            - Evaluate market structure
            - Assess risk/reward ratio
            - Provide clear entry/exit points
            
            {market_context}
            
            Respond with a JSON object containing:
            {{
                "signal": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "explanation",
                "entry_price": number,
                "stop_loss": number,
                "take_profit": number
            }}
            """
            
            console.print(f"[bold cyan]Debug: Combined prompt length: {len(prompt)}[/bold cyan]")
            console.print("[bold cyan]Debug: Full prompt being sent to Gemini:[/bold cyan]\n" + prompt)
            
            # Call the Gemini API
            console.print("[bold cyan]Debug: Calling Gemini API[/bold cyan]")
            console.print(f"[bold cyan]Debug: About to make API call with prompt length: {len(prompt)}[/bold cyan]")
            
            try:
                response = self.model.generate_content(prompt)
                if response and response.text:
                    # Parse the response text as JSON
                    response_text = response.text.strip()
                    console.print(f"[bold green]Raw response from Gemini:[/bold green]\n{response_text}")
                    
                    # Extract JSON from response if needed
                    if '{' in response_text:
                        json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                        return json.loads(json_str)
                    else:
                        console.print("[bold red]No JSON found in response[/bold red]")
                        return self._get_default_response()
                else:
                    console.print("[bold red]Empty response from Gemini[/bold red]")
                    return self._get_default_response()
                    
            except Exception as e:
                console.print(f"[bold red]Error during Gemini API call: {str(e)}[/bold red]")
                console.print("[yellow]API call failed, returning default response[/yellow]")
                return self._get_default_response()
                
        except Exception as e:
            console.print(f"[bold red]Error in generate_response: {str(e)}[/bold red]")
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