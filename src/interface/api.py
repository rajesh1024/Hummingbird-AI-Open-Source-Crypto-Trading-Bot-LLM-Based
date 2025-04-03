from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from src.main import Hummingbird
import logging
from urllib.parse import quote, unquote
import os
import traceback
import sys

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount static files directory
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir, 'web')
app.mount("/static", StaticFiles(directory=web_dir), name="static")

async def generate_signals(symbol: str, request: Request):
    try:
        # Decode the URL-encoded symbol
        decoded_symbol = unquote(symbol)
        logger.info(f"SSE connection request for symbol: {decoded_symbol}")
        
        # Initialize Hummingbird with the decoded symbol
        logger.debug(f"Initializing Hummingbird for symbol: {decoded_symbol}")
        hummingbird = Hummingbird(config_path="config/config.yaml")
        hummingbird.symbol = decoded_symbol
        hummingbird.trading_mode = "scalping"  # Changed default to scalping
        
        # Send initial connection message
        yield f"data: {json.dumps({'status': 'connected', 'symbol': decoded_symbol})}\n\n"
        
        # Start signal generation
        logger.debug(f"Starting signal generation for {decoded_symbol}")
        async for signal in hummingbird.generate_signals():
            # Check if client is still connected
            if await request.is_disconnected():
                logger.info(f"Client disconnected for symbol: {decoded_symbol}")
                break
                
            try:
                # Send the signal as SSE data
                yield f"data: {json.dumps(signal)}\n\n"
                logger.info(f"Sent signal for {decoded_symbol}")
            except Exception as e:
                logger.error(f"Error sending signal: {str(e)}")
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
                
    except Exception as e:
        logger.error(f"SSE error: {str(e)}")
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        logger.info(f"SSE connection closed for symbol: {decoded_symbol}")

@app.get("/sse/signals/{symbol}")
async def sse_endpoint(symbol: str, request: Request):
    return StreamingResponse(
        generate_signals(symbol, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

async def stream_logs(symbol: str, trading_mode: str = "scalping"):
    """
    Captures logs from main.py and streams them via SSE.
    """
    process = None
    try:
        # Get the path to main.py
        main_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        
        # Create the command with arguments
        cmd = [
            sys.executable,  # Use the current Python interpreter
            main_script,  # Use the full path to main.py
            "--symbol", symbol,
            "--model", "gemini",  # Add the model parameter
            "--mode", trading_mode
        ]
        
        logger.info(f"Starting subprocess with command: {' '.join(cmd)}")
        
        # Create subprocess with full environment
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        # Send initial connection message
        yield f"data: {json.dumps({'status': 'connected', 'message': f'Started process for {symbol}'})}\n\n"

        # Stream stdout and check stderr
        while True:
            try:
                # Read both stdout and stderr
                stdout_line = await process.stdout.readline()
                stderr_line = await process.stderr.readline()
                
                # Process stderr if available
                if stderr_line:
                    stderr_text = stderr_line.decode().strip()
                    logger.debug(f"Received stderr: {stderr_text}")
                    
                    # Check if it's a signal data
                    if stderr_text.startswith('LLM Response:'):
                        try:
                            logger.info(f"Found LLM Response: {stderr_text}")
                            json_str = stderr_text.split('LLM Response:')[1].strip()
                            signal_data = json.loads(json_str)
                            logger.info(f"Parsed signal data: {signal_data}")
                            
                            # Format the signal data for the frontend
                            formatted_signal = {
                                'signal': signal_data.get('signal', 'N/A'),
                                'confidence': float(signal_data.get('confidence', 0)),
                                'entry_price': float(signal_data.get('entry_price', 0)),
                                'stop_loss': float(signal_data.get('stop_loss', 0)),
                                'take_profit': float(signal_data.get('take_profit', 0)),
                                'reasoning': str(signal_data.get('reasoning', 'N/A')),
                                'current_price': float(signal_data.get('current_price', 0)),
                                'active_positions': int(signal_data.get('active_positions', 0)),
                                'position_management': signal_data.get('position_management', {})
                            }
                            logger.info(f"Formatted signal: {formatted_signal}")
                            yield f"event: signal\ndata: {json.dumps(formatted_signal)}\n\n"
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing LLM response: {e}")
                            logger.error(f"Raw stderr text: {stderr_text}")
                            yield f"event: terminal\ndata: {json.dumps({'message': stderr_text})}\n\n"
                    else:
                        # Send as terminal output
                        yield f"event: terminal\ndata: {json.dumps({'message': stderr_text})}\n\n"
                
                # Process stdout if available
                if stdout_line:
                    stdout_text = stdout_line.decode().strip()
                    logger.debug(f"Received stdout: {stdout_text}")
                    try:
                        # Try to parse as JSON if possible
                        data = json.loads(stdout_text)
                        logger.info(f"Parsed stdout JSON: {data}")
                        # Send as signal data event
                        yield f"event: signal\ndata: {json.dumps(data)}\n\n"
                    except json.JSONDecodeError:
                        # If not JSON, send as terminal output
                        yield f"event: terminal\ndata: {json.dumps({'message': stdout_text})}\n\n"
                
                # Break if both streams are empty
                if not stdout_line and not stderr_line:
                    break
                    
                await asyncio.sleep(0.1)  # Prevent tight loop
            except Exception as e:
                logger.error(f"Error processing stream: {str(e)}")
                logger.error(traceback.format_exc())
                yield f"event: terminal\ndata: {json.dumps({'error': f'Error processing stream: {str(e)}'})}\n\n"
                break

        # Final check for any remaining stderr content
        if process.returncode != 0:
            try:
                error_output = await process.stderr.read()
                if error_output:
                    error_message = error_output.decode().strip()
                    yield f"event: terminal\ndata: {json.dumps({'message': error_message})}\n\n"
            except Exception as e:
                logger.error(f"Error reading final stderr: {str(e)}")
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Error in stream_logs: {str(e)}")
        logger.error(traceback.format_exc())
        yield f"event: terminal\ndata: {json.dumps({'error': str(e)})}\n\n"
    finally:
        if process:
            try:
                process.terminate()
                await process.wait()
            except ProcessLookupError:
                logger.debug("Process already terminated")
            except Exception as e:
                logger.error(f"Error terminating process: {str(e)}")
                logger.error(traceback.format_exc())

@app.get("/stream/{symbol}")
async def stream_endpoint(request: Request, symbol: str, mode: str = "scalping"):
    """
    Endpoint to stream logs from main.py
    """
    try:
        # Log the incoming request
        logger.info(f"Received stream request for symbol: {symbol}, mode: {mode}")
        logger.debug(f"Request headers: {request.headers}")
        logger.debug(f"Request query params: {request.query_params}")
        
        # Convert underscore back to forward slash and decode
        symbol = unquote(symbol.replace('_', '/'))
        logger.info(f"Decoded symbol: {symbol}")
        
        # Validate symbol
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
            
        # Validate mode
        valid_modes = ["swing", "scalping", "position"]
        if mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}")
            
        # Check if main.py exists
        main_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        if not os.path.exists(main_script):
            raise HTTPException(status_code=500, detail=f"main.py not found at {main_script}")
            
        logger.info(f"Using main script at: {main_script}")
            
        async def event_generator():
            try:
                async for event in stream_logs(symbol, mode):
                    if await request.is_disconnected():
                        logger.info("Client disconnected, stopping event generator")
                        break
                    yield event
            except Exception as e:
                logger.error(f"Error in event generator: {str(e)}")
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                logger.info(f"Event generator closed for symbol: {symbol}")
            
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Headers": "*",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        logger.error(f"Error in stream_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Serve the web interface"""
    try:
        return FileResponse(os.path.join(web_dir, 'index.html'))
    except Exception as e:
        logger.error(f"Error serving web interface: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve web interface")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.interface.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=True
    ) 