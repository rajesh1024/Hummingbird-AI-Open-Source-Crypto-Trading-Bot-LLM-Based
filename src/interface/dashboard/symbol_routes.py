from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from src.data.symbol_manager import SymbolManager
from src.main import Hummingbird

router = APIRouter()

# Initialize Hummingbird instance
config_path = "../../../config/config.yaml"  # Update this path as needed
hummingbird = Hummingbird(config_path)

# Initialize SymbolManager with Hummingbird instance
symbol_manager = SymbolManager(hummingbird)

class SymbolRequest(BaseModel):
    symbol: str

@router.post("/symbols/start")
async def start_symbol(request: SymbolRequest):
    """Start analysis for a symbol"""
    success = await symbol_manager.start_symbol(request.symbol)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to start symbol {request.symbol}")
    return {"status": "success", "message": f"Started analysis for {request.symbol}"}

@router.post("/symbols/stop")
async def stop_symbol(request: SymbolRequest):
    """Stop analysis for a symbol"""
    success = await symbol_manager.stop_symbol(request.symbol)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to stop symbol {request.symbol}")
    return {"status": "success", "message": f"Stopped analysis for {request.symbol}"}

@router.get("/symbols/status")
async def get_symbol_status(symbol: str = None):
    """Get status of a specific symbol or all symbols"""
    if symbol:
        return symbol_manager.get_symbol_status(symbol)
    return symbol_manager.get_all_symbols_status() 