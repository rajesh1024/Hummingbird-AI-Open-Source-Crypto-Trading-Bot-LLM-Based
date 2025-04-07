from fastapi import APIRouter
from src.interface.dashboard.symbol_routes import router as symbol_router

router = APIRouter()

# Include symbol routes without the /api prefix since it will be added by the main app
router.include_router(symbol_router, tags=["symbols"]) 