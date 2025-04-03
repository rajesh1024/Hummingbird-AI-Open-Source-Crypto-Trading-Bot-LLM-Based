import logging
from typing import Dict, Optional
from datetime import datetime
from src.llm.analyzer import LLMAnalyzer
from src.technical.analysis import TechnicalAnalysis
from src.data.market_data import MarketData

logger = logging.getLogger(__name__)

class TradingSignalManager:
    def __init__(self):
        """Initialize trading signal manager"""
        self.latest_signal = None
        self.symbol = "BTC/USDT"  # Default symbol
        self.llm_analyzer = None
        self.market_data = None
        self.technical_analysis = None

    def initialize(self, llm_analyzer: LLMAnalyzer, market_data: MarketData, technical_analysis: TechnicalAnalysis):
        """Initialize with required components"""
        self.llm_analyzer = llm_analyzer
        self.market_data = market_data
        self.technical_analysis = technical_analysis

    def get_latest_signal(self) -> Optional[Dict]:
        """Get the latest trading signal"""
        try:
            if not self.llm_analyzer:
                logger.warning("LLM Analyzer not initialized")
                return None

            # Generate new signal using LLM analyzer
            signal = self.llm_analyzer.generate_signal(self.symbol)
            if signal:
                self.latest_signal = {
                    'symbol': self.symbol,
                    'signal': signal.get('signal', 'NEUTRAL'),
                    'reason': signal.get('reason', ''),
                    'timestamp': datetime.now(),
                    'confidence': signal.get('confidence', 0.0)
                }
                return self.latest_signal
            return None
        except Exception as e:
            logger.error(f"Error getting latest signal: {str(e)}")
            return None

    def set_signal(self, signal: Dict):
        """Set a new trading signal"""
        try:
            self.latest_signal = {
                'symbol': signal.get('symbol', self.symbol),
                'signal': signal.get('signal', 'NEUTRAL'),
                'reason': signal.get('reason', ''),
                'timestamp': datetime.now(),
                'confidence': signal.get('confidence', 0.0)
            }
        except Exception as e:
            logger.error(f"Error setting signal: {str(e)}")

    def set_symbol(self, symbol: str):
        """Set the current trading symbol"""
        self.symbol = symbol 