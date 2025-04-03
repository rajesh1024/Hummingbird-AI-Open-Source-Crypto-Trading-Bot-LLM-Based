import gradio as gr
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.models import Position, AccountBalance, PositionStatus, PositionType
from src.data.database import get_database_url
import json
import time
from src.llm.analyzer import LLMAnalyzer
from src.technical.technical_analysis import TechnicalAnalysis
from src.technical.market_structure import MarketStructureAnalyzer
from src.data.market_data import MarketData
from src.main import Hummingbird
import yaml
import os
import threading
from queue import Queue
import logging
import subprocess
import sys
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDashboard:
    def __init__(self):
        self.engine = create_engine(get_database_url())
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Initialize components for market analysis
        self.config = self._load_config()
        
        # Initialize state
        self.signals = []
        self.last_update = 0
        self.signal_process = None
        self.is_running = False
        self.symbols = self.config.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.output_queue = Queue()
        self.output_thread = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config/config.yaml')
            
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return {
                'llm': {
                    'model': 'gemini',
                    'temperature': 0.7,
                    'max_tokens': 1000
                },
                'trading': {
                    'symbols': ['BTC/USDT', 'ETH/USDT'],
                    'signal_interval': 60
                }
            }

    def _collect_output(self):
        """Collect output from the process in a separate thread"""
        while self.is_running and self.signal_process:
            try:
                line = self.signal_process.stdout.readline()
                if not line:
                    break
                if not any(spinner in line for spinner in ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
                    self.output_queue.put(line)
                    logger.info(f"Received output: {line.strip()}")
            except Exception as e:
                logger.error(f"Error collecting output: {str(e)}")
                break

    def _start_signal_generation(self, symbol):
        """Start the WebSocket connection for signal generation"""
        try:
            # URL encode the symbol
            encoded_symbol = quote(symbol)
            
            # Create WebSocket connection
            self.ws = gr.WebSocket(
                url=f"ws://localhost:8000/ws/signals/{encoded_symbol}",
                on_message=self._handle_signal
            )
            self.is_running = True
            logger.info(f"Started WebSocket connection for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error starting WebSocket connection: {str(e)}")
            return False

    def _stop_signal_generation(self):
        """Stop the WebSocket connection"""
        try:
            if hasattr(self, 'ws'):
                self.ws.close()
                self.is_running = False
                logger.info("Stopped WebSocket connection")
                return True
        except Exception as e:
            logger.error(f"Error stopping WebSocket connection: {str(e)}")
        return False

    def _handle_signal(self, signal):
        """Handle incoming WebSocket signals"""
        try:
            # Add signal to queue
            self.output_queue.put(json.dumps(signal, indent=2))
            logger.info(f"Received signal: {signal}")
        except Exception as e:
            logger.error(f"Error handling signal: {str(e)}")

    def _process_signal_output(self):
        """Process output from the WebSocket connection"""
        try:
            # Get all available output from the queue
            output = ""
            while not self.output_queue.empty():
                line = self.output_queue.get_nowait()
                output += line + "\n"
            return output if output else "No new output available"
        except Exception as e:
            logger.error(f"Error processing signal output: {str(e)}")
            return str(e)

    def _parse_signal(self, line):
        """Parse a signal line from main.py output"""
        try:
            # Extract signal details from the table format
            signal_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'TRADE',
                'signal': 'HOLD',  # Default value
                'status': 'NEW',
                'confidence': 0.0,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'reasoning': ''
            }
            
            # Read the next few lines to get signal details
            for _ in range(10):  # Read up to 10 lines to get all details
                try:
                    detail_line = self.signal_process.stdout.readline()
                    if not detail_line:
                        break
                        
                    if "Signal Type" in detail_line:
                        signal_data['signal'] = detail_line.split("│")[2].strip()
                    elif "Confidence" in detail_line:
                        signal_data['confidence'] = float(detail_line.split("│")[2].strip())
                    elif "Entry Price" in detail_line:
                        signal_data['entry_price'] = float(detail_line.split("│")[2].strip())
                    elif "Stop Loss" in detail_line:
                        signal_data['stop_loss'] = float(detail_line.split("│")[2].strip())
                    elif "Take Profit" in detail_line:
                        signal_data['take_profit'] = float(detail_line.split("│")[2].strip())
                    elif "Reasoning" in detail_line:
                        signal_data['reasoning'] = detail_line.split("│")[2].strip()
                except Exception as e:
                    logger.error(f"Error parsing signal detail: {str(e)}")
                    break
            
            self.signals.append(signal_data)
            self.last_update = time.time()
            logger.info(f"Added new signal: {signal_data}")
            
        except Exception as e:
            logger.error(f"Error parsing signal: {str(e)}")

    def _get_active_positions(self):
        """Get active positions as DataFrame"""
        positions = self.session.query(Position).filter(
            Position.status.in_([PositionStatus.OPEN, PositionStatus.PENDING])
        ).all()

        if positions:
            position_data = []
            for pos in positions:
                position_data.append({
                    "ID": pos.id,
                    "Symbol": pos.symbol,
                    "Type": pos.position_type,
                    "Status": pos.status,
                    "Entry Price": f"${pos.entry_price:.2f}" if pos.entry_price else "$0.00",
                    "Current Price": f"${pos.current_price:.2f}" if pos.current_price else "$0.00",
                    "Stop Loss": f"${pos.stop_loss:.2f}" if pos.stop_loss else "$0.00",
                    "Take Profit": f"${pos.take_profit:.2f}" if pos.take_profit else "$0.00",
                    "Size": f"{pos.size:.4f}" if pos.size else "0.0000",
                    "PnL %": f"{pos.profit_loss_percentage:+.2f}%" if pos.profit_loss_percentage else "0.00%",
                    "R:R": f"{pos.risk_reward_ratio:.2f}" if pos.risk_reward_ratio else "0.00",
                    "Confidence": f"{pos.model_confidence:.2f}" if pos.model_confidence else "0.00"
                })
            return pd.DataFrame(position_data)
        return pd.DataFrame()

    def _get_position_history(self):
        """Get position history as DataFrame"""
        positions = self.session.query(Position).filter(
            Position.status == PositionStatus.CLOSED
        ).order_by(Position.closed_at.desc()).all()

        if positions:
            position_data = []
            for pos in positions:
                position_data.append({
                    "ID": pos.id,
                    "Symbol": pos.symbol,
                    "Type": pos.position_type,
                    "Entry Price": f"${pos.entry_price:.2f}" if pos.entry_price else "$0.00",
                    "Exit Price": f"${pos.current_price:.2f}" if pos.current_price else "$0.00",
                    "Size": f"{pos.size:.4f}" if pos.size else "0.0000",
                    "PnL": f"${pos.profit_loss:+.2f}" if pos.profit_loss else "$0.00",
                    "PnL %": f"{pos.profit_loss_percentage:+.2f}%" if pos.profit_loss_percentage else "0.00%",
                    "Closed At": pos.closed_at.strftime("%Y-%m-%d %H:%M:%S") if pos.closed_at else "N/A",
                    "Reason": pos.closed_reason or "N/A"
                })
            return pd.DataFrame(position_data)
        return pd.DataFrame()

    def _get_account_stats(self):
        """Get account statistics"""
        account_balance = self.session.query(AccountBalance).first()
        if account_balance:
            return {
                "Account Balance": f"${account_balance.balance:.2f}",
                "Total P/L": f"${account_balance.total_profit_loss:+.2f}" if account_balance.total_profit_loss else "$0.00",
                "Win Rate": f"{account_balance.win_rate:.1f}%" if account_balance.win_rate else "0.0%",
                "Total Trades": str(account_balance.total_trades or 0),
                "Winning Trades": str(account_balance.winning_trades or 0),
                "Losing Trades": str(account_balance.losing_trades or 0)
            }
        return {}

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Hummingbird Trading Dashboard", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🐦 Hummingbird Trading Dashboard")
            
            # Account Statistics
            with gr.Row():
                with gr.Column():
                    account_stats = gr.JSON(label="Account Statistics")
                with gr.Column():
                    signal_status = gr.Textbox(label="Signal Generation Status", value="Stopped")
            
            # Signal Generation Controls
            with gr.Row():
                symbol = gr.Dropdown(
                    choices=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT'],
                    value='BTC/USDT',
                    label="Select Symbol"
                )
                start_btn = gr.Button("Start Signal Generation")
                stop_btn = gr.Button("Stop Signal Generation")
                refresh_btn = gr.Button("Refresh Data")
            
            # Main Content
            with gr.Tabs():
                # Active Positions Tab
                with gr.Tab("Active Positions"):
                    active_positions = gr.DataFrame(label="Active Positions")
                
                # Position History Tab
                with gr.Tab("Position History"):
                    position_history = gr.DataFrame(label="Position History")
                
                # Trading Statistics Tab
                with gr.Tab("Trading Statistics"):
                    trading_stats = gr.JSON(label="Trading Statistics")
                
                # Signals Tab
                with gr.Tab("Signals"):
                    terminal_output = gr.Textbox(
                        label="Terminal Output",
                        lines=30,
                        interactive=False
                    )
            
            # Event handlers
            def start_signals(symbol):
                if self._start_signal_generation(symbol):
                    return "Running"
                return "Failed to start"
            
            def stop_signals():
                if self._stop_signal_generation():
                    return "Stopped"
                return "Failed to stop"
            
            def update_display():
                return (
                    self._get_account_stats(),
                    self._get_active_positions(),
                    self._get_position_history(),
                    self._process_signal_output()
                )
            
            # Connect events
            start_btn.click(
                start_signals,
                inputs=[symbol],
                outputs=[signal_status]
            )
            
            stop_btn.click(
                stop_signals,
                outputs=[signal_status]
            )
            
            # Connect refresh button
            refresh_btn.click(
                update_display,
                outputs=[
                    account_stats,
                    active_positions,
                    position_history,
                    terminal_output
                ]
            )
            
            # Initial load
            interface.load(
                fn=update_display,
                outputs=[
                    account_stats,
                    active_positions,
                    position_history,
                    terminal_output
                ]
            )
        
        return interface

if __name__ == "__main__":
    dashboard = TradingDashboard()
    interface = dashboard.create_interface()
    interface.launch(share=True) 