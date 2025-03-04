# Hummingbird - LLM-Based Trading System

Hummingbird is an advanced trading system that combines technical analysis with Large Language Models (LLM) to generate high-accuracy trading signals. The system analyzes multiple timeframes, applies various technical indicators, and uses LLM to understand market patterns and generate trading signals.

## Features

- Multi-timeframe analysis (1D, 4H, 1H, 15M, 5M)
- Technical indicators (RSI, SMI, SMC Strategy)
- Order block detection
- Supply/Demand zone identification
- Real-time market data processing
- LLM-based signal generation
- Dynamic strategy adaptation
- High-accuracy trading signals with stop-loss and take-profit levels

## Project Structure

```
hummingbird/
├── data/               # Data storage
├── models/            # LLM models
├── src/
│   ├── data/         # Data fetching and processing
│   ├── technical/    # Technical analysis
│   ├── llm/          # LLM integration
│   └── utils/        # Utility functions
├── config/           # Configuration files
└── tests/            # Test files
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

1. Configure your trading parameters in `config/config.yaml`
2. Run the main trading system:
```bash
python src/main.py --symbol BTC/USDT
```

## Model Selection

The system supports various LLM models:
- Llama 2 8B (recommended for local deployment)
- Mistral 7B
- Phi-2

## License

MIT License 