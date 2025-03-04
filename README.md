# Hummingbird - LLM-Based Trading System

Hummingbird is an advanced trading system that combines technical analysis with Large Language Models (LLM) to generate high-accuracy trading signals. The system analyzes multiple timeframes, applies various technical indicators, and uses LLM to understand market patterns and generate trading signals.

## Features

- Multi-timeframe analysis (1D, 4H, 1H, 15M, 5M, 3M)
- Technical indicators (RSI, MACD, EMA)
- Smart Money Concepts (SMC) implementation:
  - Order block detection
  - Supply/Demand zone identification
  - Fair Value Gaps (FVG)
  - Liquidity sweeps
- Real-time market data processing via Binance API
- LLM-based signal generation with confidence thresholds
- Dynamic strategy adaptation
- High-accuracy trading signals with stop-loss and take-profit levels
- Beautiful CLI interface with rich progress bars and tables
- Scalping mode for short-term trading opportunities

## Project Structure

```
hummingbird/
├── data/               # Data storage and market data handling
├── models/            # LLM models and configurations
├── src/
│   ├── data/         # Data fetching and processing
│   │   └── market_data.py
│   ├── technical/    # Technical analysis
│   │   └── analysis.py
│   ├── llm/          # LLM integration
│   │   └── analyzer.py
│   └── utils/        # Utility functions
├── config/           # Configuration files
│   └── config.yaml   # Main configuration
├── tests/            # Test files
└── evn/              # Environment files
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

## Configuration

1. Configure your trading parameters in `config/config.yaml`:
   - Trading timeframes
   - Technical indicator parameters
   - LLM model settings
   - Confidence thresholds
   - Historical data settings

2. Set up your environment variables in `.env`:
   - Binance API credentials
   - LLM API keys (if using cloud models)
   - Other configuration parameters

## Usage

1. Run the main trading system:
```bash
# Standard mode with default model
python src/main.py --symbol BTC/USDT

# Standard mode with specific model
python src/main.py --symbol BTC/USDT --model gemini-pro

# Scalping mode (optimized for short-term trades)
python src/main.py --symbol BTC/USDT --scalping

# Scalping mode with specific model
python src/main.py --symbol BTC/USDT --scalping --model gemini-pro
```

Available model options:
- `gemini-pro`: Google's Gemini Pro model (requires API key)
- `local`: Local model (default if no model specified)

The system will:
- Fetch historical data for all configured timeframes
- Calculate technical indicators
- Identify market patterns (order blocks, supply/demand zones)
- Generate trading signals using LLM
- Display results in a beautiful CLI interface

## Model Selection

The system supports various LLM models:

### Local Models (Recommended)
- Local models for privacy and speed
- No API costs
- Lower latency

### Cloud Models
#### Gemini Pro Setup
1. Get your API key from Google AI Studio (https://makersuite.google.com/app/apikey)
2. Add to your `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```
3. Configure in `config/config.yaml`:
```yaml
llm:
  default_model: "gemini-pro"
  api_key: ${GEMINI_API_KEY}
  temperature: 0.7
  max_tokens: 1000
  confidence_threshold: 0.8
```

Gemini Pro features:
- Advanced reasoning capabilities
- Real-time market analysis
- Pattern recognition
- Risk assessment
- Trade execution recommendations

## License

MIT License 