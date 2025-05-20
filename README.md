# QuantM10: Advanced Algorithmic Trading System

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

An advanced algorithmic trading system with comprehensive technical analysis capabilities, pattern recognition, and systematic backtesting. QuantM10 analyzes market data using multiple indicators, detects chart patterns, generates trading signals, and provides detailed performance metrics through rigorous backtesting.

**Author**: Rahul Reddy Allu  
**Last Updated**: 2025-05-20

## 🌟 Features

- **Secure by Design**: Environment-based configuration with no hardcoded credentials
- **Modular Architecture**: Cleanly separated components with well-defined interfaces
- **Comprehensive Analysis**: 30+ technical indicators and 15+ pattern detection algorithms
- **Advanced Signal Generation**: Multi-factor signal calculation with confidence metrics
- **Robust Backtesting**: Historical performance evaluation with Monte Carlo simulations
- **Parameter Optimization**: Walk-forward testing with anti-overfitting measures
- **Real-time Alerts**: Telegram integration for trading signals and reports
- **Database Integration**: Performance tracking and signal history storage
- **Extensive Documentation**: Type hints, docstrings, and usage examples

## 📋 System Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Market Data API  │────▶│ Technical Analysis│────▶│   Signal Engine   │
└──────────────────┘     └───────────────────┘     └─────────┬─────────┘
                                                             │
┌──────────────────┐     ┌───────────────────┐     ┌─────────▼─────────┐
│ Telegram Alerts  │◀────│    Risk Manager   │◀────│ Trading Signals   │
└──────────────────┘     └───────────────────┘     └───────────────────┘
        ▲                                                    │
        │             ┌───────────────────┐      ┌───────────▼─────────┐
        └─────────────│ Notification Mgr  │◀─────│ Backtesting Engine  │
                      └───────────────────┘      └─────────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulreddyallu/QuantM10.git
   cd QuantM10
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an environment file:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` to add your API credentials:
   ```
   UPSTOX_API_KEY=your_api_key
   UPSTOX_API_SECRET=your_api_secret
   UPSTOX_REDIRECT_URI=https://localhost
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## ⚙️ Configuration

The system uses a YAML configuration file for all settings. A sample configuration is provided in `config.yaml.example`.

To use a custom configuration:

```bash
cp config.yaml.example config.yaml
```

Then edit `config.yaml` to customize:

- **API Settings**: Connection parameters for data providers
- **Schedule Settings**: Timing for automated runs
- **Analysis Parameters**: Lookback periods and intervals
- **Technical Indicators**: Parameters for each indicator
- **Pattern Detection**: Sensitivity thresholds
- **Signal Generation**: Weights and thresholds
- **Backtesting**: Parameter optimization settings
- **Stock List**: Instruments to analyze

## 🚀 Usage

### Command Line Interface

QuantM10 provides a versatile command-line interface:

#### Run Once

```bash
# Run analysis for all configured stocks
python -m quantm10.main run

# Run analysis for a specific stock
python -m quantm10.main run --stock NSE_EQ|INE009A01021

# Use a custom configuration file
python -m quantm10.main run --config my_config.yaml
```

#### Scheduled Mode

```bash
# Run on schedule based on configured times
python -m quantm10.main scheduled

# Use a custom configuration file
python -m quantm10.main scheduled --config my_config.yaml
```

#### Backtesting

```bash
# Run backtest for a specific stock
python -m quantm10.main backtest NSE_EQ|INE009A01021 --days 365

# Save results to file
python -m quantm10.main backtest NSE_EQ|INE009A01021 --output results.json
```

#### Multi-Stock Backtest

```bash
# Run backtest for all configured stocks
python -m quantm10.main multi --days 365

# Save results to file
python -m quantm10.main multi --output all_results.json
```

#### Parameter Optimization

```bash
# Run walk-forward optimization
python -m quantm10.main optimize NSE_EQ|INE009A01021

# Save optimization results
python -m quantm10.main optimize NSE_EQ|INE009A01021 --output optimized.json
```

### Using as a Library

QuantM10 can also be imported and used as a library in your Python code:

```python
import asyncio
from quantm10.app.trading_bot import TradingBot
from quantm10.config import ConfigManager

async def generate_signals():
    # Load configuration
    config = ConfigManager("config.yaml").get_config()
    
    # Initialize bot
    bot = TradingBot(config)
    await bot.initialize()
    
    # Generate signal for a specific stock
    signal = await bot.generate_signal("NSE_EQ|INE009A01021")
    print(f"Signal: {signal['signal_type']} (Strength: {signal['signal_strength']})")
    
    # Run backtest
    backtest = await bot.run_backtest("NSE_EQ|INE009A01021", days=365)
    print(f"Backtest Return: {backtest['total_return']}")

# Run the async function
asyncio.run(generate_signals())
```

## 📊 Technical Components

### Technical Indicators

The system implements a comprehensive suite of technical indicators:

| Category | Indicators |
|----------|------------|
| Trend | Moving Averages (SMA, EMA, DEMA, TEMA, WMA, HMA), Supertrend, Parabolic SAR, ADX, Aroon |
| Momentum | MACD, RSI, Stochastic, Stochastic RSI, ROC, Williams %R, Ultimate Oscillator |
| Volatility | Bollinger Bands, ATR, ATR Bands |
| Volume | OBV, VWAP, Chaikin Money Flow |
| Support/Resistance | Fibonacci Retracements, CPR (Central Pivot Range), Support/Resistance Levels |

### Pattern Detection

The system detects numerous candlestick and chart patterns:

- **Candlestick Patterns**: Marubozu, Doji, Hammer, Engulfing, Harami, Morning/Evening Star, etc.
- **Chart Patterns**: Head and Shoulders, Double Top/Bottom, Wedges, Rectangles, Cup and Handle, etc.

### Signal Generation

The signal generation engine combines:

- Technical indicator signals
- Pattern detection signals
- Market context
- Risk parameters

Each signal includes:
- Signal type (BUY, SELL, NEUTRAL)
- Signal strength (1-5)
- Confidence score
- Entry, stop-loss and target prices
- Risk/reward ratio
- Component breakdown

### Backtesting Engine

The comprehensive backtesting system includes:

- Position sizing and risk management
- Commission handling
- Equity curve tracking
- Performance metrics (return, Sharpe ratio, drawdown)
- Monte Carlo simulations
- Walk-forward optimization

## 🔍 Example Signals

```
🔍 TRADING SIGNAL ALERT 🔍

INFOSYS (NSE_EQ|INE009A01021)
💰 Current Price: 1850.75
🏭 Industry: Information Technology

🚨 STRONG BUY SIGNAL 🚨
Signal Strength: ⭐⭐⭐⭐

📊 PRIMARY INDICATORS:
Moving Averages: BUY
RSI: BUY
MACD: STRONG BUY

📈 PATTERN RECOGNITION:
• bullish_engulfing
• double_bottom

🎯 TRADE SETUP:
Entry: 1850.75
Stop Loss: 1795.50 (2.98%)
Target: 1950.25 (5.38%)
R:R Ratio: 1.80

🔄 TREND ANALYSIS:
Strong Uptrend with confirmation from multiple timeframes

👉 Consider buying at current market price with stop loss set at specified level.

⏰ Generated: 2025-05-20 15:30
```

## 📈 Database and Caching

QuantM10 includes a SQLite database for storing:

- Signal history with performance tracking
- Backtest results
- Optimization results
- Instrument details

The system also implements intelligent caching to:
- Minimize API calls
- Speed up repeated calculations
- Store frequently accessed data

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=quantm10
```

## 📄 License

This project is licensed - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📧 Contact

Rahul Reddy Allu - rahulallu.career@gmail.com
