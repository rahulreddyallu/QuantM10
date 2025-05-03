```markdown name=README.md
# Master Trading Signal Bot

![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

An advanced algorithmic trading signal generator with comprehensive technical analysis capabilities, pattern recognition, and systematic backtesting. The system analyzes market data using multiple indicators, candlestick patterns, and chart formations to generate high-confidence trading signals.

## Features

- **Comprehensive Technical Analysis**: 30+ technical indicators including Moving Averages, MACD, RSI, Stochastic, Bollinger Bands, Supertrend, etc.
- **Pattern Recognition**: Detection of 15+ candlestick patterns and 10+ chart patterns
- **Advanced Signal Generation**: Multi-factor signal strength calculation with confidence metrics
- **Risk Management**: Automated stop-loss and take-profit level calculation
- **Backtesting System**: Rigorous strategy validation with Monte Carlo simulations
- **Walk-Forward Optimization**: Parameter optimization with anti-overfitting measures
- **Alert System**: Telegram integration for real-time trading signals
- **Modular Architecture**: Easily extendable for custom indicators and strategies

## System Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Market Data API  │────▶│ Technical Analysis │────▶│   Signal Engine   │
└──────────────────┘     └───────────────────┘     └─────────┬─────────┘
                                                             │
┌──────────────────┐     ┌───────────────────┐     ┌─────────▼─────────┐
│ Telegram Alerts  │◀────│    Risk Manager   │◀────│ Trading Signals   │
└──────────────────┘     └───────────────────┘     └───────────────────┘
        ▲                                                    │
        │             ┌───────────────────┐      ┌───────────▼─────────┐
        └─────────────│ Notification Mgr  │◀─────│ Backtesting Engine  │
                      └───────────────────┘      └───────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/master-trading-signal-bot.git
   cd master-trading-signal-bot
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

4. Set up environment variables:
   ```bash
   # For Linux/Mac
   export UPSTOX_API_KEY="your_upstox_api_key"
   export UPSTOX_API_SECRET="your_upstox_api_secret"
   export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
   export TELEGRAM_CHAT_ID="your_telegram_chat_id"
   
   # For Windows
   set UPSTOX_API_KEY=your_upstox_api_key
   set UPSTOX_API_SECRET=your_upstox_api_secret
   set TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   set TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

## Configuration

The system uses a `config.py` file for all configuration parameters. Key settings include:

- **API Credentials**: Upstox API keys and Telegram bot credentials
- **Scheduling**: Define when the bot runs during market hours
- **Stock List**: Instruments to analyze (using ISIN-based instrument keys)
- **Technical Indicators**: Customize parameters for each indicator
- **Pattern Detection**: Configure sensitivity thresholds for pattern recognition
- **Signal Parameters**: Adjust weights and thresholds for signal generation
- **Backtesting**: Define historical testing parameters and optimization settings

Reference the `config.py` file comments for detailed parameter descriptions.

## Usage

### Running Modes

The system can be run in several modes:

1. **Scheduled Mode** (default):
   ```bash
   python main.py
   ```
   Runs on schedule based on configured market times.

2. **Adhoc Mode**:
   ```bash
   python main.py adhoc
   ```
   Executes one analysis cycle and exits.

3. **Backtest Mode**:
   ```bash
   python main.py backtest NSE_EQ|INE009A01021 --days 365 --output results.json
   ```
   Runs backtest for a specific instrument, optionally saving results to a file.

4. **Parameter Optimization**:
   ```bash
   python main.py optimize NSE_EQ|INE009A01021 --output optimized.json
   ```
   Executes walk-forward optimization to find ideal parameters.

5. **Multi-Stock Backtest**:
   ```bash
   python main.py multi --days 365 --output all_results.json
   ```
   Backtests all configured stocks and generates a comparative report.

## Technical Components

### Technical Indicators

The system implements a comprehensive suite of technical indicators:

| Category | Indicators |
|----------|------------|
| Trend | Moving Averages (SMA, EMA, DEMA, TEMA, WMA, HMA), Supertrend, Parabolic SAR, ADX, Aroon, Alligator |
| Momentum | MACD, RSI, Stochastic, Stochastic RSI, ROC, Williams %R, Ultimate Oscillator |
| Volatility | Bollinger Bands, ATR, ATR Bands |
| Volume | OBV, VWAP, Chaikin Money Flow, Volume Profile |
| Support/Resistance | Fibonacci Retracements, CPR (Central Pivot Range), Support/Resistance Levels |

Each indicator is fully configurable through the `config.py` file.

### Candlestick Patterns

The system detects the following candlestick patterns:

- **Single Candle**: Marubozu, Doji, Spinning Top, Paper Umbrella, Hammer, Hanging Man, Shooting Star
- **Multi-Candle**: Engulfing, Harami, Piercing Pattern, Dark Cloud Cover, Morning Star, Evening Star, Three White Soldiers, Three Black Crows

Pattern detection uses precise mathematical criteria with configurable thresholds.

### Chart Patterns

The system implements detection algorithms for these chart patterns:

- **Reversal Patterns**: Head and Shoulders, Inverse Head and Shoulders, Double Top/Bottom, Triple Top/Bottom, Rounding Top/Bottom
- **Continuation Patterns**: Cup and Handle, Wedges, Rectangles, Flags, Pennants

### Backtesting Engine

The backtesting system features:

- Historical performance evaluation with detailed metrics
- Position sizing and risk management simulation
- Monte Carlo simulations for robustness testing
- Walk-forward optimization for parameter tuning
- Equity curve and drawdown analysis
- Detailed performance reporting

## API Integration

### Upstox API

The system integrates with Upstox for market data:

```python
# Example: Fetching historical data
async def fetch_historical_data(self, instrument_key, interval='1D', days=250):
    """
    Fetch historical data for a given instrument
    
    Args:
        instrument_key: Instrument identifier
        interval: Candle interval
        days: Number of days to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    # Implementation details in compute.py
```

### Telegram Integration

Real-time alerts are sent via Telegram:

```python
# Example: Sending a signal alert
async def send_signal_alert(self, result):
    """
    Send a signal alert via Telegram
    
    Args:
        result: Signal result dictionary
        
    Returns:
        True if alert was sent successfully, False otherwise
    """
    # Implementation details in compute.py
```

## Extending the System

### Adding New Indicators

To add a new technical indicator:

1. Create a new method in the `TechnicalIndicators` class in `compute.py`:
   ```python
   def calculate_new_indicator(self):
       """Calculate your new indicator"""
       # Implementation
       self.df['new_indicator'] = # calculation
       
       # Generate signals
       # ...
       
       self.indicators['new_indicator'] = {
           'signal': current_signal,
           'signal_strength': signal_strength,
           'values': {
               # indicator values
           }
       }
   ```

2. Add the method call in the `calculate_all()` method.

3. Add configuration parameters to `config.py`.

### Custom Strategies

Create custom backtesting strategies by defining a strategy function:

```python
def my_strategy(data, **params):
    """
    Custom trading strategy
    
    Args:
        data: DataFrame with price data
        params: Strategy parameters
        
    Returns:
        Signal value (1=buy, -1=sell, 0=neutral)
    """
    # Your strategy logic here
    return signal # 1, -1, or 0
```

Run with your custom strategy:

```bash
python main.py backtest NSE_EQ|INE009A01021 --strategy my_strategy
```

## Performance Considerations

- **Data Caching**: Historical data is cached to minimize API calls
- **Async Processing**: Asynchronous operations for improved throughput
- **Error Handling**: Comprehensive try-except blocks for robustness
- **Resource Management**: Proper cleanup of resources like API connections

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API Connection Errors | Check API credentials and network connectivity |
| Missing Dependencies | Run `pip install -r requirements.txt` to install all required packages |
| Insufficient Historical Data | Ensure selected stocks have adequate trading history |
| No Signals Generated | Check minimum signal strength threshold in configuration |
| Backtest Errors | Verify data completeness and parameter validity |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Rahul Reddy Allu**
- Version 4.0.0
- Last Updated: 2025-05-03

---

© 2025 Master Trading Signal Bot
```
