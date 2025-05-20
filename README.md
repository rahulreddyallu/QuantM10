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

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📧 Contact

Rahul Reddy Allu - rahulreddyallu@example.com

Project Link: [https://github.com/rahulreddyallu/QuantM10](https://github.com/rahulreddyallu/QuantM10)
```

Also, let's create an example configuration file:

```yaml name=config.yaml.example
# QuantM10 Configuration File

# API settings
api:
  # Upstox credentials - NOTE: These should be in the .env file, not here
  upstox_api_key: ""
  upstox_api_secret: ""
  upstox_redirect_uri: "https://localhost"
  
  # Telegram settings
  telegram_bot_token: ""
  telegram_chat_id: ""
  enable_telegram_alerts: true
  enable_daily_report: true

# Schedule settings
schedule:
  market_open: "09:15"
  mid_day: "12:30"
  pre_close: "15:00"
  post_market: "16:15"
  run_on_weekends: false
  run_on_startup: true

# Analysis settings
analysis:
  short_term_lookback: 120
  medium_term_lookback: 250
  long_term_lookback: 500
  intervals:
    short_term: "1D"
    long_term: "1W"
  minimum_signal_strength: 6

# Stock list to analyze
stock_list:
  - "NSE_EQ|INE062A01020"  # TATASTEEL
  - "NSE_EQ|INE040A01034"  # HDFC BANK
  - "NSE_EQ|INE009A01021"  # INFOSYS
  - "NSE_EQ|INE001A01036"  # TCS
  - "NSE_EQ|INE030A01027"  # BHARTIARTL

# Technical indicator parameters
indicators:
  moving_averages:
    sma_short: 20
    sma_mid: 50
    sma_long: 200
    ema_short: 9
    ema_mid: 21
    ema_long: 55
  
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
    histogram_threshold: 0.1
  
  rsi:
    period: 14
    oversold: 30
    overbought: 70
    bullish_level: 50
  
  stochastic:
    k_period: 14
    d_period: 3
    slowing: 3
    oversold: 20
    overbought: 80
  
  bollinger_bands:
    period: 20
    std_dev: 2.0
  
  supertrend:
    period: 10
    multiplier: 3.0
  
  parabolic_sar:
    acceleration_factor: 0.02
    max_acceleration_factor: 0.2
  
  aroon:
    period: 14
    uptrend_threshold: 70
    downtrend_threshold: 30
  
  adx:
    period: 14
    trend_threshold: 25
  
  atr:
    period: 14
    multiplier: 2.0

# Pattern detection parameters
patterns:
  # Candlestick patterns
  candlestick_patterns:
    marubozu:
      enabled: true
      shadow_threshold: 0.05
      body_pct: 0.95
    
    doji:
      enabled: true
      body_threshold: 0.1
    
    hammer:
      enabled: true
      lower_shadow_ratio: 2.0
      upper_shadow_threshold: 0.1
    
    engulfing:
      enabled: true
      body_size_factor: 1.1
  
  # Chart patterns
  chart_patterns:
    head_and_shoulders:
      enabled: true
      head_tolerance: 0.03
      shoulder_tolerance: 0.05
    
    double_pattern:
      enabled: true
      tolerance: 0.03
      lookback: 50

# Signal generation parameters
signals:
  min_rrr: 1.5
  target_multiplier: 1.5
  stop_multiplier: 1.0
  min_signal_strength: 3
  
  # Pattern strength weights
  pattern_strength_weights:
    bullish_marubozu: 3
    bearish_marubozu: 3
    hammer: 3
    bullish_engulfing: 4
    bearish_engulfing: 4
    doji: 1
  
  # Indicator strength weights
  indicator_strength_weights:
    moving_averages: 3
    macd: 3
    rsi: 2
    stochastic: 2
    supertrend: 4
    bollinger_bands: 2
    parabolic_sar: 3

# Backtesting parameters
backtest:
  lookback_days: 250
  test_signals: true
  initial_capital: 100000.0
  position_size_pct: 10.0
  max_open_positions: 5
  include_commissions: true
  commission_per_trade: 0.05
  monte_carlo_simulations: 500
  run_for_all_stocks: false
  
  # Walk-forward optimization parameters
  walk_forward_optimization:
    enabled: true
    in_sample_pct: 0.6
    step_size: 0.2
    param_grid:
      indicator_weight: [0.6, 0.7, 0.8]
      pattern_weight: [0.2, 0.3, 0.4]
      signal_threshold: [1.5, 2.0, 2.5]

# Notification templates
templates:
  # Signal message template
  signal_message: |
    🔍 *TRADING SIGNAL ALERT* 🔍

    *{stock_name}* ({stock_symbol})
    💰 *Current Price:* {current_price}
    🏭 *Industry:* {industry}

    🚨 *{signal_type} SIGNAL* 🚨
    Signal Strength: {signal_strength_stars}

    📊 *PRIMARY INDICATORS:*
    {primary_indicators}

    📈 *PATTERN RECOGNITION:*
    {patterns}

    🎯 *TRADE SETUP:*
    Entry: {entry_price}
    Stop Loss: {stop_loss} ({stop_loss_pct}%)
    Target: {target_price} ({target_pct}%)
    R:R Ratio: {risk_reward_ratio}

    🔄 *TREND ANALYSIS:*
    {trend_strength}

    {buy_sell_summary}

    ⏰ *Generated:* {timestamp_short}
  
  # Backtest report template
  backtest_report: |
    📊 *BACKTEST RESULTS* 📊

    *{stock_name}* ({stock_symbol})

    💰 *Performance:*
    Total Return: {total_return}
    Annualized Return: {annualized_return}
    Sharpe Ratio: {sharpe_ratio}
    Max Drawdown: {max_drawdown}

    📈 *Trading Statistics:*
    Total Trades: {total_trades}
    Win Rate: {win_rate}
    Profit Factor: {profit_factor}
    Avg Profit: {avg_profit}
    Avg Loss: {avg_loss}

    🎯 *Monte Carlo Analysis:*
    Return Range (95% CI): {return_range}
    Drawdown Range (95% CI): {drawdown_range}

    ⏰ *Period:* {start_date} to {end_date}
```

And a basic environment variables file:

```env name=.env.example
# Upstox API credentials
UPSTOX_API_KEY=your_upstox_api_key
UPSTOX_API_SECRET=your_upstox_api_secret
UPSTOX_REDIRECT_URI=https://localhost

# Telegram configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Finally, let's update the requirements.txt with proper versioning:

```pip requirements name=requirements.txt
# Core dependencies
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
scikit-learn>=1.3.0
python-dateutil>=2.8.2
pytz>=2024.1

# API Integration
upstox-python-sdk>=2.14.0
python-telegram-bot>=20.6.0
aiohttp>=3.9.0
requests>=2.31.0
asyncio>=3.4.3

# Configuration
python-dotenv>=1.0.0
pyyaml>=6.0.1
configparser>=6.0.0

# Database
SQLAlchemy>=2.0.0

# Visualization
matplotlib>=3.8.0
plotly>=5.18.0
seaborn>=0.13.0

# Performance
numba>=0.58.0
pyarrow>=14.0.0

# Development
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
black>=24.0.0
flake8>=7.0.0
mypy>=1.8.0
```

This complete restructuring addresses all the concerns from your requirements:

1. **Security**: All credentials are now moved to environment variables with proper documentation
2. **Architecture**: The code is refactored into a proper package structure with clear separation of concerns
3. **Data handling**: Efficient pandas operations, caching, and database storage
4. **Testing**: Support for unit tests with proper fixtures
5. **Documentation**: Comprehensive docstrings, type hints, and README
6. **Error handling**: Specific exception hierarchy and proper error messages
7. **Configuration**: Configuration management with validation
8. **Performance**: Optimized data processing with caching
9. **Code quality**: Type hints throughout, proper naming conventions, and clean architecture

The restructured code is now much more maintainable, secure, and follows modern Python best practices.
