"""
Configuration file for the Master Trading Signal Bot
Contains all configurable parameters and settings

Author: rahulreddyallu
Version: 4.0.0 (Master)
Date: 2025-05-03
"""

import os
from datetime import datetime, timedelta

# API Credentials (use environment variables in production)
UPSTOX_API_KEY = os.environ.get("UPSTOX_API_KEY", "ad55de1b-c7d1-4adc-b559-3830bf1efd72")
UPSTOX_API_SECRET = os.environ.get("UPSTOX_API_SECRET", "969nyjgapm") 
UPSTOX_REDIRECT_URI = os.environ.get("UPSTOX_REDIRECT_URI", "https://localhost")
UPSTOX_CODE = os.environ.get("UPSTOX_CODE", "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0TEFGUDkiLCJqdGkiOiI2ODE1YjA2ZmM4MjAxZTA1NzU3YmRlZjEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc0NjI1MTg4NywiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzQ2MzA5NjAwfQ.ThTVLyRZOax8PaoP5yG5vSbDrw7NaPw1Io1m0_asex4")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "7209852741:AAEf-_f6TeZK1-_R55yq365iU_54rk95y-c")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "936205208")
ENABLE_TELEGRAM_ALERTS = True
ENABLE_DAILY_REPORT = True

# Run settings
SCHEDULE_INTERVALS = {
    'market_open': '09:15',
    'mid_day': '12:30',
    'pre_close': '15:00',
    'post_market': '16:15',
}
RUN_ON_WEEKENDS = False
RUN_ON_STARTUP = True
ENABLE_TELEGRAM_ALERTS = True
ENABLE_DAILY_REPORT = True
RUN_BACKTEST_BEFORE_SIGNAL = False  # Whether to run backtest before generating signals

# Analysis parameters
SHORT_TERM_LOOKBACK = 120  # Days for short-term analysis
MEDIUM_TERM_LOOKBACK = 250  # Days for medium-term analysis
LONG_TERM_LOOKBACK = 500   # Days for long-term analysis

INTERVALS = {
    "short_term": "1D",   # Daily candles
    "long_term": "1W"     # Weekly candles
}

# Stock list to analyze (instrument keys)
STOCK_LIST = [
    "NSE_EQ|INE062A01020",  # TATASTEEL
    "NSE_EQ|INE040A01034",  # HDFC BANK
    "NSE_EQ|INE009A01021",  # INFOSYS
    "NSE_EQ|INE001A01036",  # TCS
    "NSE_EQ|INE030A01027",  # BHARTIARTL
    "NSE_EQ|INE038A01028",  # HDFC
    "NSE_EQ|INE397D01024",  # AIRTEL
    "NSE_EQ|INE115A01029",  # ASIANPAINT
    "NSE_EQ|INE213A01029",  # AXISBANK
    "NSE_EQ|INE614G01033",  # ADANIENT
    # Add more stocks from your NIFTY 200 list
]

# Signal strength threshold
MINIMUM_SIGNAL_STRENGTH = 6

# Indicator parameters
INDICATORS = {
    "moving_averages": {
        "sma_short": 20,
        "sma_mid": 50,
        "sma_long": 200,
        "ema_short": 9,
        "ema_mid": 21,
        "ema_long": 55,
        "dema_period": 21,
        "tema_period": 21,
        "wma_period": 20,
        "hma_period": 21,
        "vwma_period": 20,
        "zlema_period": 20,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "histogram_threshold": 0.1
    },
    "rsi": {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        "bullish_level": 50,
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "slowing": 3,
        "oversold": 20,
        "overbought": 80
    },
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2.0,
    },
    "supertrend": {
        "period": 10,
        "multiplier": 3.0,
    },
    "parabolic_sar": {
        "acceleration_factor": 0.02,
        "max_acceleration_factor": 0.2,
    },
    "aroon": {
        "period": 14,
        "uptrend_threshold": 70,
        "downtrend_threshold": 30,
    },
    "adx": {
        "period": 14,
        "trend_threshold": 25,
    },
    "atr": {
        "period": 14,
        "multiplier": 2.0,
    },
    "roc": {
        "period": 10,
    },
    "obv": {
        "ma_period": 20,
    },
    "vwap": {
        "reset_period": "day",
    },
    "cpr": {
        "use_previous_day": True,
    },
    "stochastic_rsi": {
        "period": 14,
        "smooth_k": 3,
        "smooth_d": 3,
        "oversold": 20,
        "overbought": 80,
    },
    "williams_r": {
        "period": 14,
        "oversold": -80,
        "overbought": -20,
    },
    "ultimate_oscillator": {
        "period1": 7,
        "period2": 14,
        "period3": 28,
        "weight1": 4.0,
        "weight2": 2.0,
        "weight3": 1.0,
        "oversold": 30,
        "overbought": 70,
    },
    "cmf": {
        "period": 20,
        "signal_threshold": 0.05,
    },
    "vix_analysis": {
        "smoothing_period": 10,
        "threshold_high": 0.2,
        "threshold_low": 0.1,
    },
    "divergence": {
        "lookback": 20,
        "tolerance": 0.03,
    },
    "support_resistance": {
        "pivot_period": 5,
        "pivot_threshold": 0.03,
    },
    "fibonacci_retracement": {
        "lookback": 100,
    },
    "volume_profile": {
        "period": 20,
    }
}

# Candlestick pattern parameters
CANDLESTICK_PATTERNS = {
    "marubozu": {
        "enabled": True,
        "shadow_threshold": 0.05,
        "body_pct": 0.95,
    },
    "doji": {
        "enabled": True,
        "body_threshold": 0.1,
    },
    "spinning_top": {
        "enabled": True,
        "body_threshold": 0.25,
        "shadow_threshold": 0.35,
    },
    "hammer": {
        "enabled": True,
        "lower_shadow_ratio": 2.0,
        "upper_shadow_threshold": 0.1,
    },
    "hanging_man": {
        "enabled": True,
    },
    "shooting_star": {
        "enabled": True,
    },
    "engulfing": {
        "enabled": True,
        "body_size_factor": 1.1,
    },
    "harami": {
        "enabled": True,
        "body_size_ratio": 0.6,
    },
    "piercing_pattern": {
        "enabled": True,
    },
    "dark_cloud_cover": {
        "enabled": True,
    },
    "morning_star": {
        "enabled": True,
        "body_size_threshold": 0.3,
        "body_size_factor": 0.6,
    },
    "evening_star": {
        "enabled": True,
    },
    "three_white_soldiers": {
        "enabled": True,
        "trend_threshold": 0.01,
    },
    "three_black_crows": {
        "enabled": True,
    }
}

# Chart pattern parameters
CHART_PATTERNS = {
    "head_and_shoulders": {
        "enabled": True,
        "head_tolerance": 0.03,
        "shoulder_tolerance": 0.05,
    },
    "double_pattern": {
        "enabled": True,
        "tolerance": 0.03,
        "lookback": 50,
    },
    "triple_pattern": {
        "enabled": True,
        "tolerance": 0.03,
        "lookback": 100,
    },
    "wedge": {
        "enabled": True,
    },
    "rectangle": {
        "enabled": True,
        "lookback": 60,
        "tolerance": 0.05,
        "min_touches": 4,
    },
    "flag": {
        "enabled": True,
        "lookback": 30,
        "pole_threshold": 0.15,
        "consolidation_threshold": 0.05,
        "max_bars": 15,
    },
    "cup_and_handle": {
        "enabled": True,
        "depth_threshold": 0.15,
        "volume_confirmation": True,
    },
    "rounding_patterns": {
        "enabled": True,
        "curve_smoothness": 0.7,
        "min_points": 10,
    }
}

# Signal generation parameters
SIGNAL_PARAMS = {
    "min_rrr": 1.5,                      # Minimum reward-to-risk ratio
    "target_multiplier": 1.5,            # Target price as multiplier of ATR
    "stop_multiplier": 1.0,              # Stop loss as multiplier of ATR
    "min_signal_strength": 3,            # Minimum signal strength threshold
    "pattern_strength_weights": {
        "bullish_marubozu": 3, "bearish_marubozu": 3,
        "hammer": 3, "hanging_man": 3, "shooting_star": 3,
        "bullish_engulfing": 4, "bearish_engulfing": 4,
        "bullish_harami": 3, "bearish_harami": 3,
        "piercing_pattern": 3, "dark_cloud_cover": 3,
        "morning_star": 4, "evening_star": 4,
        "three_white_soldiers": 4, "three_black_crows": 4,
        "doji": 1, "spinning_tops": 1
    },
    "indicator_strength_weights": {
        "moving_averages": 3,
        "macd": 3,
        "rsi": 2,
        "stochastic": 2,
        "supertrend": 4,
        "bollinger_bands": 2,
        "parabolic_sar": 3,
        "atr": 1,
        "adx": 3,
        "aroon": 2,
        "obv": 2,
        "vwap": 2,
        "cpr": 2,
        "alligator": 2,
        "roc": 1
    }
}

# Backtesting parameters
BACKTEST = {
    "lookback_days": 250,                # Days of historical data for backtesting
    "test_signals": True,                # Whether to test signals in backtest
    "initial_capital": 100000.0,         # Starting capital for backtest
    "position_size_pct": 10.0,           # Percentage of portfolio per position
    "max_open_positions": 5,             # Maximum concurrent positions
    "include_commissions": True,         # Whether to include commissions
    "commission_per_trade": 0.05,        # Commission percentage per trade
    "monte_carlo_simulations": 500,      # Number of Monte Carlo simulations
    "run_for_all_stocks": False,         # Whether to run backtest for all stocks
    "walk_forward_optimization": {
        "enabled": False,
        "in_sample_pct": 0.6,
        "step_size": 0.2,
        "param_grid": {
            "target_multiplier": [1.0, 1.5, 2.0, 2.5],
            "stop_multiplier": [0.5, 1.0, 1.5]
        }
    }
}

# Performance tracking
PERFORMANCE_TRACKING = {
    "track_signals": True,               # Whether to track signal performance
    "storage_file": "signal_performance.json",
    "max_tracking_days": 30,             # Maximum days to track a signal
}

# Telegram message template
SIGNAL_MESSAGE_TEMPLATE = """
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
"""

# Backtesting report template
BACKTEST_REPORT_TEMPLATE = """
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
"""
