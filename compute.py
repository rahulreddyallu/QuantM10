"""
Core logic for Master Trading Signal Bot
Contains comprehensive technical analysis, pattern recognition, and signal generation

Author: rahulreddyallu
Version: 4.0.0 (Master)
Date: 2025-05-03
"""

import asyncio
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Optional

# Attempt to import required libraries
try:
    from telegram import Bot
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
except ImportError:
    logging.critical("Required dependency 'python-telegram-bot' not found. Please install it using: pip install python-telegram-bot")
    sys.exit(1)

try:
    from upstox_client.api_client import ApiClient
    from upstox_client.api.login_api import LoginApi
    from upstox_client.api.market_quote_api import MarketQuoteApi
    from upstox_client.api.history_api import HistoryApi
    from upstox_client.models.ohlc import Ohlc as OHLCInterval
except ImportError:
    logging.critical("Required Upstox client dependencies not found. Please install them.")
    sys.exit(1)

# ===============================================================
# Custom Exceptions
# ===============================================================
class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass

class DataFetchError(TradingBotError):
    """Exception raised for errors when fetching data"""
    def __init__(self, symbol, message="Failed to fetch data", cause=None):
        self.symbol = symbol
        self.cause = cause
        self.message = f"{message} for {symbol}: {cause}"
        super().__init__(self.message)

class PatternDetectionError(TradingBotError):
    """Exception raised for errors in pattern detection"""
    pass

class InvalidConfigurationError(TradingBotError):
    """Exception raised for invalid configuration"""
    pass

class APIConnectionError(TradingBotError):
    """Exception raised for API connection issues"""
    pass

class EmptyDataError(DataFetchError):
    """Exception raised when fetched data is empty"""
    def __init__(self, symbol, message="No data returned"):
        super().__init__(symbol, message)

# ===============================================================
# Parameter Configuration
# ===============================================================
class TradingParameters:
    """Centralized configuration for all trading parameters and thresholds"""
    
    def __init__(self, config=None):
        """
        Initialize with parameters from config or use defaults
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        
        # Pattern detection thresholds
        self.PATTERN_THRESHOLDS = {
            # Doji thresholds
            "doji_body_threshold": self.config.get("doji_body_threshold", 0.1),
            
            # Marubozu thresholds
            "marubozu_shadow_threshold": self.config.get("marubozu_shadow_threshold", 0.05),
            "marubozu_body_pct": self.config.get("marubozu_body_pct", 0.95),
            
            # Spinning top thresholds
            "spinning_top_body_threshold": self.config.get("spinning_top_body_threshold", 0.25),
            "spinning_top_shadow_threshold": self.config.get("spinning_top_shadow_threshold", 0.35),
            
            # Hammer & shooting star thresholds
            "hammer_lower_shadow_ratio": self.config.get("hammer_lower_shadow_ratio", 2.0),
            "hammer_upper_shadow_threshold": self.config.get("hammer_upper_shadow_threshold", 0.1),
            
            # Paper umbrella thresholds
            "umbrella_lower_shadow_ratio": self.config.get("umbrella_lower_shadow_ratio", 2.0),
            "umbrella_upper_shadow_threshold": self.config.get("umbrella_upper_shadow_threshold", 0.1),
            
            # Engulfing pattern tolerance
            "engulfing_body_size_factor": self.config.get("engulfing_body_size_factor", 1.1),
            
            # Harami pattern thresholds  
            "harami_body_size_ratio": self.config.get("harami_body_size_ratio", 0.6),
            
            # Star pattern thresholds
            "star_body_size_threshold": self.config.get("star_body_size_threshold", 0.3),
            "star_body_size_factor": self.config.get("star_body_size_factor", 0.6),
            
            # Three soldiers/crows thresholds
            "three_candle_trend_threshold": self.config.get("three_candle_trend_threshold", 0.01),
        }
        
        # Technical indicator parameters
        self.INDICATOR_PARAMS = {
            # RSI parameters
            "rsi_period": self.config.get("rsi_period", 14),
            "rsi_oversold": self.config.get("rsi_oversold", 30),
            "rsi_overbought": self.config.get("rsi_overbought", 70),
            
            # MACD parameters
            "macd_fast": self.config.get("macd_fast", 12),
            "macd_slow": self.config.get("macd_slow", 26),
            "macd_signal": self.config.get("macd_signal", 9),
            
            # Bollinger Bands parameters
            "bb_period": self.config.get("bb_period", 20),
            "bb_std_dev": self.config.get("bb_std_dev", 2),
            
            # ATR parameters
            "atr_period": self.config.get("atr_period", 14),
            "atr_multiplier": self.config.get("atr_multiplier", 2),
            
            # Stochastic parameters
            "stoch_k_period": self.config.get("stoch_k_period", 14),
            "stoch_d_period": self.config.get("stoch_d_period", 3),
            "stoch_slowing": self.config.get("stoch_slowing", 3),
            "stoch_oversold": self.config.get("stoch_oversold", 20),
            "stoch_overbought": self.config.get("stoch_overbought", 80),
            
            # ADX parameters
            "adx_period": self.config.get("adx_period", 14),
            "adx_threshold": self.config.get("adx_threshold", 25),
            
            # Supertrend parameters
            "supertrend_period": self.config.get("supertrend_period", 10),
            "supertrend_multiplier": self.config.get("supertrend_multiplier", 3),
            
            # Support/Resistance parameters
            "sr_lookback": self.config.get("sr_lookback", 100),
            "sr_price_tolerance": self.config.get("sr_price_tolerance", 0.02),
            "sr_window_size": self.config.get("sr_window_size", 5),
            
            # Fibonacci parameters
            "fib_lookback": self.config.get("fib_lookback", 100),
            
            # Moving Average parameters
            "sma_periods": self.config.get("sma_periods", [5, 10, 20, 50, 200]),
            "ema_periods": self.config.get("ema_periods", [5, 12, 26, 50]),
            
            # Volume parameters
            "volume_ma_periods": self.config.get("volume_ma_periods", [10, 20, 50]),
            "high_volume_threshold": self.config.get("high_volume_threshold", 1.5),
            "low_volume_threshold": self.config.get("low_volume_threshold", 0.5),
            
            # Alligator parameters
            "alligator_jaw": self.config.get("alligator_jaw", 13),
            "alligator_teeth": self.config.get("alligator_teeth", 8),
            "alligator_lips": self.config.get("alligator_lips", 5),
            
            # CPR parameters
            "cpr_use_previous_day": self.config.get("cpr_use_previous_day", True),
            
            # ROC parameters
            "roc_period": self.config.get("roc_period", 10),
            
            # Aroon parameters
            "aroon_period": self.config.get("aroon_period", 14),
            "aroon_uptrend": self.config.get("aroon_uptrend", 70),
            "aroon_downtrend": self.config.get("aroon_downtrend", 30),
            
            # Parabolic SAR parameters
            "psar_af": self.config.get("psar_af", 0.02),
            "psar_max_af": self.config.get("psar_max_af", 0.2),
        }
        
        # Signal generation parameters
        self.SIGNAL_PARAMS = {
            # Minimum RRR threshold for valid signals
            "min_rrr": self.config.get("min_rrr", 1.5),
            
            # ATR multipliers for targets and stops
            "target_multiplier": self.config.get("target_multiplier", 1.5),
            "stop_multiplier": self.config.get("stop_multiplier", 1.0),
            
            # Minimum signal strength threshold
            "min_signal_strength": self.config.get("min_signal_strength", 3),
            
            # Checklist thresholds
            "checklist_high_confidence_threshold": self.config.get("checklist_high_confidence_threshold", 4),
            "checklist_medium_confidence_threshold": self.config.get("checklist_medium_confidence_threshold", 3),
            "checklist_low_confidence_threshold": self.config.get("checklist_low_confidence_threshold", 2),
            
            # Pattern strength factors
            "pattern_strength_weights": self.config.get("pattern_strength_weights", {
                "bullish_marubozu": 3, "bearish_marubozu": 3,
                "hammer": 3, "hanging_man": 3, "shooting_star": 3,
                "bullish_engulfing": 4, "bearish_engulfing": 4,
                "bullish_harami": 3, "bearish_harami": 3,
                "piercing_pattern": 3, "dark_cloud_cover": 3,
                "morning_star": 4, "evening_star": 4,
                "three_white_soldiers": 4, "three_black_crows": 4,
                "doji": 1, "spinning_tops": 1
            }),
            
            # Indicator strength weights
            "indicator_strength_weights": self.config.get("indicator_strength_weights", {
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
            }),
            
            # Support & Resistance significance thresholds
            "sr_near_threshold_pct": self.config.get("sr_near_threshold_pct", 2.0),
            "sr_very_near_threshold_pct": self.config.get("sr_very_near_threshold_pct", 1.0),
        }
        
        # Chart pattern detection parameters
        self.CHART_PATTERN_PARAMS = {
            # Double top/bottom parameters
            "double_pattern_tolerance": self.config.get("double_pattern_tolerance", 0.03),
            "double_pattern_lookback": self.config.get("double_pattern_lookback", 50),
            
            # Triple top/bottom parameters
            "triple_pattern_tolerance": self.config.get("triple_pattern_tolerance", 0.03),
            "triple_pattern_lookback": self.config.get("triple_pattern_lookback", 100),
            
            # Trading range parameters
            "range_lookback": self.config.get("range_lookback", 60),
            "range_tolerance": self.config.get("range_tolerance", 0.05),
            "range_min_touches": self.config.get("range_min_touches", 4),
            
            # Flag pattern parameters
            "flag_lookback": self.config.get("flag_lookback", 30),
            "flag_pole_threshold": self.config.get("flag_pole_threshold", 0.15),
            "flag_consolidation_threshold": self.config.get("flag_consolidation_threshold", 0.05),
            "flag_max_bars": self.config.get("flag_max_bars", 15),
            
            # Head and shoulders parameters
            "head_shoulders_head_tolerance": self.config.get("head_shoulders_head_tolerance", 0.03),
            "head_shoulders_shoulder_tolerance": self.config.get("head_shoulders_shoulder_tolerance", 0.05),
            
            # Cup and handle parameters
            "cup_depth_threshold": self.config.get("cup_depth_threshold", 0.15),
            "cup_volume_confirmation": self.config.get("cup_volume_confirmation", True),
            
            # Rounding parameters
            "rounding_curve_smoothness": self.config.get("rounding_curve_smoothness", 0.7),
            "rounding_min_points": self.config.get("rounding_min_points", 10),
        }
        
        # Backtesting parameters
        self.BACKTEST_PARAMS = {
            "test_period_days": self.config.get("test_period_days", 250),
            "win_threshold_pct": self.config.get("win_threshold_pct", 2.0),
            "loss_threshold_pct": self.config.get("loss_threshold_pct", 2.0),
            "max_holding_days": self.config.get("max_holding_days", 30),
            "commission_pct": self.config.get("commission_pct", 0.05),
            "min_win_rate": self.config.get("min_win_rate", 0.5),
        }
    
    def get_pattern_param(self, param_name):
        """Get a pattern detection parameter"""
        return self.PATTERN_THRESHOLDS.get(param_name)
        
    def get_indicator_param(self, param_name):
        """Get a technical indicator parameter"""
        return self.INDICATOR_PARAMS.get(param_name)
        
    def get_signal_param(self, param_name):
        """Get a signal generation parameter"""
        return self.SIGNAL_PARAMS.get(param_name)
        
    def get_chart_pattern_param(self, param_name):
        """Get a chart pattern detection parameter"""
        return self.CHART_PATTERN_PARAMS.get(param_name)
        
    def get_backtest_param(self, param_name):
        """Get a backtesting parameter"""
        return self.BACKTEST_PARAMS.get(param_name)

# ===============================================================
# Helper Functions
# ===============================================================
def setup_logging(config=None):
    """
    Configure logging for the application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    # Generate a unique logger name
    instance_id = str(uuid.uuid4())[:8]
    logger_name = f'trading_bot_{instance_id}'
    
    # Get a unique logger for this instance
    logger = logging.getLogger(logger_name)
    
    # If the logger already has handlers, it's already configured
    if logger.handlers:
        return logger
    
    # Get log directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'trading_bot_{timestamp}_{instance_id}.log')
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Disable propagation to the root logger
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set level and add handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def escape_telegram_markdown(text):
    """
    Escape special characters for Telegram MarkdownV2 formatting.
    
    Args:
        text: The text to escape
        
    Returns:
        Escaped text string
    """
    if not text:
        return "N/A"
    
    # Characters that need escaping in MarkdownV2
    special_chars = '_*[]()~`>#+-=|{}.!'
    
    # Escape each special character with a backslash
    result = ""
    for char in str(text):
        if char in special_chars:
            result += f"\\{char}"
        else:
            result += char
    
    return result

@asynccontextmanager
async def get_telegram_bot(token):
    """
    Context manager for Telegram bot to ensure proper resource management
    
    Args:
        token: Telegram bot token
        
    Yields:
        Bot instance
    """
    bot = Bot(token=token)
    try:
        yield bot
    finally:
        await bot.session.close()

async def send_telegram_message(message, config, logger, retry_attempts=5):
    """
    Send message to Telegram with retry mechanism and proper resource management
    
    Args:
        message: The message text to send
        config: Configuration with Telegram credentials
        logger: Logger instance
        retry_attempts: Number of retry attempts
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    if not config.get('ENABLE_TELEGRAM_ALERTS', False):
        return False
    
    if not config.get('TELEGRAM_BOT_TOKEN', '') or not config.get('TELEGRAM_CHAT_ID', ''):
        logger.error("Telegram credentials are missing")
        return False
        
    delay = 1  # Initial delay in seconds
    
    for attempt in range(retry_attempts):
        try:
            async with get_telegram_bot(config.get('TELEGRAM_BOT_TOKEN', '')) as bot:
                # Use plain text format for now to avoid escaping issues
                await bot.send_message(
                    chat_id=config.get('TELEGRAM_CHAT_ID', ''), 
                    text=message,
                    parse_mode='MarkdownV2'
                )
                logger.info(f"Successfully sent telegram message (attempt {attempt+1})")
                return True
        except Exception as e:
            if "Too Many Requests" in str(e):
                retry_after = int(str(e).split("retry after ")[-1].split()[0]) if "retry after" in str(e) else delay
                logger.error(f"Error sending Telegram message: {e}. Retrying in {retry_after} seconds.")
                await asyncio.sleep(retry_after)
            else:
                logger.error(f"Error sending Telegram message: {e}. Retrying in {delay} seconds.")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    logger.error(f"Failed to send Telegram message after {retry_attempts} attempts")
    return False

def get_stock_info_by_key(instrument_key, stock_info_dict):
    """
    Get stock info from instrument key (e.g., NSE_EQ|INE117A01022)
    
    Args:
        instrument_key: The instrument key to lookup
        stock_info_dict: Dictionary mapping ISINs to stock information
        
    Returns:
        Dictionary with stock information
    """
    # Create a reverse mapping from symbol to ISIN
    symbol_to_isin = {info["symbol"]: isin for isin, info in stock_info_dict.items()}
    
    parts = instrument_key.split('|')
    if len(parts) == 2:
        isin = parts[1]
        if isin in stock_info_dict:
            return stock_info_dict[isin]
    
    # Try direct symbol match as fallback
    if instrument_key in symbol_to_isin:
        isin = symbol_to_isin[instrument_key]
        return stock_info_dict[isin]
    
    return {"name": "", "industry": "", "symbol": instrument_key, "series": ""}

# ===============================================================
# Market Data Handling
# ===============================================================
class UpstoxClient:
    """Client for interacting with Upstox API"""
    
    def __init__(self, config):
        """Initialize Upstox client with API credentials"""
        self.api_key = config.get('UPSTOX_API_KEY', '')
        self.api_secret = config.get('UPSTOX_API_SECRET', '')
        self.redirect_uri = config.get('UPSTOX_REDIRECT_URI', '')
        self.code = config.get('UPSTOX_CODE', '')
        self.access_token = None
        self.client = None
        self.logger = logging.getLogger(__name__)
        
    def authenticate(self):
        """Authenticate with Upstox API"""
        try:
            # Initialize API client
            api_client = ApiClient()
            login_api = LoginApi(api_client)
            
            # Assuming you have the client_id and other necessary details
            client_id = self.api_key
            redirect_uri = self.redirect_uri
            api_version = "v2"
            
            # This will just return the authorization URL
            # In a real application, you'd need to handle the OAuth flow manually
            auth_url = login_api.authorize(client_id, redirect_uri, api_version)
            self.logger.info(f"Authorization URL: {auth_url}")
            
            # Assuming you have the access token after the OAuth flow
            api_client.configuration.access_token = self.code  # The UPSTOX_CODE is actually the access token
            self.client = MarketQuoteApi(api_client)
            
            # Test the connection
            try:
                # This is a simplified example - in reality, you'd check with a valid method
                market_indices = self.client.get_market_indices_v2(["NSE_INDEX|Nifty 50"])
                self.logger.info(f"Authentication successful for Upstox API")
                return True
            except Exception as e:
                self.logger.error(f"Failed to validate access token: {str(e)}")
                # If the access token is invalid or expired, try to refresh it
                return self._refresh_token()
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            raise APIConnectionError(f"Failed to connect to Upstox API: {str(e)}")
    
    def _refresh_token(self):
        """Refresh the access token if needed"""
        try:
            # Generate and set access token using authorization code
            url = "https://api.upstox.com/v2/login/authorization/token"
            headers = {
                'accept': 'application/json',
                'Api-Version': '2.0',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {
                'code': self.code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            import requests  # Import here to avoid dependency if not used
            response = requests.post(url, headers=headers, data=data)
            response_data = json.loads(response.text)
            
            if 'access_token' not in response_data:
                self.logger.error(f"Authentication failed: {response.text}")
                return False
            
            # Store access token
            self.access_token = response_data['access_token']
            
            # Create client with access token
            api_client = ApiClient()
            api_client.configuration.access_token = self.access_token
            self.client = MarketQuoteApi(api_client)
            
            self.logger.info("Authentication refreshed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Token refresh error: {str(e)}")
            return False
    
    def get_historical_data(self, instrument_key, interval, from_date, to_date):
        """
        Get historical OHLCV data from Upstox
        
        Args:
            instrument_key: Instrument identifier
            interval: Time interval (1D, 1W, etc.)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Create a historical candle API client
            api_client = ApiClient()
            api_client.configuration.access_token = self.code
            historical_client = HistoryApi(api_client)
            
            # Convert dates to epoch
            from_epoch = int(datetime.datetime.strptime(from_date, "%Y-%m-%d").timestamp())
            to_epoch = int(datetime.datetime.strptime(to_date, "%Y-%m-%d").timestamp())
            
            # Make API request
            historical_data = historical_client.get_historical_candle_data(
                instrument_key=instrument_key,
                interval=interval,
                to_date=to_epoch,
                from_date=from_epoch
            )
            
            # Extract candle data
            candles = historical_data.data.candles
            
            # Create DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types for all columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'oi']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp (oldest to newest)
            df.sort_index(inplace=True)
            
            # Check for minimum data points
            if len(df) < 30:
                self.logger.warning(f"Retrieved only {len(df)} candles for {instrument_key}, which may be insufficient for reliable analysis")
            
            self.logger.info(f"Successfully fetched {len(df)} candles for {instrument_key}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            raise DataFetchError(instrument_key, cause=e)

    def get_instrument_details(self, instrument_key):
        """Get instrument details from Upstox"""
        try:
            # Get market quote for the instrument
            market_quote = self.client.get_full_market_quote([instrument_key])
            
            # Extract basic instrument details from response
            data = market_quote.data[instrument_key]
            
            instrument_details = {
                'name': data.company_name,
                'tradingsymbol': data.symbol,
                'exchange': data.exchange,
                'last_price': data.last_price,
                'change_pct': data.change_percentage,
                'volume': data.volume,
                'ohlc': {
                    'open': data.ohlc.open,
                    'high': data.ohlc.high,
                    'low': data.ohlc.low,
                    'close': data.ohlc.close
                }
            }
            
            return instrument_details
            
        except Exception as e:
            self.logger.error(f"Error getting instrument details: {str(e)}")
            raise DataFetchError(instrument_key, f"Failed to get instrument details", cause=e)

# ===============================================================
# Candlestick Pattern Recognition
# ===============================================================
class CandlestickPatterns:
    """Complete candlestick pattern detection with precise validation criteria"""
    
    def __init__(self, df, params=None):
        """
        Initialize with DataFrame containing OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            params: Optional TradingParameters instance for pattern thresholds
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Initialize parameters (use defaults if not provided)
        self.params = params or TradingParameters()
        
        # Calculate body sizes and shadows for all candles
        self._calculate_candle_dimensions()
        
        # Calculate trends for context
        self._calculate_trend_context()
    
    def _calculate_candle_dimensions(self):
        """
        Calculate candle body, upper shadow, and lower shadow sizes using vectorized operations
        """
        # Get body size and direction
        self.df['body_size'] = abs(self.df['close'] - self.df['open'])
        self.df['candle_range'] = self.df['high'] - self.df['low']
        
        # Avoid division by zero with where
        self.df['body_pct'] = np.where(
            self.df['candle_range'] > 0,
            self.df['body_size'] / self.df['candle_range'],
            0
        )
        
        self.df['is_bullish'] = self.df['close'] > self.df['open']
        
        # Calculate shadows using vectorized operations
        # For bullish candles: Upper shadow = High - Close, Lower shadow = Open - Low
        # For bearish candles: Upper shadow = High - Open, Lower shadow = Close - Low
        self.df['upper_shadow'] = np.where(
            self.df['is_bullish'],
            self.df['high'] - self.df['close'],
            self.df['high'] - self.df['open']
        )
        
        self.df['lower_shadow'] = np.where(
            self.df['is_bullish'],
            self.df['open'] - self.df['low'],
            self.df['close'] - self.df['low']
        )
        
        # Calculate shadow percentages relative to range (avoid division by zero)
        self.df['upper_shadow_pct'] = np.where(
            self.df['candle_range'] > 0,
            self.df['upper_shadow'] / self.df['candle_range'],
            0
        )
        
        self.df['lower_shadow_pct'] = np.where(
            self.df['candle_range'] > 0,
            self.df['lower_shadow'] / self.df['candle_range'],
            0
        )
        
        # Handle volume analysis
        if 'volume' in self.df.columns:
            # Calculate volume moving average
            self.df['volume_ma20'] = self.df['volume'].rolling(window=20).mean()
            
            # Flag high volume candles (50% above average)
            high_volume_threshold = self.params.get_indicator_param('high_volume_threshold')
            self.df['high_volume'] = self.df['volume'] > (self.df['volume_ma20'] * high_volume_threshold)
            
            # Calculate volume ratio to average (avoid division by zero)
            self.df['volume_ratio'] = np.where(
                self.df['volume_ma20'] > 0,
                self.df['volume'] / self.df['volume_ma20'],
                0
            )
    
    def _calculate_trend_context(self):
        """
        Calculate trend indicators for pattern context with volume confirmation using vectorized operations
        """
        # Short-term trend (5-day)
        self.df['ma5'] = self.df['close'].rolling(window=5).mean()
        
        # Medium-term trend (20-day)
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        
        # Basic trend determination (vectorized)
        price_uptrend = (self.df['ma5'] > self.df['ma5'].shift(3)) & (self.df['close'] > self.df['ma20'])
        price_downtrend = (self.df['ma5'] < self.df['ma5'].shift(3)) & (self.df['close'] < self.df['ma20'])
        
        # Volume confirmation if available
        if 'volume' in self.df.columns:
            # Calculate volume trend (increasing or decreasing volume)
            self.df['volume_ma5'] = self.df['volume'].rolling(window=5).mean()
            volume_increasing = self.df['volume_ma5'] > self.df['volume_ma5'].shift(3)
            
            # Strong uptrend: Price rising with increasing volume (vectorized condition)
            self.df['uptrend'] = price_uptrend & (
                volume_increasing | 
                # Or high volume on up days
                (self.df['high_volume'] & self.df['is_bullish'])
            )
            
            # Strong downtrend: Price falling with increasing volume (vectorized condition)
            self.df['downtrend'] = price_downtrend & (
                volume_increasing | 
                # Or high volume on down days
                (self.df['high_volume'] & ~self.df['is_bullish'])
            )
        else:
            # If no volume data, use just price
            self.df['uptrend'] = price_uptrend
            self.df['downtrend'] = price_downtrend
    
    def detect_marubozu(self):
        """
        Detect Marubozu patterns (candles with no shadows) using vectorized operations
        
        Returns:
            Dictionary with detected patterns and their indices
        """
        # Ensure we have data
        if len(self.df) == 0:
            return {'marubozu': pd.Series(dtype=bool), 
                    'bullish_marubozu': pd.Series(dtype=bool), 
                    'bearish_marubozu': pd.Series(dtype=bool)}
            
        # Get parameters from config
        shadow_threshold = self.params.get_pattern_param('marubozu_shadow_threshold')
        body_pct_threshold = self.params.get_pattern_param('marubozu_body_pct')
        
        # Detect Bullish Marubozu (vectorized)
        bullish_marubozu = (
            (self.df['is_bullish']) &
            (self.df['upper_shadow_pct'] <= shadow_threshold) &
            (self.df['lower_shadow_pct'] <= shadow_threshold) &
            (self.df['body_pct'] >= body_pct_threshold)  # Body is at least 95% of range
        )
        
        # Detect Bearish Marubozu (vectorized)
        bearish_marubozu = (
            (~self.df['is_bullish']) &
            (self.df['upper_shadow_pct'] <= shadow_threshold) &
            (self.df['lower_shadow_pct'] <= shadow_threshold) &
            (self.df['body_pct'] >= body_pct_threshold)  # Body is at least 95% of range
        )
        
        # Combine results
        return {
            'marubozu': bullish_marubozu | bearish_marubozu,
            'bullish_marubozu': bullish_marubozu,
            'bearish_marubozu': bearish_marubozu
        }
    
    def detect_doji(self):
        """
        Detect Doji patterns (candles with small bodies and shadows) using vectorized operations
        
        Returns:
            Series with True at indices where Doji patterns are detected
        """
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(False, index=self.df.index)
            
        # Get parameter from config
        body_threshold = self.params.get_pattern_param('doji_body_threshold')
        
        # Doji criteria: open and close are virtually equal (vectorized)
        doji = (
            (self.df['body_pct'] <= body_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return doji
    
    def detect_spinning_tops(self):
        """
        Detect Spinning Top patterns (small bodies, long shadows) using vectorized operations
        
        Returns:
            Series with True at indices where Spinning Tops are detected
        """
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(False, index=self.df.index)
            
        # Get parameters from config
        body_threshold = self.params.get_pattern_param('spinning_top_body_threshold')
        shadow_threshold = self.params.get_pattern_param('spinning_top_shadow_threshold')
        
        # Spinning Top criteria (vectorized)
        spinning_tops = (
            (self.df['body_pct'] <= body_threshold) &
            (self.df['upper_shadow_pct'] >= shadow_threshold) &
            (self.df['lower_shadow_pct'] >= shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return spinning_tops
    
    def detect_paper_umbrella(self):
        """
        Detect Paper Umbrella patterns (small body at top, long lower shadow) using vectorized operations
        
        Returns:
            Series with True at indices where Paper Umbrella patterns are detected
        """
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(False, index=self.df.index)
            
        # Get parameters from config
        lower_shadow_ratio = self.params.get_pattern_param('umbrella_lower_shadow_ratio')
        upper_shadow_threshold = self.params.get_pattern_param('umbrella_upper_shadow_threshold')
        
        # Paper Umbrella criteria (vectorized)
        paper_umbrella = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return paper_umbrella
    
    def detect_hammer(self):
        """
        Detect Hammer patterns (bullish reversal with small body at top, long lower shadow) using vectorized operations
        
        Returns:
            Series with True at indices where Hammer patterns are detected
        """
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
        # Get parameters from config
        lower_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')
        upper_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
        
        # Detect hammer candle structure (vectorized)
        hammer_structure = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior downtrend
        downtrend_context = self.df['downtrend'].shift(1)
        
        # Combine conditions
        hammer = hammer_structure & downtrend_context
        
        return hammer
    
    def detect_hanging_man(self):
        """
        Detect Hanging Man patterns (bearish reversal with small body at top, long lower shadow) using vectorized operations
        
        Returns:
            Series with True at indices where Hanging Man patterns are detected
        """
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
        # Get parameters from config - using same as hammer since structure is similar
        lower_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')
        upper_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
        
        # Detect hanging man candle structure (same as hammer, vectorized)
        hanging_man_structure = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior uptrend
        uptrend_context = self.df['uptrend'].shift(1)
        
        # Combine conditions
        hanging_man = hanging_man_structure & uptrend_context
        
        return hanging_man
    
    def detect_shooting_star(self):
        """
        Detect Shooting Star patterns (bearish reversal with small body at bottom, long upper shadow) using vectorized operations
        
        Returns:
            Series with True at indices where Shooting Star patterns are detected
        """
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
        # Get parameters from config
        # Using same ratio as hammer but for upper shadow
        upper_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')  
        lower_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
        
        # Detect shooting star candle structure (vectorized)
        shooting_star_structure = (
            (self.df['upper_shadow'] >= (self.df['body_size'] * upper_shadow_ratio)) &
            (self.df['lower_shadow_pct'] <= lower_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior uptrend
        uptrend_context = self.df['uptrend'].shift(1)
        
        # Combine conditions
        shooting_star = shooting_star_structure & uptrend_context
        
        return shooting_star
    
    def detect_engulfing(self):
        """
        Detect Engulfing patterns (two-candle reversal pattern) using vectorized operations
        
        Returns:
            Dictionary with bullish_engulfing and bearish_engulfing Series
        """
        # Initialize result Series
        bullish_engulfing = pd.Series(False, index=self.df.index)
        bearish_engulfing = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect engulfing patterns
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return {'bullish_engulfing': bullish_engulfing, 'bearish_engulfing': bearish_engulfing}
        
        # Create shifted dataframes for comparison
        df_prev = self.df.shift(1)
        
        # Bullish Engulfing (vectorized)
        bullish_engulfing_condition = (
            ~self.df['is_bullish'].shift(1) &  # Previous candle is bearish
            self.df['is_bullish'] &            # Current candle is bullish
            (self.df['open'] <= df_prev['close']) &  # Current open below previous close
            (self.df['close'] >= df_prev['open'])    # Current close above previous open
        )
        
        # Add trend context (at least 5 periods of history)
        downtrend_context = self.df['downtrend'].shift(1)
        bullish_engulfing_valid = bullish_engulfing_condition & downtrend_context
        
        # Bearish Engulfing (vectorized)
        bearish_engulfing_condition = (
            self.df['is_bullish'].shift(1) &    # Previous candle is bullish
            ~self.df['is_bullish'] &            # Current candle is bearish
            (self.df['open'] >= df_prev['close']) &  # Current open above previous close
            (self.df['close'] <= df_prev['open'])    # Current close below previous open
        )
        
        # Add trend context
        uptrend_context = self.df['uptrend'].shift(1)
        bearish_engulfing_valid = bearish_engulfing_condition & uptrend_context
        
        return {
            'bullish_engulfing': bullish_engulfing_valid,
            'bearish_engulfing': bearish_engulfing_valid
        }
    
    def detect_harami(self):
        """
        Detect Harami patterns (two-candle pattern where second candle is contained within first) using vectorized operations
        
        Returns:
            Dictionary with bullish_harami and bearish_harami Series
        """
        # Initialize result Series with index from self.df
        bullish_harami = pd.Series(False, index=self.df.index)
        bearish_harami = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect harami patterns
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return {'bullish_harami': bullish_harami, 'bearish_harami': bearish_harami}
        
        # Create shifted dataframes for comparison
        df_prev = self.df.shift(1)
        
        # Get harami body size ratio parameter
        harami_body_size_ratio = self.params.get_pattern_param('harami_body_size_ratio')
        
        # Bullish Harami (vectorized)
        bullish_harami_condition = (
            ~df_prev['is_bullish'] &  # Previous candle is bearish
            self.df['is_bullish'] &   # Current candle is bullish
            (self.df['open'] > df_prev['close']) &  # Current open inside previous body
            (self.df['open'] < df_prev['open']) &
            (self.df['close'] > df_prev['close']) &  # Current close inside previous body
            (self.df['close'] < df_prev['open']) &
            (self.df['body_size'] < df_prev['body_size'] * harami_body_size_ratio)  # Current body smaller than previous
        )
        
        # Add trend context
        downtrend_context = self.df['downtrend'].shift(1)
        bullish_harami_valid = bullish_harami_condition & downtrend_context
        
        # Bearish Harami (vectorized)
        bearish_harami_condition = (
            df_prev['is_bullish'] &    # Previous candle is bullish
            ~self.df['is_bullish'] &   # Current candle is bearish
            (self.df['open'] < df_prev['close']) &  # Current open inside previous body
            (self.df['open'] > df_prev['open']) &
            (self.df['close'] < df_prev['close']) &  # Current close inside previous body
            (self.df['close'] > df_prev['open']) &
            (self.df['body_size'] < df_prev['body_size'] * harami_body_size_ratio)  # Current body smaller than previous
        )
        
        # Add trend context
        uptrend_context = self.df['uptrend'].shift(1)
        bearish_harami_valid = bearish_harami_condition & uptrend_context
        
        return {
            'bullish_harami': bullish_harami_valid,
            'bearish_harami': bearish_harami_valid
        }
    
    def detect_piercing_pattern(self):
        """
        Detect Piercing Pattern (bullish reversal pattern) using vectorized operations
        
        Returns:
            Series with True at indices where Piercing Pattern is detected
        """
        # Initialize result Series with index from self.df
        piercing_pattern = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect piercing patterns
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return piercing_pattern
        
        # Create shifted dataframes for comparison
        df_prev = self.df.shift(1)
        
        # Piercing Pattern (vectorized)
        piercing_condition = (
            ~df_prev['is_bullish'] &  # Previous candle is bearish
            self.df['is_bullish'] &   # Current candle is bullish
            (df_prev['body_size'] / df_prev['candle_range'] > 0.6) &  # Previous is a long candle
            (self.df['open'] < df_prev['low']) &  # Gap down opening
            (self.df['close'] > (df_prev['open'] + df_prev['close']) / 2) &  # Close above midpoint
            (self.df['close'] < df_prev['open'])  # Not a complete bullish engulfing
        )
        
        # Add trend context
        downtrend_context = self.df['downtrend'].shift(1)
        piercing_pattern_valid = piercing_condition & downtrend_context
        
        return piercing_pattern_valid
    
    def detect_dark_cloud_cover(self):
        """
        Detect Dark Cloud Cover (bearish reversal pattern) using vectorized operations
        
        Returns:
            Series with True at indices where Dark Cloud Cover is detected
        """
        # Initialize result Series with index from self.df
        dark_cloud_cover = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect dark cloud cover
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return dark_cloud_cover
        
        # Create shifted dataframes for comparison
        df_prev = self.df.shift(1)
        
        # Dark Cloud Cover (vectorized)
        dark_cloud_condition = (
            df_prev['is_bullish'] &    # Previous candle is bullish
            ~self.df['is_bullish'] &   # Current candle is bearish
            (df_prev['body_size'] / df_prev['candle_range'] > 0.6) &  # Previous is a long candle
            (self.df['open'] > df_prev['high']) &  # Gap up opening
            (self.df['close'] < (df_prev['open'] + df_prev['close']) / 2) &  # Close below midpoint
            (self.df['close'] > df_prev['open'])  # Not a complete bearish engulfing
        )
        
        # Add trend context
        uptrend_context = self.df['uptrend'].shift(1)
        dark_cloud_cover_valid = dark_cloud_condition & uptrend_context
        
        return dark_cloud_cover_valid
    
    def detect_morning_star(self):
        """
        Detect Morning Star (bullish reversal pattern)
        
        Note: This pattern is complex and retains some loop logic but is optimized
        
        Returns:
            Series with True at indices where Morning Star is detected
        """
        # Initialize result Series with index from self.df
        morning_star = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles to detect morning star
        if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
            return morning_star
        
        # Get star body size threshold
        star_body_size_threshold = self.params.get_pattern_param('star_body_size_threshold')
        
        # Create shifted dataframes for comparison
        df_prev1 = self.df.shift(1)  # second day
        df_prev2 = self.df.shift(2)  # first day
        
        # Morning Star conditions (vectorized where possible)
        condition = (
            ~df_prev2['is_bullish'] &  # First candle is bearish
            self.df['is_bullish'] &    # Third candle is bullish
            (df_prev2['body_size'] / df_prev2['candle_range'] > 0.6) &  # First is a long candle
            (df_prev1['body_size'] / df_prev1['candle_range'] < star_body_size_threshold) &  # Second has a small body
            (df_prev1['close'] < df_prev2['close']) &  # Second gaps down from first
            (self.df['open'] > df_prev1['close']) &  # Third opens above second close
            (self.df['close'] > (df_prev2['open'] + df_prev2['close']) / 2)  # Third closes above midpoint of first
        )
        
        # Add trend context
        downtrend_context = self.df['downtrend'].shift(2)
        morning_star_valid = condition & downtrend_context
        
        return morning_star_valid
    
    def detect_evening_star(self):
        """
        Detect Evening Star (bearish reversal pattern)
        
        Note: This pattern is complex and retains some loop logic but is optimized
        
        Returns:
            Series with True at indices where Evening Star is detected
        """
        # Initialize result Series with index from self.df
        evening_star = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles to detect evening star
        if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
            return evening_star
        
        # Get star body size threshold
        star_body_size_threshold = self.params.get_pattern_param('star_body_size_threshold')
        
        # Create shifted dataframes for comparison
        df_prev1 = self.df.shift(1)  # second day
        df_prev2 = self.df.shift(2)  # first day
        
        # Evening Star conditions (vectorized where possible)
        condition = (
            df_prev2['is_bullish'] &    # First candle is bullish
            ~self.df['is_bullish'] &    # Third candle is bearish
            (df_prev2['body_size'] / df_prev2['candle_range'] > 0.6) &  # First is a long candle
            (df_prev1['body_size'] / df_prev1['candle_range'] < star_body_size_threshold) &  # Second has a small body
            (df_prev1['close'] > df_prev2['close']) &  # Second gaps up from first
            (self.df['open'] < df_prev1['close']) &  # Third opens below second close
            (self.df['close'] < (df_prev2['open'] + df_prev2['close']) / 2)  # Third closes below midpoint of first
        )
        
        # Add trend context
        uptrend_context = self.df['uptrend'].shift(2)
        evening_star_valid = condition & uptrend_context
        
        return evening_star_valid
    
    def detect_three_white_soldiers(self):
        """
        Detect Three White Soldiers (bullish reversal pattern)
        
        Returns:
            Series with True at indices where Three White Soldiers is detected
        """
        # Initialize result Series with index from self.df
        three_white_soldiers = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles to detect this pattern
        if len(self.df) < 7:  # Need at least 7 candles (4 for trend context + 3 pattern candles)
            return three_white_soldiers
        
        # Create shifted dataframes for comparison
        df_prev1 = self.df.shift(1)  # yesterday
        df_prev2 = self.df.shift(2)  # two days ago
        
        # Get Three White Soldiers threshold
        trend_threshold = self.params.get_pattern_param('three_candle_trend_threshold')
        
        # Three White Soldiers conditions
        condition = (
            # Each candle is bullish
            self.df['is_bullish'] & 
            df_prev1['is_bullish'] & 
            df_prev2['is_bullish'] &
            
            # Each candle closes higher than the previous
            (self.df['close'] > df_prev1['close']) &
            (df_prev1['close'] > df_prev2['close']) &
            
            # Each candle opens within the body of the previous candle
            (self.df['open'] > df_prev1['open']) &
            (self.df['open'] < df_prev1['close']) &
            (df_prev1['open'] > df_prev2['open']) &
            (df_prev1['open'] < df_prev2['close']) &
            
            # Each candle has relatively small upper shadows
            (self.df['upper_shadow_pct'] < 0.2) &
            (df_prev1['upper_shadow_pct'] < 0.2) &
            (df_prev2['upper_shadow_pct'] < 0.2) &
            
            # No significant gaps
            (abs(self.df['open'] - df_prev1['close']) / df_prev1['close'] < trend_threshold) &
            (abs(df_prev1['open'] - df_prev2['close']) / df_prev2['close'] < trend_threshold)
        )
        
        # Add trend context - should occur after a downtrend
        downtrend_context = self.df['downtrend'].shift(3)
        three_white_soldiers_valid = condition & downtrend_context
        
        return three_white_soldiers_valid

    def detect_three_black_crows(self):
        """
        Detect Three Black Crows (bearish reversal pattern)
        
        Returns:
            Series with True at indices where Three Black Crows is detected
        """
        # Initialize result Series with index from self.df
        three_black_crows = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles to detect this pattern
        if len(self.df) < 7:  # Need at least 7 candles (4 for trend context + 3 pattern candles)
            return three_black_crows
        
        # Create shifted dataframes for comparison
        df_prev1 = self.df.shift(1)  # yesterday
        df_prev2 = self.df.shift(2)  # two days ago
        
        # Get Three Black Crows threshold
        trend_threshold = self.params.get_pattern_param('three_candle_trend_threshold')
        
        # Three Black Crows conditions
        condition = (
            # Each candle is bearish
            ~self.df['is_bullish'] & 
            ~df_prev1['is_bullish'] & 
            ~df_prev2['is_bullish'] &
            
            # Each candle closes lower than the previous
            (self.df['close'] < df_prev1['close']) &
            (df_prev1['close'] < df_prev2['close']) &
            
            # Each candle opens within the body of the previous candle
            (self.df['open'] < df_prev1['open']) &
            (self.df['open'] > df_prev1['close']) &
            (df_prev1['open'] < df_prev2['open']) &
            (df_prev1['open'] > df_prev2['close']) &
            
            # Each candle has relatively small lower shadows
            (self.df['lower_shadow_pct'] < 0.2) &
            (df_prev1['lower_shadow_pct'] < 0.2) &
            (df_prev2['lower_shadow_pct'] < 0.2) &
            
            # No significant gaps
            (abs(self.df['open'] - df_prev1['close']) / df_prev1['close'] < trend_threshold) &
            (abs(df_prev1['open'] - df_prev2['close']) / df_prev2['close'] < trend_threshold)
        )
        
        # Add trend context - should occur after an uptrend
        uptrend_context = self.df['uptrend'].shift(3)
        three_black_crows_valid = condition & uptrend_context
        
        return three_black_crows_valid
    
    def detect_all_patterns(self):
        """
        Detect all implemented candlestick patterns
        
        Returns:
            Dictionary with all detected patterns
        """
        # Get basic patterns
        marubozu_patterns = self.detect_marubozu()
        
        # Build comprehensive pattern dictionary
        patterns = {
            # Single candle patterns
            'marubozu': marubozu_patterns['marubozu'],
            'bullish_marubozu': marubozu_patterns['bullish_marubozu'],
            'bearish_marubozu': marubozu_patterns['bearish_marubozu'],
            'doji': self.detect_doji(),
            'spinning_tops': self.detect_spinning_tops(),
            'paper_umbrella': self.detect_paper_umbrella(),
            'hammer': self.detect_hammer(),
            'hanging_man': self.detect_hanging_man(),
            'shooting_star': self.detect_shooting_star(),
            
            # Two-candle patterns
            **self.detect_engulfing(),  # Adds bullish_engulfing and bearish_engulfing
            **self.detect_harami(),     # Adds bullish_harami and bearish_harami
            'piercing_pattern': self.detect_piercing_pattern(),
            'dark_cloud_cover': self.detect_dark_cloud_cover(),
            
            # Multi-candle patterns
            'morning_star': self.detect_morning_star(),
            'evening_star': self.detect_evening_star(),
            'three_white_soldiers': self.detect_three_white_soldiers(),
            'three_black_crows': self.detect_three_black_crows()
        }
        
        return patterns
    
    def get_latest_patterns(self):
        """
        Get the latest detected patterns (those at the end of the DataFrame)
        
        Returns:
            Dictionary of patterns found in the latest candle
        """
        # Detect all patterns if not already detected
        all_patterns = self.detect_all_patterns()
        
        # Get only patterns at the latest candle
        latest_patterns = {}
        for pattern_name, pattern_series in all_patterns.items():
            if len(pattern_series) > 0 and pattern_series.iloc[-1]:
                latest_patterns[pattern_name] = True
        
        return latest_patterns
    
    def get_pattern_signals(self):
        """
        Get trading signals from detected patterns
        
        Returns:
            Dictionary with buy and sell signals based on detected patterns
        """
        # Get latest patterns
        latest_patterns = self.get_latest_patterns()
        
        # Define bullish and bearish patterns
        bullish_patterns = [
            'bullish_marubozu', 'hammer', 'piercing_pattern',
            'bullish_engulfing', 'bullish_harami', 'morning_star',
            'three_white_soldiers'
        ]
        
        bearish_patterns = [
            'bearish_marubozu', 'hanging_man', 'shooting_star',
            'dark_cloud_cover', 'bearish_engulfing', 'bearish_harami',
            'evening_star', 'three_black_crows'
        ]
        
        # Group detected patterns by signal type
        buy_signals = []
        sell_signals = []
        
        # Get pattern strength weights from parameters
        pattern_weights = self.params.get_signal_param('pattern_strength_weights')
        
        # Check for each bullish pattern
        for pattern in bullish_patterns:
            if pattern in latest_patterns:
                strength = pattern_weights.get(pattern, 1)  # Default strength is 1
                buy_signals.append({
                    'pattern': pattern.replace('_', ' ').title(),
                    'strength': strength
                })
        
        # Check for each bearish pattern
        for pattern in bearish_patterns:
            if pattern in latest_patterns:
                strength = pattern_weights.get(pattern, 1)  # Default strength is 1
                sell_signals.append({
                    'pattern': pattern.replace('_', ' ').title(),
                    'strength': strength
                })
        
        # Return dictionary with buy and sell signals
        return {
            'buy': buy_signals,
            'sell': sell_signals
        }

# ===============================================================
# Technical Indicators
# ===============================================================
class TechnicalIndicators:
    """Calculate and interpret technical indicators for trading signals"""
    
    def __init__(self, df, params=None):
        """
        Initialize Technical Analysis with OHLCV DataFrame
        
        Args:
            df: DataFrame with OHLCV data (index=timestamp, columns=[open, high, low, close, volume])
            params: Optional TradingParameters instance for indicator parameters
        """
        # Ensure column names are lowercase
        self.df = df.copy()
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if 'volume' in missing_columns:
            # If volume is missing, create placeholder volume data
            self.df['volume'] = 0
            missing_columns.remove('volume')
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")
        
        # Set parameters from config or use defaults
        self.params = params or TradingParameters()
        
        # Initialize results dictionary and signals list
        self.indicators = {}
        self.signals = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self):
        """Calculate all technical indicators"""
        indicator_methods = {
            # Trend indicators
            "moving_averages": self.calculate_moving_averages,
            "supertrend": self.calculate_supertrend,
            "parabolic_sar": self.calculate_parabolic_sar,
            "adx": self.calculate_adx,
            "alligator": self.calculate_alligator,
            "cpr": self.calculate_cpr,
            "aroon": self.calculate_aroon,
            
            # Momentum indicators
            "macd": self.calculate_macd,
            "rsi": self.calculate_rsi,
            "stochastic": self.calculate_stochastic,
            "roc": self.calculate_rate_of_change,
            "stochastic_rsi": self.calculate_stochastic_rsi,
            "williams_r": self.calculate_williams_r,
            
            # Volatility indicators
            "bollinger_bands": self.calculate_bollinger_bands,
            "atr": self.calculate_atr,
            "atr_bands": self.calculate_atr_bands,
            
            # Volume indicators
            "obv": self.calculate_obv,
            "vwap": self.calculate_vwap,
            "volume_profile": self.calculate_volume_profile,
            
            # Support/Resistance indicators
            "support_resistance": self.calculate_support_resistance,
            "fibonacci_retracement": self.calculate_fibonacci_retracement
        }
        
        # Calculate each indicator and handle errors
        for indicator_name, calculation_method in indicator_methods.items():
            try:
                calculation_method()
                self.logger.debug(f"Successfully calculated {indicator_name}")
            except Exception as e:
                self.logger.warning(f"Error calculating {indicator_name}: {str(e)}")
                # Add error entry to indicators
                self.indicators[indicator_name] = {
                    'signal': 0,
                    'error': str(e)
                }
        
        return self.indicators
    
    def calculate_moving_averages(self):
        """Calculate Simple and Exponential Moving Averages"""
        # Get parameters for moving averages
        sma_periods = self.params.get_indicator_param('sma_periods')
        ema_periods = self.params.get_indicator_param('ema_periods')
        
        # Calculate SMAs for different periods
        for period in sma_periods:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
        
        # Calculate EMAs for different periods
        for period in ema_periods:
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        
        # Generate crossover signals
        # EMA crossover (typically 9 & 21)
        self.df['ema_crossover'] = 0
        short_ema = f'ema_{ema_periods[0]}'  # Typically 9
        long_ema = f'ema_{ema_periods[1]}'   # Typically 21
        
        self.df.loc[self.df[short_ema] > self.df[long_ema], 'ema_crossover'] = 1
        self.df.loc[self.df[short_ema] < self.df[long_ema], 'ema_crossover'] = -1
        
        # Detect crossovers (signals)
        self.df['ema_buy_signal'] = ((self.df['ema_crossover'].shift(1) == -1) & 
                                   (self.df['ema_crossover'] == 1)).astype(int)
        self.df['ema_sell_signal'] = ((self.df['ema_crossover'].shift(1) == 1) & 
                                    (self.df['ema_crossover'] == -1)).astype(int)
        
        # Golden Cross / Death Cross (SMA 50 and 200)
        mid_term = f'sma_{sma_periods[3]}'  # Typically 50
        long_term = f'sma_{sma_periods[4]}' # Typically 200
        
        self.df['golden_cross'] = ((self.df[mid_term].shift(1) <= self.df[long_term].shift(1)) & 
                                 (self.df[mid_term] > self.df[long_term])).astype(int)
        self.df['death_cross'] = ((self.df[mid_term].shift(1) >= self.df[long_term].shift(1)) & 
                                (self.df[mid_term] < self.df[long_term])).astype(int)
        
        # Determine overall trend direction
        self.df['uptrend'] = (
            (self.df['close'] > self.df[mid_term]) & 
            (self.df[mid_term] > self.df[long_term])
        ).astype(int)
        
        self.df['downtrend'] = (
            (self.df['close'] < self.df[mid_term]) & 
            (self.df[mid_term] < self.df[long_term])
        ).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        
        # Check for fresh EMA crossover
        if self.df['ema_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_strength = 2
        elif self.df['ema_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_strength = 2
        
        # Check for golden/death cross (stronger signals)
        if self.df['golden_cross'].iloc[-1] == 1:
            current_signal = 1
            signal_strength = 3
        elif self.df['death_cross'].iloc[-1] == 1:
            current_signal = -1
            signal_strength = 3
        
        # Check price position relative to key MAs for trend confirmation
        price_above_short = self.df['close'].iloc[-1] > self.df[short_ema].iloc[-1]
        price_above_mid = self.df['close'].iloc[-1] > self.df[mid_term].iloc[-1]
        price_above_long = self.df['close'].iloc[-1] > self.df[long_term].iloc[-1]
        
        # Save to indicators dictionary
        self.indicators['moving_averages'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'ema_short': round(self.df[short_ema].iloc[-1], 2) if not pd.isna(self.df[short_ema].iloc[-1]) else None,
                'ema_long': round(self.df[long_ema].iloc[-1], 2) if not pd.isna(self.df[long_ema].iloc[-1]) else None,
                'sma_mid': round(self.df[mid_term].iloc[-1], 2) if not pd.isna(self.df[mid_term].iloc[-1]) else None,
                'sma_long': round(self.df[long_term].iloc[-1], 2) if not pd.isna(self.df[long_term].iloc[-1]) else None,
                'price_above_ema_short': price_above_short,
                'price_above_sma_mid': price_above_mid,
                'price_above_sma_long': price_above_long,
                'golden_cross': self.df['golden_cross'].iloc[-1] == 1,
                'death_cross': self.df['death_cross'].iloc[-1] == 1,
                'uptrend': self.df['uptrend'].iloc[-1] == 1,
                'downtrend': self.df['downtrend'].iloc[-1] == 1
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            signal_name = "EMA Crossover"
            if self.df['golden_cross'].iloc[-1] == 1:
                signal_name = "Golden Cross"
            elif self.df['death_cross'].iloc[-1] == 1:
                signal_name = "Death Cross"
                
            self.signals.append({
                'indicator': 'Moving Averages',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_name
            })
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        # Get MACD parameters
        fast_period = self.params.get_indicator_param('macd_fast')
        slow_period = self.params.get_indicator_param('macd_slow')
        signal_period = self.params.get_indicator_param('macd_signal')
        
        # Calculate MACD components
        fast_ema = self.df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.df['close'].ewm(span=slow_period, adjust=False).mean()
        
        self.df['macd_line'] = fast_ema - slow_ema
        self.df['signal_line'] = self.df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['signal_line']
        
        # Generate signals based on crossovers
        self.df['macd_crossover'] = 0
        self.df.loc[self.df['macd_line'] > self.df['signal_line'], 'macd_crossover'] = 1
        self.df.loc[self.df['macd_line'] < self.df['signal_line'], 'macd_crossover'] = -1
        
        # Detect crossovers (signals)
        self.df['macd_buy_signal'] = ((self.df['macd_crossover'].shift(1) == -1) & 
                                    (self.df['macd_crossover'] == 1)).astype(int)
        self.df['macd_sell_signal'] = ((self.df['macd_crossover'].shift(1) == 1) & 
                                     (self.df['macd_crossover'] == -1)).astype(int)
        
        # Detect histogram direction changes
        self.df['hist_direction'] = np.sign(self.df['macd_histogram'] - self.df['macd_histogram'].shift(1))
        
        # Bullish divergence: Price makes lower lows but MACD makes higher lows
        # Bearish divergence: Price makes higher highs but MACD makes lower highs
        # (Complex calculation implemented in a simplified way)
        bullish_divergence = False
        bearish_divergence = False
        
        # Generate signal
        current_signal = 0
        signal_strength = this_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['macd_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            this_strength = 2
        elif self.df['macd_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            this_strength = 2
        
        # Strengthen signal if MACD line is above/below zero line
        if current_signal == 1 and self.df['macd_line'].iloc[-1] > 0:
            signal_type += " Above Zero"
            this_strength += 1
        elif current_signal == -1 and self.df['macd_line'].iloc[-1] < 0:
            signal_type += " Below Zero"
            this_strength += 1
            
        # Check for favorable histogram direction
        histogram_favorable = False
        if (current_signal == 1 and self.df['hist_direction'].iloc[-1] > 0) or \
           (current_signal == -1 and self.df['hist_direction'].iloc[-1] < 0):
            signal_type += " with Expanding Histogram"
            histogram_favorable = True
            this_strength += 1
        
        signal_strength = this_strength
        
        self.indicators['macd'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'macd_line': round(self.df['macd_line'].iloc[-1], 4),
                'signal_line': round(self.df['signal_line'].iloc[-1], 4),
                'histogram': round(self.df['macd_histogram'].iloc[-1], 4),
                'hist_direction': 'Increasing' if self.df['hist_direction'].iloc[-1] > 0 else 
                                 'Decreasing' if self.df['hist_direction'].iloc[-1] < 0 else 'Unchanged',
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'MACD',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_rsi(self):
        """Calculate Relative Strength Index (RSI)"""
        # Get RSI parameters
        period = self.params.get_indicator_param('rsi_period')
        oversold = self.params.get_indicator_param('rsi_oversold')
        overbought = self.params.get_indicator_param('rsi_overbought')
        
        # Calculate price changes
        delta = self.df['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss over the specified period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        self.df['rsi_oversold'] = self.df['rsi'] < oversold
        self.df['rsi_overbought'] = self.df['rsi'] > overbought
        
        # Detect crosses above oversold and below overbought
        self.df['rsi_buy_signal'] = ((self.df['rsi'] > oversold) & 
                                   (self.df['rsi'].shift(1) <= oversold)).astype(int)
        self.df['rsi_sell_signal'] = ((self.df['rsi'] < overbought) & 
                                    (self.df['rsi'].shift(1) >= overbought)).astype(int)
        
        # Detect divergences
        # (Simplified implementation for brevity)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for oversold/overbought conditions
        if self.df['rsi_oversold'].iloc[-1]:
            current_signal = 1
            signal_type = "Oversold"
            signal_strength = 1
        elif self.df['rsi_overbought'].iloc[-1]:
            current_signal = -1
            signal_type = "Overbought"
            signal_strength = 1
        
        # Check for crosses (stronger signals)
        if self.df['rsi_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Cross from Oversold"
            signal_strength = 2
        elif self.df['rsi_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Cross from Overbought"
            signal_strength = 2
        
        self.indicators['rsi'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'rsi': round(self.df['rsi'].iloc[-1], 2) if not pd.isna(self.df['rsi'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['rsi_oversold'].iloc[-1],
                'is_overbought': self.df['rsi_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'RSI',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_stochastic(self):
        """Calculate Stochastic Oscillator"""
        # Get Stochastic parameters
        k_period = self.params.get_indicator_param('stoch_k_period')
        d_period = self.params.get_indicator_param('stoch_d_period')
        slowing = self.params.get_indicator_param('stoch_slowing')
        oversold = self.params.get_indicator_param('stoch_oversold')
        overbought = self.params.get_indicator_param('stoch_overbought')
        
        # Calculate %K (The current close in relation to the range over k_period)
        lowest_low = self.df['low'].rolling(window=k_period).min()
        highest_high = self.df['high'].rolling(window=k_period).max()
        self.df['stoch_k_raw'] = 100 * ((self.df['close'] - lowest_low) / 
                                      (highest_high - lowest_low))
        
        # Apply slowing for %K
        self.df['stoch_k'] = self.df['stoch_k_raw'].rolling(window=slowing).mean()
        
        # Calculate %D (Simple moving average of %K)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
        
        # Generate signals
        self.df['stoch_oversold'] = self.df['stoch_k'] < oversold
        self.df['stoch_overbought'] = self.df['stoch_k'] > overbought
        
        # Detect K crossing above D in oversold region
        self.df['stoch_buy_signal'] = ((self.df['stoch_k'] > self.df['stoch_d']) & 
                                     (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1)) &
                                     (self.df['stoch_k'] < oversold + 5)).astype(int)
        
        # Detect K crossing below D in overbought region
        self.df['stoch_sell_signal'] = ((self.df['stoch_k'] < self.df['stoch_d']) & 
                                      (self.df['stoch_k'].shift(1) >= self.df['stoch_d'].shift(1)) &
                                      (self.df['stoch_k'] > overbought - 5)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers in oversold/overbought regions
        if self.df['stoch_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover from Oversold"
            signal_strength = 2
        elif self.df['stoch_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover from Overbought"
            signal_strength = 2
        
        # Also check for extreme oversold/overbought conditions
        elif self.df['stoch_oversold'].iloc[-1] and self.df['stoch_k'].iloc[-1] < oversold - 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 1
        elif self.df['stoch_overbought'].iloc[-1] and self.df['stoch_k'].iloc[-1] > overbought + 10:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 1
        
        self.indicators['stochastic'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'stoch_k': round(self.df['stoch_k'].iloc[-1], 2) if not pd.isna(self.df['stoch_k'].iloc[-1]) else None,
                'stoch_d': round(self.df['stoch_d'].iloc[-1], 2) if not pd.isna(self.df['stoch_d'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['stoch_oversold'].iloc[-1],
                'is_overbought': self.df['stoch_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Stochastic',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        # Get Bollinger Bands parameters
        period = self.params.get_indicator_param('bb_period')
        std_dev = self.params.get_indicator_param('bb_std_dev')
        
        # Calculate middle band (SMA)
        self.df['bb_middle'] = self.df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        self.df['bb_std'] = self.df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        self.df['bb_upper'] = self.df['bb_middle'] + (std_dev * self.df['bb_std'])
        self.df['bb_lower'] = self.df['bb_middle'] - (std_dev * self.df['bb_std'])
        
        # Calculate %B (position within bands)
        self.df['bb_pct_b'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # Calculate bandwidth
        self.df['bb_bandwidth'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        
        # Generate signals
        # Price touching or breaking bands
        self.df['bb_touch_upper'] = self.df['high'] >= self.df['bb_upper']
        self.df['bb_touch_lower'] = self.df['low'] <= self.df['bb_lower']
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for price at bands
        if self.df['bb_touch_lower'].iloc[-1]:
            current_signal = 1
            signal_type = "Price at Lower Band"
            signal_strength = 1
        elif self.df['bb_touch_upper'].iloc[-1]:
            current_signal = -1
            signal_type = "Price at Upper Band"
            signal_strength = 1
        
        # Check for Bollinger Band squeeze (narrowing bandwidth)
        # Get average bandwidth over last 20 periods
        avg_bandwidth = self.df['bb_bandwidth'].iloc[-20:].mean()
        current_bandwidth = self.df['bb_bandwidth'].iloc[-1]
        
        is_squeeze = current_bandwidth < (avg_bandwidth * 0.8)
        
        # Check for %B signals
        if self.df['bb_pct_b'].iloc[-1] < 0:
            current_signal = 1
            signal_type = "Price Below Lower Band"
            signal_strength = 2
        elif self.df['bb_pct_b'].iloc[-1] > 1:
            current_signal = -1
            signal_type = "Price Above Upper Band"
            signal_strength = 2
        
        # Improve signal with RSI confirmation
        if 'rsi' in self.df.columns:
            if current_signal == 1 and self.df['rsi'].iloc[-1] < 40:
                signal_type += " with RSI Confirmation"
                signal_strength += 1
            elif current_signal == -1 and self.df['rsi'].iloc[-1] > 60:
                signal_type += " with RSI Confirmation"
                signal_strength += 1
        
        self.indicators['bollinger_bands'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'middle': round(self.df['bb_middle'].iloc[-1], 2) if not pd.isna(self.df['bb_middle'].iloc[-1]) else None,
                'upper': round(self.df['bb_upper'].iloc[-1], 2) if not pd.isna(self.df['bb_upper'].iloc[-1]) else None,
                'lower': round(self.df['bb_lower'].iloc[-1], 2) if not pd.isna(self.df['bb_lower'].iloc[-1]) else None,
                'percent_b': round(self.df['bb_pct_b'].iloc[-1], 2) if not pd.isna(self.df['bb_pct_b'].iloc[-1]) else None,
                'bandwidth': round(self.df['bb_bandwidth'].iloc[-1], 3) if not pd.isna(self.df['bb_bandwidth'].iloc[-1]) else None,
                'is_squeeze': is_squeeze,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Bollinger Bands',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_supertrend(self):
        """Calculate Supertrend indicator"""
        # Get Supertrend parameters
        period = self.params.get_indicator_param('supertrend_period')
        multiplier = self.params.get_indicator_param('supertrend_multiplier')
        
        # Calculate ATR
        if 'atr' not in self.df.columns:
            self.calculate_atr()
        
        # Calculate basic bands
        hl2 = (self.df['high'] + self.df['low']) / 2
        
        # Calculate upper and lower bands
        self.df['supertrend_basic_upper'] = hl2 + (multiplier * self.df['atr'])
        self.df['supertrend_basic_lower'] = hl2 - (multiplier * self.df['atr'])
        
        # Initialize Supertrend columns
        self.df['supertrend_upper'] = 0.0
        self.df['supertrend_lower'] = 0.0
        self.df['supertrend'] = 0.0
        self.df['supertrend_direction'] = 0  # 1 for bullish, -1 for bearish
        
        # Calculate Supertrend
        for i in range(period, len(self.df)):
            # Upper band
            if ((self.df['supertrend_basic_upper'].iloc[i] < self.df['supertrend_upper'].iloc[i-1]) or 
                (self.df['close'].iloc[i-1] > self.df['supertrend_upper'].iloc[i-1])):
                self.df.loc[self.df.index[i], 'supertrend_upper'] = self.df['supertrend_basic_upper'].iloc[i]
            else:
                self.df.loc[self.df.index[i], 'supertrend_upper'] = self.df['supertrend_upper'].iloc[i-1]
            
            # Lower band
            if ((self.df['supertrend_basic_lower'].iloc[i] > self.df['supertrend_lower'].iloc[i-1]) or 
                (self.df['close'].iloc[i-1] < self.df['supertrend_lower'].iloc[i-1])):
                self.df.loc[self.df.index[i], 'supertrend_lower'] = self.df['supertrend_basic_lower'].iloc[i]
            else:
                self.df.loc[self.df.index[i], 'supertrend_lower'] = self.df['supertrend_lower'].iloc[i-1]
            
            # Supertrend
            if (self.df['close'].iloc[i] > self.df['supertrend_upper'].iloc[i-1]):
                self.df.loc[self.df.index[i], 'supertrend'] = self.df['supertrend_lower'].iloc[i]
                self.df.loc[self.df.index[i], 'supertrend_direction'] = 1
            elif (self.df['close'].iloc[i] < self.df['supertrend_lower'].iloc[i-1]):
                self.df.loc[self.df.index[i], 'supertrend'] = self.df['supertrend_upper'].iloc[i]
                self.df.loc[self.df.index[i], 'supertrend_direction'] = -1
            else:
                self.df.loc[self.df.index[i], 'supertrend'] = self.df['supertrend'].iloc[i-1]
                self.df.loc[self.df.index[i], 'supertrend_direction'] = self.df['supertrend_direction'].iloc[i-1]
        
        # Generate signals for crossovers
        self.df['supertrend_buy_signal'] = ((self.df['supertrend_direction'] == 1) & 
                                         (self.df['supertrend_direction'].shift(1) == -1)).astype(int)
        self.df['supertrend_sell_signal'] = ((self.df['supertrend_direction'] == -1) & 
                                          (self.df['supertrend_direction'].shift(1) == 1)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['supertrend_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 3
        elif self.df['supertrend_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 3
        
        # Also check current direction if no recent crossover
        elif self.df['supertrend_direction'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Trend"
            signal_strength = 2
        elif self.df['supertrend_direction'].iloc[-1] == -1:
            current_signal = -1
            signal_type = "Bearish Trend"
            signal_strength = 2
        
        self.indicators['supertrend'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'supertrend': round(self.df['supertrend'].iloc[-1], 2) if not pd.isna(self.df['supertrend'].iloc[-1]) else None,
                'direction': 'Bullish' if self.df['supertrend_direction'].iloc[-1] == 1 else 'Bearish',
                'distance': round(abs(self.df['close'].iloc[-1] - self.df['supertrend'].iloc[-1]), 2) if not pd.isna(self.df['supertrend'].iloc[-1]) else None,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Supertrend',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_parabolic_sar(self):
        """Calculate Parabolic SAR"""
        # Get PSAR parameters
        af = self.params.get_indicator_param('psar_af')        # Acceleration Factor
        max_af = self.params.get_indicator_param('psar_max_af') # Maximum Acceleration Factor
        
        # Initialize variables
        high = self.df['high'].values
        low = self.df['low'].values
        close = self.df['close'].values
        
        # Initialize SAR array
        sar = np.zeros_like(close)
        trend = np.zeros_like(close)  # 1 for uptrend, -1 for downtrend
        ep = np.zeros_like(close)     # Extreme Point
        af_current = np.zeros_like(close)
        
        # Initial values
        trend[0] = 1  # Start with uptrend
        sar[0] = low[0]
        ep[0] = high[0]
        af_current[0] = af
        
        # Calculate PSAR for each period
        for i in range(1, len(close)):
            # Previous trend was uptrend
            if trend[i-1] == 1:
                # SAR can't be above prior period's low
                sar[i] = min(sar[i-1] + af_current[i-1] * (ep[i-1] - sar[i-1]), low[i-1], low[i-2] if i>1 else low[i-1])
                
                # Check if trend reverses
                if low[i] < sar[i]:
                    trend[i] = -1  # Switch to downtrend
                    sar[i] = ep[i-1]  # SAR becomes previous extreme point
                    ep[i] = low[i]  # New extreme point is current low
                    af_current[i] = af  # Reset acceleration factor
                else:
                    trend[i] = 1  # Continue uptrend
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]  # New extreme point
                        af_current[i] = min(af_current[i-1] + af, max_af)  # Increase acceleration factor
                    else:
                        ep[i] = ep[i-1]  # Keep previous extreme point
                        af_current[i] = af_current[i-1]  # Keep previous acceleration factor
            
            # Previous trend was downtrend
            else:
                # SAR can't be below prior period's high
                sar[i] = max(sar[i-1] + af_current[i-1] * (ep[i-1] - sar[i-1]), high[i-1], high[i-2] if i>1 else high[i-1])
                
                # Check if trend reverses
                if high[i] > sar[i]:
                    trend[i] = 1  # Switch to uptrend
                    sar[i] = ep[i-1]  # SAR becomes previous extreme point
                    ep[i] = high[i]  # New extreme point is current high
                    af_current[i] = af  # Reset acceleration factor
                else:
                    trend[i] = -1  # Continue downtrend
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]  # New extreme point
                        af_current[i] = min(af_current[i-1] + af, max_af)  # Increase acceleration factor
                    else:
                        ep[i] = ep[i-1]  # Keep previous extreme point
                        af_current[i] = af_current[i-1]  # Keep previous acceleration factor
        
        # Add calculated values to dataframe
        self.df['psar'] = sar
        self.df['psar_trend'] = trend
        
        # Generate signals for crossovers
        self.df['psar_buy_signal'] = ((self.df['psar_trend'] == 1) & 
                                    (self.df['psar_trend'].shift(1) == -1)).astype(int)
        self.df['psar_sell_signal'] = ((self.df['psar_trend'] == -1) & 
                                     (self.df['psar_trend'].shift(1) == 1)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['psar_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 2
        elif self.df['psar_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 2
        
        # Also check current trend
        elif self.df['psar_trend'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Trend"
            signal_strength = 1
        elif self.df['psar_trend'].iloc[-1] == -1:
            current_signal = -1
            signal_type = "Bearish Trend"
            signal_strength = 1
        
        self.indicators['parabolic_sar'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'psar': round(self.df['psar'].iloc[-1], 2),
                'trend': 'Bullish' if self.df['psar_trend'].iloc[-1] == 1 else 'Bearish',
                'distance': round(abs(self.df['close'].iloc[-1] - self.df['psar'].iloc[-1]), 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Parabolic SAR',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_atr(self):
        """Calculate Average True Range (ATR)"""
        # Get ATR parameters
        period = self.params.get_indicator_param('atr_period')
        
        # Calculate True Range
        high_low = self.df['high'] - self.df['low']
        high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
        low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
        
        # Take the maximum of the three
        self.df['tr'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR (simple moving average of TR for first 'period' values, then smoothed)
        self.df['atr'] = self.df['tr'].rolling(window=period).mean()
        
        # Calculate ATR percentage (relative to price)
        self.df['atr_pct'] = 100 * self.df['atr'] / self.df['close']
        
        # ATR doesn't generate signals directly, but provides volatility information
        atr_value = self.df['atr'].iloc[-1]
        atr_pct = self.df['atr_pct'].iloc[-1]
        
        # Determine if volatility is high (above average)
        avg_atr_pct = self.df['atr_pct'].rolling(window=20).mean().iloc[-1]
        high_volatility = atr_pct > (avg_atr_pct * 1.2)
        
        # Save values
        self.indicators['atr'] = {
            'signal': 0,  # ATR doesn't generate buy/sell signals directly
            'signal_strength': 0,
            'values': {
                'atr': round(atr_value, 2),
                'atr_pct': round(atr_pct, 2),
                'high_volatility': high_volatility,
            }
        }
    
    def calculate_atr_bands(self):
        """Calculate ATR Bands (similar to Keltner Channels)"""
        # Ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.calculate_atr()
            
        # Get parameters
        multiplier_upper = self.params.get_indicator_param('atr_bands')['multiplier_upper']
        multiplier_lower = self.params.get_indicator_param('atr_bands')['multiplier_lower']
        
        # Calculate ATR Bands
        self.df['atr_band_middle'] = self.df['close'].rolling(window=20).mean()  # 20-period SMA
        self.df['atr_band_upper'] = self.df['atr_band_middle'] + (multiplier_upper * self.df['atr'])
        self.df['atr_band_lower'] = self.df['atr_band_middle'] - (multiplier_lower * self.df['atr'])
        
        # Generate signals
        self.df['atr_band_touch_upper'] = self.df['high'] >= self.df['atr_band_upper']
        self.df['atr_band_touch_lower'] = self.df['low'] <= self.df['atr_band_lower']
        
        # Check for band touches in recent periods
        lookback = 5
        upper_touches = self.df['atr_band_touch_upper'].iloc[-lookback:].sum()
        lower_touches = self.df['atr_band_touch_lower'].iloc[-lookback:].sum()
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for recent touches
        if lower_touches > 0:
            current_signal = 1
            signal_type = "Price at Lower Band"
            signal_strength = 1
        elif upper_touches > 0:
            current_signal = -1
            signal_type = "Price at Upper Band"
            signal_strength = 1
        
        # Check if price is breaking out of bands
        if self.df['close'].iloc[-1] < self.df['atr_band_lower'].iloc[-1]:
            current_signal = 1
            signal_type = "Price Below Lower Band"
            signal_strength = 2
        elif self.df['close'].iloc[-1] > self.df['atr_band_upper'].iloc[-1]:
            current_signal = -1
            signal_type = "Price Above Upper Band"
            signal_strength = 2
        
        self.indicators['atr_bands'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'middle': round(self.df['atr_band_middle'].iloc[-1], 2),
                'upper': round(self.df['atr_band_upper'].iloc[-1], 2),
                'lower': round(self.df['atr_band_lower'].iloc[-1], 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'ATR Bands',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_adx(self):
        """Calculate Average Directional Index (ADX)"""
        # Get ADX parameters
        period = self.params.get_indicator_param('adx_period')
        threshold = self.params.get_indicator_param('adx_threshold')
        
        # Calculate True Range
        if 'tr' not in self.df.columns:
            high_low = self.df['high'] - self.df['low']
            high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
            low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
            self.df['tr'] = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate Directional Movement
        self.df['up_move'] = self.df['high'] - self.df['high'].shift(1)
        self.df['down_move'] = self.df['low'].shift(1) - self.df['low']
        
        # Calculate Positive (DM+) and Negative (DM-) Directional Movement
        self.df['dm_plus'] = np.where(
            (self.df['up_move'] > self.df['down_move']) & (self.df['up_move'] > 0),
            self.df['up_move'],
            0
        )
        
        self.df['dm_minus'] = np.where(
            (self.df['down_move'] > self.df['up_move']) & (self.df['down_move'] > 0),
            self.df['down_move'],
            0
        )
        
        # Calculate Smoothed Directional Movement and True Range
        self.df['tr_period'] = self.df['tr'].rolling(window=period).sum()
        self.df['dm_plus_period'] = self.df['dm_plus'].rolling(window=period).sum()
        self.df['dm_minus_period'] = self.df['dm_minus'].rolling(window=period).sum()
        
        # Calculate Directional Indicators (DI+ and DI-)
        self.df['di_plus'] = 100 * self.df['dm_plus_period'] / self.df['tr_period']
        self.df['di_minus'] = 100 * self.df['dm_minus_period'] / self.df['tr_period']
        
        # Calculate Directional Index (DX)
        self.df['dx'] = 100 * abs(self.df['di_plus'] - self.df['di_minus']) / (self.df['di_plus'] + self.df['di_minus'])
        
        # Calculate ADX (Average of DX)
        self.df['adx'] = self.df['dx'].rolling(window=period).mean()
        
        # Generate signals
        # Strong trend when ADX is above threshold
        self.df['adx_strong_trend'] = self.df['adx'] > threshold
        
        # Direction of trend based on DI+ vs DI-
        self.df['adx_trend_direction'] = np.where(
            self.df['di_plus'] > self.df['di_minus'],
            1,  # Bullish
            -1  # Bearish
        )
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check if we have a strong trend
        adx_value = self.df['adx'].iloc[-1]
        is_strong_trend = adx_value > threshold
        
        # Check the direction of the trend
        trend_direction = self.df['adx_trend_direction'].iloc[-1]
        
        # Check if DI+ crosses above DI- (buy signal)
        di_crossover_buy = (self.df['di_plus'].iloc[-1] > self.df['di_minus'].iloc[-1]) and \
                           (self.df['di_plus'].iloc[-2] <= self.df['di_minus'].iloc[-2])
        
        # Check if DI- crosses above DI+ (sell signal)
        di_crossover_sell = (self.df['di_minus'].iloc[-1] > self.df['di_plus'].iloc[-1]) and \
                            (self.df['di_minus'].iloc[-2] <= self.df['di_plus'].iloc[-2])
        
        # Generate signals
        if di_crossover_buy:
            current_signal = 1
            signal_type = "DI+ crossed above DI-"
            signal_strength = 2
        elif di_crossover_sell:
            current_signal = -1
            signal_type = "DI- crossed above DI+"
            signal_strength = 2
        elif is_strong_trend:
            if trend_direction == 1:
                current_signal = 1
                signal_type = "Strong Bullish Trend"
                signal_strength = 2
            else:
                current_signal = -1
                signal_type = "Strong Bearish Trend"
                signal_strength = 2
        
        self.indicators['adx'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'adx': round(adx_value, 2),
                'di_plus': round(self.df['di_plus'].iloc[-1], 2),
                'di_minus': round(self.df['di_minus'].iloc[-1], 2),
                'trend_strength': 'Strong' if adx_value > threshold else 'Weak',
                'trend_direction': 'Bullish' if trend_direction == 1 else 'Bearish',
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'ADX',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_aroon(self):
        """Calculate Aroon indicator"""
        # Get Aroon parameters
        period = self.params.get_indicator_param('aroon_period')
        uptrend_threshold = self.params.get_indicator_param('aroon_uptrend')
        downtrend_threshold = self.params.get_indicator_param('aroon_downtrend')
        
        # Calculate days since highest high and lowest low in period window
        # This is vectorized using rolling window operations
        
        # For each window, find the index of the maximum/minimum value
        rolling_high = self.df['high'].rolling(window=period)
        rolling_low = self.df['low'].rolling(window=period)
        
        # Aroon Up: 100 * ((period - days since highest high) / period)
        # Aroon Down: 100 * ((period - days since lowest low) / period)
        
        # For Aroon Up: Find the position of the max high within each window
        self.df['aroon_up'] = np.nan
        
        # For Aroon Down: Find the position of the min low within each window
        self.df['aroon_down'] = np.nan
        
        # Loop implementation for clarity - could be vectorized for larger data
        for i in range(period, len(self.df)):
            window_high = self.df['high'].iloc[i-period+1:i+1]
            window_low = self.df['low'].iloc[i-period+1:i+1]
            
            # Calculate days since highest high in window
            days_since_high = period - 1 - window_high.values.argmax()
            
            # Calculate days since lowest low in window
            days_since_low = period - 1 - window_low.values.argmin()
            
            # Calculate Aroon Up/Down
            self.df.loc[self.df.index[i], 'aroon_up'] = 100 * (period - days_since_high) / period
            self.df.loc[self.df.index[i], 'aroon_down'] = 100 * (period - days_since_low) / period
        
        # Calculate Aroon Oscillator
        self.df['aroon_oscillator'] = self.df['aroon_up'] - self.df['aroon_down']
        
        # Generate signals
        # Strong uptrend when Aroon Up > threshold and Aroon Down < threshold
        self.df['aroon_strong_uptrend'] = (
            (self.df['aroon_up'] > uptrend_threshold) & 
            (self.df['aroon_down'] < downtrend_threshold)
        )
        
        # Strong downtrend when Aroon Down > threshold and Aroon Up < threshold
        self.df['aroon_strong_downtrend'] = (
            (self.df['aroon_down'] > uptrend_threshold) & 
            (self.df['aroon_up'] < downtrend_threshold)
        )
        
        # Crossovers
        self.df['aroon_crossover'] = 0
        self.df.loc[self.df['aroon_up'] > self.df['aroon_down'], 'aroon_crossover'] = 1
        self.df.loc[self.df['aroon_up'] < self.df['aroon_down'], 'aroon_crossover'] = -1
        
        # Detect bullish crossover (Aroon Up crosses above Aroon Down)
        self.df['aroon_buy_signal'] = (
            (self.df['aroon_crossover'] == 1) & 
            (self.df['aroon_crossover'].shift(1) == -1)
        ).astype(int)
        
        # Detect bearish crossover (Aroon Down crosses above Aroon Up)
        self.df['aroon_sell_signal'] = (
            (self.df['aroon_crossover'] == -1) & 
            (self.df['aroon_crossover'].shift(1) == 1)
        ).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['aroon_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 2
        elif self.df['aroon_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 2
        
        # Check for strong trends
        elif self.df['aroon_strong_uptrend'].iloc[-1]:
            current_signal = 1
            signal_type = "Strong Uptrend"
            signal_strength = 3
        elif self.df['aroon_strong_downtrend'].iloc[-1]:
            current_signal = -1
            signal_type = "Strong Downtrend"
            signal_strength = 3
        
        self.indicators['aroon'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'aroon_up': round(self.df['aroon_up'].iloc[-1], 2),
                'aroon_down': round(self.df['aroon_down'].iloc[-1], 2),
                'aroon_oscillator': round(self.df['aroon_oscillator'].iloc[-1], 2),
                'strong_uptrend': self.df['aroon_strong_uptrend'].iloc[-1],
                'strong_downtrend': self.df['aroon_strong_downtrend'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Aroon',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['obv'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Calculate OBV
        obv = np.zeros(len(self.df))
        obv[0] = self.df['volume'].iloc[0]
        
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + self.df['volume'].iloc[i]
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - self.df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        self.df['obv'] = obv
        
        # Calculate OBV moving average
        self.df['obv_ma'] = self.df['obv'].rolling(window=20).mean()
        
        # Generate signals
        # Signal based on OBV vs its moving average
        self.df['obv_signal'] = 0
        self.df.loc[self.df['obv'] > self.df['obv_ma'], 'obv_signal'] = 1
        self.df.loc[self.df['obv'] < self.df['obv_ma'], 'obv_signal'] = -1
        
        # Detect crossovers
        self.df['obv_buy_signal'] = ((self.df['obv_signal'] == 1) & 
                                   (self.df['obv_signal'].shift(1) == -1)).astype(int)
        self.df['obv_sell_signal'] = ((self.df['obv_signal'] == -1) & 
                                    (self.df['obv_signal'].shift(1) == 1)).astype(int)
        
        # Check for divergence (simplified calculation)
        # Bullish divergence: Price making lower lows but OBV making higher lows
        # Bearish divergence: Price making higher highs but OBV making lower highs
        
        # Get last 20 periods for local analysis
        last_n = 20
        if len(self.df) > last_n:
            subset = self.df.iloc[-last_n:]
            
            # Get local min/max for price and OBV
            price_min_idx = subset['close'].idxmin()
            price_max_idx = subset['close'].idxmax()
            obv_min_idx = subset['obv'].idxmin()
            obv_max_idx = subset['obv'].idxmax()
            
            # Check if the timing of min/max values shows divergence
            bullish_div = price_min_idx > obv_min_idx and self.df['close'].iloc[-1] < self.df['close'].iloc[-10].mean()
            bearish_div = price_max_idx > obv_max_idx and self.df['close'].iloc[-1] > self.df['close'].iloc[-10].mean()
        else:
            bullish_div = False
            bearish_div = False
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['obv_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover"
            signal_strength = 2
        elif self.df['obv_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover"
            signal_strength = 2
        
        # Check for divergence (stronger signals)
        elif bullish_div:
            current_signal = 1
            signal_type = "Bullish Divergence"
            signal_strength = 3
        elif bearish_div:
            current_signal = -1
            signal_type = "Bearish Divergence"
            signal_strength = 3
        
        # Also check if OBV is trending
        obv_rising = self.df['obv'].iloc[-1] > self.df['obv'].iloc[-5]
        
        self.indicators['obv'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'obv': int(self.df['obv'].iloc[-1]),
                'obv_ma': int(self.df['obv_ma'].iloc[-1]) if not pd.isna(self.df['obv_ma'].iloc[-1]) else None,
                'rising': obv_rising,
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'On-Balance Volume',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_vwap(self):
        """Calculate Volume Weighted Average Price (VWAP)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['vwap'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Get parameters
        reset_period = self.params.get_indicator_param('vwap')['reset_period']
        
        # Calculate typical price
        self.df['typical_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Calculate volume * typical price
        self.df['vol_tp'] = self.df['typical_price'] * self.df['volume']
        
        # Reset cumulative values at the start of each period
        if reset_period == 'day':
            # Check if index includes date information
            if pd.api.types.is_datetime64_any_dtype(self.df.index):
                # Create date groups
                self.df['date_group'] = self.df.index.date
                
                # Calculate group-wise VWAP
                groups = self.df.groupby('date_group')
                
                # Initialize VWAP column
                self.df['vwap'] = 0.0
                
                # Calculate VWAP for each group
                for name, group in groups:
                    cumulative_vol_tp = group['vol_tp'].cumsum()
                    cumulative_vol = group['volume'].cumsum()
                    
                    # Avoid division by zero
                    mask = cumulative_vol > 0
                    group_vwap = pd.Series(np.where(mask, cumulative_vol_tp / cumulative_vol, 0), index=group.index)
                    
                    # Assign VWAP values back to main dataframe
                    self.df.loc[group.index, 'vwap'] = group_vwap
                
                # Remove the date_group column
                self.df.drop('date_group', axis=1, inplace=True)
            else:
                # If index doesn't have date information, use a simple approach
                self.df['vwap'] = self.df['vol_tp'].cumsum() / self.df['volume'].cumsum()
        else:
            # No reset, calculate cumulative VWAP
            self.df['vwap'] = self.df['vol_tp'].cumsum() / self.df['volume'].cumsum()
        
        # Generate signals
        self.df['price_above_vwap'] = self.df['close'] > self.df['vwap']
        
        # Detect crosses above/below VWAP
        self.df['vwap_cross_above'] = ((self.df['close'] > self.df['vwap']) & 
                                      (self.df['close'].shift(1) <= self.df['vwap'].shift(1))).astype(int)
        self.df['vwap_cross_below'] = ((self.df['close'] < self.df['vwap']) & 
                                      (self.df['close'].shift(1) >= self.df['vwap'].shift(1))).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['vwap_cross_above'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Price Crossed Above VWAP"
            signal_strength = 2
        elif self.df['vwap_cross_below'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Price Crossed Below VWAP"
            signal_strength = 2
        
        # Check price position relative to VWAP
        elif self.df['price_above_vwap'].iloc[-1]:
            current_signal = 1
            signal_type = "Price Above VWAP"
            signal_strength = 1
        else:
            current_signal = -1
            signal_type = "Price Below VWAP"
            signal_strength = 1
        
        # Confirmation by volume
        if self.df['high_volume'].iloc[-1] if 'high_volume' in self.df.columns else False:
            signal_type += " with High Volume"
            signal_strength += 1
        
        self.indicators['vwap'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'vwap': round(self.df['vwap'].iloc[-1], 2),
                'price_to_vwap': round(self.df['close'].iloc[-1] / self.df['vwap'].iloc[-1], 3),
                'crossed_above': self.df['vwap_cross_above'].iloc[-1] == 1,
                'crossed_below': self.df['vwap_cross_below'].iloc[-1] == 1,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'VWAP',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_stochastic_rsi(self):
        """Calculate Stochastic RSI"""
        # Ensure RSI is calculated
        if 'rsi' not in self.df.columns:
            self.calculate_rsi()
            
        # Get parameters
        period = self.params.get_indicator_param('stochastic_rsi')['period']
        smooth_k = self.params.get_indicator_param('stochastic_rsi')['smooth_k']
        smooth_d = self.params.get_indicator_param('stochastic_rsi')['smooth_d']
        oversold = self.params.get_indicator_param('stochastic_rsi')['oversold']
        overbought = self.params.get_indicator_param('stochastic_rsi')['overbought']
        
        # Calculate Stochastic RSI
        # First find the lowest low and highest high of RSI within the period
        lowest_rsi = self.df['rsi'].rolling(window=period).min()
        highest_rsi = self.df['rsi'].rolling(window=period).max()
        
        # Calculate raw K (current RSI relative to its range)
        self.df['stoch_rsi_k_raw'] = 100 * (self.df['rsi'] - lowest_rsi) / (highest_rsi - lowest_rsi)
        
        # Smooth K
        self.df['stoch_rsi_k'] = self.df['stoch_rsi_k_raw'].rolling(window=smooth_k).mean()
        
        # Calculate D (moving average of K)
        self.df['stoch_rsi_d'] = self.df['stoch_rsi_k'].rolling(window=smooth_d).mean()
        
        # Generate signals
        self.df['stoch_rsi_oversold'] = self.df['stoch_rsi_k'] < oversold
        self.df['stoch_rsi_overbought'] = self.df['stoch_rsi_k'] > overbought
        
        # Detect K crossing above D from oversold region
        self.df['stoch_rsi_buy_signal'] = ((self.df['stoch_rsi_k'] > self.df['stoch_rsi_d']) & 
                                         (self.df['stoch_rsi_k'].shift(1) <= self.df['stoch_rsi_d'].shift(1)) &
                                         (self.df['stoch_rsi_k'] < 30)).astype(int)
        
        # Detect K crossing below D from overbought region
        self.df['stoch_rsi_sell_signal'] = ((self.df['stoch_rsi_k'] < self.df['stoch_rsi_d']) & 
                                          (self.df['stoch_rsi_k'].shift(1) >= self.df['stoch_rsi_d'].shift(1)) &
                                          (self.df['stoch_rsi_k'] > 70)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers in oversold/overbought regions
        if self.df['stoch_rsi_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Bullish Crossover from Oversold"
            signal_strength = 3
        elif self.df['stoch_rsi_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Bearish Crossover from Overbought"
            signal_strength = 3
        
        # Also check for extreme oversold/overbought conditions
        elif self.df['stoch_rsi_oversold'].iloc[-1] and self.df['stoch_rsi_k'].iloc[-1] < 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 2
        elif self.df['stoch_rsi_overbought'].iloc[-1] and self.df['stoch_rsi_k'].iloc[-1] > 90:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 2
        
        self.indicators['stochastic_rsi'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'k': round(self.df['stoch_rsi_k'].iloc[-1], 2) if not pd.isna(self.df['stoch_rsi_k'].iloc[-1]) else None,
                'd': round(self.df['stoch_rsi_d'].iloc[-1], 2) if not pd.isna(self.df['stoch_rsi_d'].iloc[-1]) else None,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['stoch_rsi_oversold'].iloc[-1] if not pd.isna(self.df['stoch_rsi_oversold'].iloc[-1]) else False,
                'is_overbought': self.df['stoch_rsi_overbought'].iloc[-1] if not pd.isna(self.df['stoch_rsi_overbought'].iloc[-1]) else False,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Stochastic RSI',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_rate_of_change(self):
        """Calculate Rate of Change (ROC)"""
        # Get parameters
        period = self.params.get_indicator_param('roc_period')
        
        # Calculate ROC: ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100
        self.df['roc'] = ((self.df['close'] - self.df['close'].shift(period)) / 
                         self.df['close'].shift(period)) * 100
        
        # Generate signals
        self.df['roc_signal'] = 0
        self.df.loc[self.df['roc'] > 0, 'roc_signal'] = 1
        self.df.loc[self.df['roc'] < 0, 'roc_signal'] = -1
        
        # Detect crossovers
        self.df['roc_buy_signal'] = ((self.df['roc_signal'] == 1) & 
                                    (self.df['roc_signal'].shift(1) == -1)).astype(int)
        self.df['roc_sell_signal'] = ((self.df['roc_signal'] == -1) & 
                                     (self.df['roc_signal'].shift(1) == 1)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['roc_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Crossed Above Zero"
            signal_strength = 2
        elif self.df['roc_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Crossed Below Zero"
            signal_strength = 2
        
        # Check for extreme values
        roc_value = self.df['roc'].iloc[-1]
        if roc_value > 10:
            current_signal = -1  # Potentially overbought
            signal_type = "Extremely High Value"
            signal_strength = 3
        elif roc_value < -10:
            current_signal = 1   # Potentially oversold
            signal_type = "Extremely Low Value"
            signal_strength = 3
        
        self.indicators['roc'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'roc': round(roc_value, 2),
                'trend': 'Bullish' if roc_value > 0 else 'Bearish',
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Rate of Change',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_williams_r(self):
        """Calculate Williams %R"""
        # Get parameters
        period = self.params.get_indicator_param('williams_r')['period']
        oversold = self.params.get_indicator_param('williams_r')['oversold']
        overbought = self.params.get_indicator_param('williams_r')['overbought']
        
        # Calculate highest high and lowest low over period
        highest_high = self.df['high'].rolling(window=period).max()
        lowest_low = self.df['low'].rolling(window=period).min()
        
        # Calculate Williams %R: ((highest_high - close) / (highest_high - lowest_low)) * -100
        self.df['williams_r'] = ((highest_high - self.df['close']) / 
                                (highest_high - lowest_low)) * -100
        
        # Generate signals
        self.df['williams_oversold'] = self.df['williams_r'] <= oversold
        self.df['williams_overbought'] = self.df['williams_r'] >= overbought
        
        # Detect crosses from oversold/overbought regions
        self.df['williams_buy_signal'] = ((self.df['williams_r'] > oversold) & 
                                        (self.df['williams_r'].shift(1) <= oversold)).astype(int)
        self.df['williams_sell_signal'] = ((self.df['williams_r'] < overbought) & 
                                         (self.df['williams_r'].shift(1) >= overbought)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crosses from oversold/overbought regions
        if self.df['williams_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Cross from Oversold"
            signal_strength = 2
        elif self.df['williams_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Cross from Overbought"
            signal_strength = 2
        
        # Check for extreme conditions
        elif self.df['williams_oversold'].iloc[-1] and self.df['williams_r'].iloc[-1] < oversold - 10:
            current_signal = 1
            signal_type = "Extremely Oversold"
            signal_strength = 1
        elif self.df['williams_overbought'].iloc[-1] and self.df['williams_r'].iloc[-1] > overbought + 10:
            current_signal = -1
            signal_type = "Extremely Overbought"
            signal_strength = 1
        
        self.indicators['williams_r'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'williams_r': round(self.df['williams_r'].iloc[-1], 2),
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['williams_oversold'].iloc[-1],
                'is_overbought': self.df['williams_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Williams %R',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_ultimate_oscillator(self):
        """Calculate Ultimate Oscillator"""
        # Get parameters
        period1 = self.params.get_indicator_param('ultimate_oscillator')['period1']
        period2 = self.params.get_indicator_param('ultimate_oscillator')['period2']
        period3 = self.params.get_indicator_param('ultimate_oscillator')['period3']
        weight1 = self.params.get_indicator_param('ultimate_oscillator')['weight1']
        weight2 = self.params.get_indicator_param('ultimate_oscillator')['weight2']
        weight3 = self.params.get_indicator_param('ultimate_oscillator')['weight3']
        oversold = self.params.get_indicator_param('ultimate_oscillator')['oversold']
        overbought = self.params.get_indicator_param('ultimate_oscillator')['overbought']
        
        # Calculate buying pressure (close - min(low, prior_close))
        self.df['bp'] = self.df['close'] - self.df[['low', 'close']].shift(1).min(axis=1)
        
        # Calculate true range (max(high, prior_close) - min(low, prior_close))
        self.df['tr'] = self.df[['high', 'close']].shift(1).max(axis=1) - self.df[['low', 'close']].shift(1).min(axis=1)
        
        # Calculate average buying pressure for each period
        self.df['avg_bp1'] = self.df['bp'].rolling(window=period1).sum()
        self.df['avg_bp2'] = self.df['bp'].rolling(window=period2).sum()
        self.df['avg_bp3'] = self.df['bp'].rolling(window=period3).sum()
        
        # Calculate average true range for each period
        self.df['avg_tr1'] = self.df['tr'].rolling(window=period1).sum()
        self.df['avg_tr2'] = self.df['tr'].rolling(window=period2).sum()
        self.df['avg_tr3'] = self.df['tr'].rolling(window=period3).sum()
        
        # Calculate the three raw components
        # Avoid division by zero
        self.df['raw_uosc1'] = np.where(self.df['avg_tr1'] != 0, 100 * self.df['avg_bp1'] / self.df['avg_tr1'], 50)
        self.df['raw_uosc2'] = np.where(self.df['avg_tr2'] != 0, 100 * self.df['avg_bp2'] / self.df['avg_tr2'], 50)
        self.df['raw_uosc3'] = np.where(self.df['avg_tr3'] != 0, 100 * self.df['avg_bp3'] / self.df['avg_tr3'], 50)
        
        # Calculate Ultimate Oscillator
        total_weight = weight1 + weight2 + weight3
        self.df['uosc'] = (weight1 * self.df['raw_uosc1'] + 
                          weight2 * self.df['raw_uosc2'] + 
                          weight3 * self.df['raw_uosc3']) / total_weight
        
        # Generate signals
        self.df['uosc_oversold'] = self.df['uosc'] < oversold
        self.df['uosc_overbought'] = self.df['uosc'] > overbought
        
        # Detect crosses from oversold/overbought regions
        self.df['uosc_buy_signal'] = ((self.df['uosc'] > oversold) & 
                                    (self.df['uosc'].shift(1) <= oversold)).astype(int)
        self.df['uosc_sell_signal'] = ((self.df['uosc'] < overbought) & 
                                     (self.df['uosc'].shift(1) >= overbought)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crosses from oversold/overbought regions
        if self.df['uosc_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Cross from Oversold"
            signal_strength = 2
        elif self.df['uosc_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Cross from Overbought"
            signal_strength = 2
        
        # Check for bullish/bearish divergence (basic check)
        uosc_rising = self.df['uosc'].iloc[-1] > self.df['uosc'].iloc[-5]
        price_rising = self.df['close'].iloc[-1] > self.df['close'].iloc[-5]
        
        if uosc_rising and not price_rising:
            current_signal = 1
            signal_type = "Bullish Divergence"
            signal_strength = 3
        elif not uosc_rising and price_rising:
            current_signal = -1
            signal_type = "Bearish Divergence"
            signal_strength = 3
        
        self.indicators['ultimate_oscillator'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'uosc': round(self.df['uosc'].iloc[-1], 2),
                'oversold_threshold': oversold,
                'overbought_threshold': overbought,
                'is_oversold': self.df['uosc_oversold'].iloc[-1],
                'is_overbought': self.df['uosc_overbought'].iloc[-1],
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Ultimate Oscillator',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_cmf(self):
        """Calculate Chaikin Money Flow (CMF)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['cmf'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Get parameters
        period = self.params.get_indicator_param('cmf')['period']
        signal_threshold = self.params.get_indicator_param('cmf')['signal_threshold']
        
        # Calculate Money Flow Multiplier (MFM)
        # MFM = ((close - low) - (high - close)) / (high - low)
        range = self.df['high'] - self.df['low']
        self.df['mfm'] = np.where(
            range != 0,
            ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / range,
            0
        )
        
        # Calculate Money Flow Volume (MFV)
        self.df['mfv'] = self.df['mfm'] * self.df['volume']
        
        # Calculate Chaikin Money Flow
        # CMF = Sum(MFV) over period / Sum(Volume) over period
        self.df['cmf'] = (self.df['mfv'].rolling(window=period).sum() / 
                         self.df['volume'].rolling(window=period).sum())
        
        # Generate signals based on CMF crossing zero
        self.df['cmf_signal'] = 0
        self.df.loc[self.df['cmf'] > signal_threshold, 'cmf_signal'] = 1
        self.df.loc[self.df['cmf'] < -signal_threshold, 'cmf_signal'] = -1
        
        # Detect crossovers
        self.df['cmf_buy_signal'] = ((self.df['cmf'] > signal_threshold) & 
                                   (self.df['cmf'].shift(1) <= signal_threshold)).astype(int)
        self.df['cmf_sell_signal'] = ((self.df['cmf'] < -signal_threshold) & 
                                    (self.df['cmf'].shift(1) >= -signal_threshold)).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for crossovers
        if self.df['cmf_buy_signal'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Crossed Above Threshold"
            signal_strength = 2
        elif self.df['cmf_sell_signal'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Crossed Below Threshold"
            signal_strength = 2
        
        # Check current CMF value
        cmf_value = self.df['cmf'].iloc[-1]
        if cmf_value > 0.2:
            current_signal = 1
            signal_type = "Strong Positive Flow"
            signal_strength = 3
        elif cmf_value < -0.2:
            current_signal = -1
            signal_type = "Strong Negative Flow"
            signal_strength = 3
        
        self.indicators['cmf'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'cmf': round(cmf_value, 3),
                'threshold': signal_threshold,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Chaikin Money Flow',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_alligator(self):
        """Calculate Williams Alligator indicator"""
        # Get parameters
        jaw_period = self.params.get_indicator_param('alligator_jaw')
        jaw_offset = self.params.get_indicator_param('alligator_jaw_offset')
        teeth_period = self.params.get_indicator_param('alligator_teeth')
        teeth_offset = self.params.get_indicator_param('alligator_teeth_offset')
        lips_period = self.params.get_indicator_param('alligator_lips')
        lips_offset = self.params.get_indicator_param('alligator_lips_offset')
        
        # Calculate median price
        self.df['median_price'] = (self.df['high'] + self.df['low']) / 2
        
        # Calculate jaw (blue line)
        self.df['jaw'] = self.df['median_price'].rolling(window=jaw_period).mean().shift(jaw_offset)
        
        # Calculate teeth (red line)
        self.df['teeth'] = self.df['median_price'].rolling(window=teeth_period).mean().shift(teeth_offset)
        
        # Calculate lips (green line)
        self.df['lips'] = self.df['median_price'].rolling(window=lips_period).mean().shift(lips_offset)
        
        # Generate signals
        
        # Alligator sleeping: lines are intertwined (jaw, teeth, and lips are close together)
        self.df['alligator_sleeping'] = (
            (abs(self.df['jaw'] - self.df['teeth']) < 0.03 * self.df['close']) & 
            (abs(self.df['teeth'] - self.df['lips']) < 0.03 * self.df['close'])
        )
        
        # Alligator awakening: lips cross above teeth and then teeth cross above jaw
        lips_cross_above_teeth = ((self.df['lips'] > self.df['teeth']) & 
                                (self.df['lips'].shift(1) <= self.df['teeth'].shift(1)))
        teeth_cross_above_jaw = ((self.df['teeth'] > self.df['jaw']) & 
                               (self.df['teeth'].shift(1) <= self.df['jaw'].shift(1)))
        
        self.df['alligator_awakening_bullish'] = lips_cross_above_teeth & teeth_cross_above_jaw
        
        # Alligator eating: lines are properly aligned (lips > teeth > jaw)
        self.df['alligator_eating_bullish'] = (
            (self.df['lips'] > self.df['teeth']) & 
            (self.df['teeth'] > self.df['jaw'])
        )
        
        # Alligator going to sleep: lines start to intertwine after being aligned
        self.df['alligator_sated'] = (
            ~self.df['alligator_eating_bullish'] & 
            self.df['alligator_eating_bullish'].shift(1)
        )
        
        # Bearish equivalents
        lips_cross_below_teeth = ((self.df['lips'] < self.df['teeth']) & 
                                (self.df['lips'].shift(1) >= self.df['teeth'].shift(1)))
        teeth_cross_below_jaw = ((self.df['teeth'] < self.df['jaw']) & 
                               (self.df['teeth'].shift(1) >= self.df['jaw'].shift(1)))
        
        self.df['alligator_awakening_bearish'] = lips_cross_below_teeth & teeth_cross_below_jaw
        
        self.df['alligator_eating_bearish'] = (
            (self.df['lips'] < self.df['teeth']) & 
            (self.df['teeth'] < self.df['jaw'])
        )
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for alligator states
        if self.df['alligator_awakening_bullish'].iloc[-1]:
            current_signal = 1
            signal_type = "Alligator Awakening (Bullish)"
            signal_strength = 3
        elif self.df['alligator_awakening_bearish'].iloc[-1]:
            current_signal = -1
            signal_type = "Alligator Awakening (Bearish)"
            signal_strength = 3
        elif self.df['alligator_eating_bullish'].iloc[-1]:
            current_signal = 1
            signal_type = "Alligator Eating (Bullish)"
            signal_strength = 2
        elif self.df['alligator_eating_bearish'].iloc[-1]:
            current_signal = -1
            signal_type = "Alligator Eating (Bearish)"
            signal_strength = 2
        elif self.df['alligator_sated'].iloc[-1]:
            # No signal when alligator is going to sleep (end of trend)
            signal_type = "Alligator Sated (End of Trend)"
        elif self.df['alligator_sleeping'].iloc[-1]:
            signal_type = "Alligator Sleeping (No Trend)"
        
        self.indicators['alligator'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'jaw': round(self.df['jaw'].iloc[-1], 2) if not pd.isna(self.df['jaw'].iloc[-1]) else None,
                'teeth': round(self.df['teeth'].iloc[-1], 2) if not pd.isna(self.df['teeth'].iloc[-1]) else None,
                'lips': round(self.df['lips'].iloc[-1], 2) if not pd.isna(self.df['lips'].iloc[-1]) else None,
                'state': signal_type,
                'is_sleeping': self.df['alligator_sleeping'].iloc[-1],
                'is_eating_bullish': self.df['alligator_eating_bullish'].iloc[-1],
                'is_eating_bearish': self.df['alligator_eating_bearish'].iloc[-1]
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Alligator',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_cpr(self):
        """Calculate Central Pivot Range (CPR)"""
        # Get parameters
        use_previous_day = self.params.get_indicator_param('cpr_use_previous_day')
        
        # Initialize CPR columns
        self.df['pivot'] = np.nan
        self.df['bc'] = np.nan  # Bottom Central Pivot
        self.df['tc'] = np.nan  # Top Central Pivot
        
        # Calculate pivot points for each day (or period)
        # If index is datetime, use date grouping
        if pd.api.types.is_datetime64_any_dtype(self.df.index):
            # Group by date
            date_groups = self.df.groupby(self.df.index.date)
            
            for date, group in date_groups:
                # Calculate pivot points for this day
                high = group['high'].max()
                low = group['low'].min()
                close = group['close'].iloc[-1]
                
                # Central Pivot Range calculation
                pivot = (high + low + close) / 3
                bc = (high + low) / 2
                tc = (pivot - bc) + pivot
                
                # Get next day's index if available
                next_day_index = self.df.index[self.df.index.date > date]
                
                if use_previous_day and len(next_day_index) > 0:
                    # Assign to next day's rows
                    self.df.loc[next_day_index, 'pivot'] = pivot
                    self.df.loc[next_day_index, 'bc'] = bc
                    self.df.loc[next_day_index, 'tc'] = tc
                else:
                    # Assign to current day's rows
                    day_index = self.df.index[self.df.index.date == date]
                    self.df.loc[day_index, 'pivot'] = pivot
                    self.df.loc[day_index, 'bc'] = bc
                    self.df.loc[day_index, 'tc'] = tc
        else:
            # If not datetime index, use a rolling window approach
            window_size = 20  # Default window size
            
            for i in range(window_size, len(self.df)):
                # Get data from previous window
                prev_window = self.df.iloc[i-window_size:i]
                high = prev_window['high'].max()
                low = prev_window['low'].min()
                close = prev_window['close'].iloc[-1]
                
                # Calculate CPR
                pivot = (high + low + close) / 3
                bc = (high + low) / 2
                tc = (pivot - bc) + pivot
                
                # Assign to current row
                self.df.loc[self.df.index[i], 'pivot'] = pivot
                self.df.loc[self.df.index[i], 'bc'] = bc
                self.df.loc[self.df.index[i], 'tc'] = tc
        
        # Generate signals
        
        # Calculate width of CPR
        self.df['cpr_width'] = (self.df['tc'] - self.df['bc']) / self.df['pivot'] * 100
        
        # Narrow CPR indicates potential breakout
        self.df['narrow_cpr'] = self.df['cpr_width'] < (self.df['cpr_width'].rolling(window=20).mean() * 0.7)
        
        # Wide CPR indicates potential volatility
        self.df['wide_cpr'] = self.df['cpr_width'] > (self.df['cpr_width'].rolling(window=20).mean() * 1.3)
        
        # Price breaking above/below CPR
        self.df['break_above_cpr'] = ((self.df['close'] > self.df['tc']) & 
                                     (self.df['close'].shift(1) <= self.df['tc'])).astype(int)
        self.df['break_below_cpr'] = ((self.df['close'] < self.df['bc']) & 
                                     (self.df['close'].shift(1) >= self.df['bc'])).astype(int)
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Check for breakouts
        if self.df['break_above_cpr'].iloc[-1] == 1:
            current_signal = 1
            signal_type = "Price Broke Above CPR"
            signal_strength = 2
        elif self.df['break_below_cpr'].iloc[-1] == 1:
            current_signal = -1
            signal_type = "Price Broke Below CPR"
            signal_strength = 2
        
        # Check price position relative to CPR
        else:
            if not pd.isna(self.df['tc'].iloc[-1]) and not pd.isna(self.df['bc'].iloc[-1]):
                if self.df['close'].iloc[-1] > self.df['tc'].iloc[-1]:
                    current_signal = 1
                    signal_type = "Price Above CPR"
                    signal_strength = 1
                elif self.df['close'].iloc[-1] < self.df['bc'].iloc[-1]:
                    current_signal = -1
                    signal_type = "Price Below CPR"
                    signal_strength = 1
                else:
                    signal_type = "Price Within CPR"
        
        # Check for narrow CPR (potential for breakout)
        if self.df['narrow_cpr'].iloc[-1]:
            signal_type += " (Narrow CPR)"
            signal_strength += 1
        
        self.indicators['cpr'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'pivot': round(self.df['pivot'].iloc[-1], 2) if not pd.isna(self.df['pivot'].iloc[-1]) else None,
                'bc': round(self.df['bc'].iloc[-1], 2) if not pd.isna(self.df['bc'].iloc[-1]) else None,
                'tc': round(self.df['tc'].iloc[-1], 2) if not pd.isna(self.df['tc'].iloc[-1]) else None,
                'width': round(self.df['cpr_width'].iloc[-1], 2) if not pd.isna(self.df['cpr_width'].iloc[-1]) else None,
                'is_narrow': self.df['narrow_cpr'].iloc[-1] if not pd.isna(self.df['narrow_cpr'].iloc[-1]) else False,
                'is_wide': self.df['wide_cpr'].iloc[-1] if not pd.isna(self.df['wide_cpr'].iloc[-1]) else False,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Central Pivot Range',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_volume_profile(self):
        """Calculate Volume Profile (simplified)"""
        # Check if we have volume data
        if 'volume' not in self.df.columns or self.df['volume'].sum() == 0:
            self.indicators['volume_profile'] = {'signal': 0, 'error': 'No volume data available'}
            return
        
        # Get parameters
        period = self.params.get_indicator_param('volume_profile')['period']
        
        # Get recent subset of data
        recent_data = self.df.iloc[-period:]
        
        # Create price bins
        price_range = recent_data['high'].max() - recent_data['low'].min()
        bin_size = price_range / 10  # 10 price bins
        
        price_bins = np.arange(
            recent_data['low'].min(), 
            recent_data['high'].max() + bin_size, 
            bin_size
        )
        
        # Assign each candle to a price bin
        # Using typical price (high + low + close) / 3
        recent_data['typical_price'] = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        recent_data['price_bin'] = pd.cut(recent_data['typical_price'], bins=price_bins, labels=False)
        
        # Calculate volume per price bin
        volume_profile = recent_data.groupby('price_bin')['volume'].sum()
        
        # Find Point of Control (POC) - price level with highest volume
        poc_bin = volume_profile.idxmax()
        poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
        
        # Define Value Area (70% of total volume)
        total_volume = volume_profile.sum()
        value_area_volume = total_volume * 0.7
        
        # Sort bins by volume (descending)
        sorted_bins = volume_profile.sort_values(ascending=False)
        
        # Take bins until we reach 70% of total volume
        cumulative_volume = 0
        value_area_bins = []
        
        for bin_idx, bin_volume in sorted_bins.items():
            cumulative_volume += bin_volume
            value_area_bins.append(bin_idx)
            if cumulative_volume >= value_area_volume:
                break
        
        # Define Value Area High (VAH) and Value Area Low (VAL)
        vah_bin = max(value_area_bins)
        val_bin = min(value_area_bins)
        
        vah_price = (price_bins[vah_bin] + price_bins[vah_bin + 1]) / 2
        val_price = (price_bins[val_bin] + price_bins[val_bin + 1]) / 2
        
        # Generate signals
        current_price = self.df['close'].iloc[-1]
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Price at extreme above Value Area High
        if current_price > vah_price * 1.03:
            current_signal = -1
            signal_type = "Price Above Value Area"
            signal_strength = 2
        
        # Price at extreme below Value Area Low
        elif current_price < val_price * 0.97:
            current_signal = 1
            signal_type = "Price Below Value Area"
            signal_strength = 2
        
        # Price at Point of Control
        elif abs(current_price - poc_price) / poc_price < 0.005:
            signal_type = "Price at Point of Control"
            # This is usually neutral
        
        # If price and POC are close, no strong signal
        # But if we have a strong trend (use ADX or similar), can use POC as support/resistance
        
        self.indicators['volume_profile'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'poc': round(poc_price, 2),
                'vah': round(vah_price, 2),
                'val': round(val_price, 2),
                'current_price': round(current_price, 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Volume Profile',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_support_resistance(self):
        """Calculate support and resistance levels"""
        # Get parameters
        period = self.params.get_indicator_param('support_resistance')['pivot_period']
        pivot_threshold = self.params.get_indicator_param('support_resistance')['pivot_threshold']
        
        # Find pivot highs and lows
        # A pivot high is a high that is higher than surrounding highs within window
        # A pivot low is a low that is lower than surrounding lows within window
        
        # Initialize arrays for pivot highs and lows
        pivots_high = []
        pivots_low = []
        
        # Find pivot highs
        for i in range(period, len(self.df) - period):
            window_left = self.df['high'].iloc[i - period:i]
            window_right = self.df['high'].iloc[i + 1:i + period + 1]
            
            current_high = self.df['high'].iloc[i]
            
            if (current_high > window_left.max()) and (current_high > window_right.max()):
                pivots_high.append((self.df.index[i], current_high))
        
        # Find pivot lows
        for i in range(period, len(self.df) - period):
            window_left = self.df['low'].iloc[i - period:i]
            window_right = self.df['low'].iloc[i + 1:i + period + 1]
            
            current_low = self.df['low'].iloc[i]
            
            if (current_low < window_left.min()) and (current_low < window_right.min()):
                pivots_low.append((self.df.index[i], current_low))
        
        # Group pivot points that are close to each other (clustering)
        def cluster_levels(levels, threshold):
            if not levels:
                return []
            
            clustered = []
            current_cluster = [levels[0]]
            
            for i in range(1, len(levels)):
                if abs(levels[i][1] - current_cluster[0][1]) / current_cluster[0][1] <= threshold:
                    current_cluster.append(levels[i])
                else:
                    # Found a new cluster, calculate average for the previous cluster
                    avg_price = sum([level[1] for level in current_cluster]) / len(current_cluster)
                    avg_date = current_cluster[-1][0]  # Use most recent date
                    clustered.append((avg_date, avg_price))
                    current_cluster = [levels[i]]
            
            # Add the last cluster
            if current_cluster:
                avg_price = sum([level[1] for level in current_cluster]) / len(current_cluster)
                avg_date = current_cluster[-1][0]  # Use most recent date
                clustered.append((avg_date, avg_price))
            
            return clustered
        
        # Cluster the pivot highs and lows
        resistance_levels = cluster_levels(pivots_high, pivot_threshold)
        support_levels = cluster_levels(pivots_low, pivot_threshold)
        
        # Sort by price (descending for resistance, ascending for support)
        resistance_levels.sort(key=lambda x: -x[1])
        support_levels.sort(key=lambda x: x[1])
        
        # Get current price
        current_price = self.df['close'].iloc[-1]
        
        # Find closest support and resistance
        closest_resistance = None
        closest_support = None
        
        for level in resistance_levels:
            if level[1] > current_price:
                closest_resistance = level
                break
        
        for level in reversed(support_levels):
            if level[1] < current_price:
                closest_support = level
                break
        
        # Calculate distance to closest levels
        resistance_distance = None
        support_distance = None
        
        if closest_resistance:
            resistance_distance = (closest_resistance[1] - current_price) / current_price * 100
        
        if closest_support:
            support_distance = (current_price - closest_support[1]) / current_price * 100
        
        # Generate signals
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # Price very close to support
        if support_distance is not None and support_distance < 0.5:
            current_signal = 1
            signal_type = "Price at Support"
            signal_strength = 2
        
        # Price very close to resistance
        elif resistance_distance is not None and resistance_distance < 0.5:
            current_signal = -1
            signal_type = "Price at Resistance"
            signal_strength = 2
        
        # Adjust signal if we also have trend information
        if 'uptrend' in self.df.columns and 'downtrend' in self.df.columns:
            # Strong buy signal: Uptrend and at support
            if current_signal == 1 and self.df['uptrend'].iloc[-1] == 1:
                signal_type += " in Uptrend"
                signal_strength += 1
            
            # Strong sell signal: Downtrend and at resistance
            elif current_signal == -1 and self.df['downtrend'].iloc[-1] == 1:
                signal_type += " in Downtrend"
                signal_strength += 1
        
        # Format levels for output
        formatted_resistance = [round(level[1], 2) for level in resistance_levels[:3]] if resistance_levels else []
        formatted_support = [round(level[1], 2) for level in support_levels[:3]] if support_levels else []
        
        self.indicators['support_resistance'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'current_price': round(current_price, 2),
                'resistance_levels': formatted_resistance,
                'support_levels': formatted_support,
                'closest_resistance': round(closest_resistance[1], 2) if closest_resistance else None,
                'closest_support': round(closest_support[1], 2) if closest_support else None,
                'resistance_distance': round(resistance_distance, 2) if resistance_distance else None,
                'support_distance': round(support_distance, 2) if support_distance else None,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Support/Resistance',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_fibonacci_retracement(self):
        """Calculate Fibonacci retracement levels"""
        # Get parameters
        lookback = self.params.get_indicator_param('fibonacci_retracement')['lookback']
        
        # Identify significant swing high and low
        # Use a subset of recent data
        if len(self.df) < lookback:
            lookback = len(self.df)
            
        recent_data = self.df.iloc[-lookback:]
        
        # Find the highest high and lowest low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Determine the trend direction
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        trend = 'uptrend' if high_idx > low_idx else 'downtrend'
        
        # Calculate Fibonacci levels
        if trend == 'uptrend':
            # Uptrend: retracements from low to high
            range_size = swing_high - swing_low
            fib_0 = swing_low
            fib_236 = swing_low + 0.236 * range_size
            fib_382 = swing_low + 0.382 * range_size
            fib_50 = swing_low + 0.5 * range_size
            fib_618 = swing_low + 0.618 * range_size
            fib_786 = swing_low + 0.786 * range_size
            fib_100 = swing_high
        else:
            # Downtrend: retracements from high to low
            range_size = swing_high - swing_low
            fib_0 = swing_high
            fib_236 = swing_high - 0.236 * range_size
            fib_382 = swing_high - 0.382 * range_size
            fib_50 = swing_high - 0.5 * range_size
            fib_618 = swing_high - 0.618 * range_size
            fib_786 = swing_high - 0.786 * range_size
            fib_100 = swing_low
        
        # Get current price
        current_price = self.df['close'].iloc[-1]
        
        # Check which level the price is closest to
        fib_levels = [
            (0, fib_0),
            (0.236, fib_236),
            (0.382, fib_382),
            (0.5, fib_50),
            (0.618, fib_618),
            (0.786, fib_786),
            (1, fib_100)
        ]
        
        closest_level = min(fib_levels, key=lambda x: abs(x[1] - current_price))
        
        # Check if price is very near a Fibonacci level
        price_at_fib = abs(closest_level[1] - current_price) / current_price < 0.003
        
        # Generate signals
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        if price_at_fib:
            # At key Fibonacci level
            fib_value = closest_level[0]
            
            if trend == 'uptrend':
                # In uptrend, deeper retracements are stronger buy signals
                if 0.5 <= fib_value <= 0.786:
                    current_signal = 1
                    signal_strength = 3
                    signal_type = f"Price at {fib_value} Retracement in Uptrend"
                elif fib_value >= 0.382:
                    current_signal = 1
                    signal_strength = 2
                    signal_type = f"Price at {fib_value} Retracement in Uptrend"
                elif 0 < fib_value < 0.382:
                    current_signal = -1
                    signal_strength = 1
                    signal_type = f"Shallow {fib_value} Retracement in Uptrend"
            else:
                # In downtrend, deeper retracements are stronger sell signals
                if 0.5 <= fib_value <= 0.786:
                    current_signal = -1
                    signal_strength = 3
                    signal_type = f"Price at {fib_value} Retracement in Downtrend"
                elif fib_value >= 0.382:
                    current_signal = -1
                    signal_strength = 2
                    signal_type = f"Price at {fib_value} Retracement in Downtrend"
                elif 0 < fib_value < 0.382:
                    current_signal = 1
                    signal_strength = 1
                    signal_type = f"Shallow {fib_value} Retracement in Downtrend"
        
        self.indicators['fibonacci_retracement'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'trend': trend,
                'fib_0': round(fib_0, 2),
                'fib_236': round(fib_236, 2),
                'fib_382': round(fib_382, 2),
                'fib_50': round(fib_50, 2),
                'fib_618': round(fib_618, 2),
                'fib_786': round(fib_786, 2),
                'fib_100': round(fib_100, 2),
                'closest_level': closest_level[0],
                'closest_price': round(closest_level[1], 2),
                'price_at_fib': price_at_fib,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Fibonacci Retracement',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def calculate_divergence(self):
        """Calculate divergence between price and oscillators"""
        # Ensure we have the required indicators
        if 'rsi' not in self.df.columns:
            self.calculate_rsi()
        
        if 'macd_line' not in self.df.columns:
            self.calculate_macd()
        
        # Get parameters
        lookback = self.params.get_indicator_param('divergence')['lookback']
        tolerance = self.params.get_indicator_param('divergence')['tolerance']
        
        # Use a subset of recent data
        recent = min(lookback, len(self.df) - 1)
        
        # Find significant price swing points
        price_highs = []
        price_lows = []
        
        window_size = 5
        for i in range(window_size, len(self.df) - window_size):
            # Check for local price highs
            if self.df['high'].iloc[i] == max(self.df['high'].iloc[i-window_size:i+window_size+1]):
                price_highs.append((i, self.df['high'].iloc[i]))
            
            # Check for local price lows
            if self.df['low'].iloc[i] == min(self.df['low'].iloc[i-window_size:i+window_size+1]):
                price_lows.append((i, self.df['low'].iloc[i]))
        
        # Find oscillator swing points (using RSI)
        rsi_highs = []
        rsi_lows = []
        
        for i in range(window_size, len(self.df) - window_size):
            # Check for local RSI highs
            if self.df['rsi'].iloc[i] == max(self.df['rsi'].iloc[i-window_size:i+window_size+1]):
                rsi_highs.append((i, self.df['rsi'].iloc[i]))
            
            # Check for local RSI lows
            if self.df['rsi'].iloc[i] == min(self.df['rsi'].iloc[i-window_size:i+window_size+1]):
                rsi_lows.append((i, self.df['rsi'].iloc[i]))
        
        # Look for divergences
        # Bearish divergence: Price makes higher high but oscillator makes lower high
        # Bullish divergence: Price makes lower low but oscillator makes higher low
        
        bullish_divergence = False
        bearish_divergence = False
        divergence_strength = 0
        
        # Check recent swing points (last 2)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            # Get the last two price highs
            last_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            
            # Get the last two RSI highs
            last_rsi_high = rsi_highs[-1]
            prev_rsi_high = rsi_highs[-2]
            
            # Check timing (price and RSI highs should be close in time)
            if abs(last_price_high[0] - last_rsi_high[0]) <= 3 and abs(prev_price_high[0] - prev_rsi_high[0]) <= 3:
                # Check for bearish divergence
                if (last_price_high[1] > prev_price_high[1] and  # Price made higher high
                    last_rsi_high[1] < prev_rsi_high[1]):        # RSI made lower high
                    bearish_divergence = True
                    divergence_strength = 3
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Get the last two price lows
            last_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            
            # Get the last two RSI lows
            last_rsi_low = rsi_lows[-1]
            prev_rsi_low = rsi_lows[-2]
            
            # Check timing (price and RSI lows should be close in time)
            if abs(last_price_low[0] - last_rsi_low[0]) <= 3 and abs(prev_price_low[0] - prev_rsi_low[0]) <= 3:
                # Check for bullish divergence
                if (last_price_low[1] < prev_price_low[1] and  # Price made lower low
                    last_rsi_low[1] > prev_rsi_low[1]):        # RSI made higher low
                    bullish_divergence = True
                    divergence_strength = 3
        
        # Generate signal
        current_signal = 0
        signal_type = ""
        
        if bullish_divergence:
            current_signal = 1
            signal_type = "Bullish Divergence (RSI)"
        elif bearish_divergence:
            current_signal = -1
            signal_type = "Bearish Divergence (RSI)"
        
        # Also check MACD divergence
        macd_bullish_divergence = False
        macd_bearish_divergence = False
        
        # Similar logic for MACD divergence detection
        # (simplified for brevity)
        
        if macd_bullish_divergence:
            current_signal = 1
            signal_type = "Bullish Divergence (MACD)"
            divergence_strength = 3
        elif macd_bearish_divergence:
            current_signal = -1
            signal_type = "Bearish Divergence (MACD)"
            divergence_strength = 3
        
        self.indicators['divergence'] = {
            'signal': current_signal,
            'signal_strength': divergence_strength,
            'values': {
                'bullish_divergence_rsi': bullish_divergence,
                'bearish_divergence_rsi': bearish_divergence,
                'bullish_divergence_macd': macd_bullish_divergence,
                'bearish_divergence_macd': macd_bearish_divergence,
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Divergence',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': divergence_strength,
                'name': signal_type
            })
    
    def calculate_vix_analysis(self):
        """
        Analyze VIX-like volatility for market timing
        This is a simplified approach for stocks where VIX isn't directly available
        """
        # Get parameters
        smoothing_period = self.params.get_indicator_param('vix_analysis')['smoothing_period']
        threshold_high = self.params.get_indicator_param('vix_analysis')['threshold_high']
        threshold_low = self.params.get_indicator_param('vix_analysis')['threshold_low']
        
        # Calculate ATR as volatility proxy if not already done
        if 'atr' not in self.df.columns:
            self.calculate_atr()
        
        # Calculate ATR percentage (relative to price)
        self.df['atr_pct'] = 100 * self.df['atr'] / self.df['close']
        
        # Calculate smoothed ATR percentage
        self.df['smoothed_atr_pct'] = self.df['atr_pct'].rolling(window=smoothing_period).mean()
        
        # Calculate percentile ranks for volatility
        lookback = 100
        if len(self.df) >= lookback:
            # Get recent volatility values
            recent_vol = self.df['smoothed_atr_pct'].iloc[-lookback:]
            
            # Calculate percentile rank of current volatility
            current_vol = self.df['smoothed_atr_pct'].iloc[-1]
            higher_vol_count = (recent_vol > current_vol).sum()
            vol_percentile = 100 * higher_vol_count / len(recent_vol)
        else:
            vol_percentile = 50  # Default if not enough data
        
        # Generate signal
        current_signal = 0
        signal_strength = 0
        signal_type = ""
        
        # High volatility often indicates market bottoms (contrarian buy signal)
        if vol_percentile < 20:  # Bottom 20% of volatility
            if self.df['atr_pct'].iloc[-1] > threshold_high:
                current_signal = 1
                signal_type = "High Volatility (Contrarian Buy)"
                signal_strength = 2
        
        # Low volatility often precedes market corrections
        elif vol_percentile > 80:  # Top 20% of volatility
            if self.df['atr_pct'].iloc[-1] < threshold_low:
                current_signal = -1
                signal_type = "Low Volatility (Potential Correction)"
                signal_strength = 2
        
        # Volatility expansion/contraction
        vol_change = ((self.df['smoothed_atr_pct'].iloc[-1] / self.df['smoothed_atr_pct'].iloc[-10]) - 1) * 100
        
        if vol_change > 20:  # Volatility expanded by 20%+
            if current_signal == 0:
                current_signal = 1
                signal_type = "Volatility Expansion"
                signal_strength = 1
        elif vol_change < -20:  # Volatility contracted by 20%+
            if current_signal == 0:
                current_signal = -1
                signal_type = "Volatility Contraction"
                signal_strength = 1
        
        self.indicators['vix_analysis'] = {
            'signal': current_signal,
            'signal_strength': signal_strength,
            'values': {
                'current_vol': round(self.df['atr_pct'].iloc[-1], 2),
                'smoothed_vol': round(self.df['smoothed_atr_pct'].iloc[-1], 2),
                'vol_percentile': round(vol_percentile, 2),
                'vol_change_pct': round(vol_change, 2),
                'signal_type': signal_type
            }
        }
        
        # Add to signals list if signal exists
        if current_signal != 0:
            self.signals.append({
                'indicator': 'Volatility Analysis',
                'signal': 'BUY' if current_signal == 1 else 'SELL',
                'strength': signal_strength,
                'name': signal_type
            })
    
    def get_signals(self):
        """
        Get all trading signals based on indicators
        
        Returns:
            Dictionary with all signals
        """
        # Calculate all indicators if not already done
        if not self.indicators:
            self.calculate_all()
        
        # Return signals
        return {
            'buy_signals': [s for s in self.signals if s['signal'] == 'BUY'],
            'sell_signals': [s for s in self.signals if s['signal'] == 'SELL'],
            'indicators': self.indicators
        }
    
    def get_overall_signal(self):
        """
        Determine overall trading signal based on all indicator signals.
        
        Returns:
            Dictionary with overall signal information
        """
        # Calculate all indicators if not already done
        if not self.indicators:
            self.calculate_all()
        
        # Get indicator weights from parameters
        indicator_weights = self.params.get_signal_param('indicator_strength_weights')
        
        # Count buy and sell signals with their weights
        buy_signals = [s for s in self.signals if s['signal'] == 'BUY']
        sell_signals = [s for s in self.signals if s['signal'] == 'SELL']
        
        # Calculate weighted signal strength
        buy_strength = 0
        for signal in buy_signals:
            indicator = signal['indicator']
            # Get weight for this indicator (default to 1 if not specified)
            weight = indicator_weights.get(indicator.lower().replace(' ', '_'), 1)
            buy_strength += signal['strength'] * weight
        
        sell_strength = 0
        for signal in sell_signals:
            indicator = signal['indicator']
            # Get weight for this indicator (default to 1 if not specified)
            weight = indicator_weights.get(indicator.lower().replace(' ', '_'), 1)
            sell_strength += signal['strength'] * weight
        
        # Determine overall signal based on weighted strengths
        if buy_strength > sell_strength * 1.5:  # Strongly bullish
            signal_type = 'STRONG BUY'
            strength = 5
            confidence = min(100, int(buy_strength * 100 / (buy_strength + sell_strength)))
        elif buy_strength > sell_strength:  # Moderately bullish
            signal_type = 'BUY'
            strength = 4
            confidence = min(100, int(buy_strength * 100 / (buy_strength + sell_strength)))
        elif sell_strength > buy_strength * 1.5:  # Strongly bearish
            signal_type = 'STRONG SELL'
            strength = 5
            confidence = min(100, int(sell_strength * 100 / (buy_strength + sell_strength)))
        elif sell_strength > buy_strength:  # Moderately bearish
            signal_type = 'SELL'
            strength = 4
            confidence = min(100, int(sell_strength * 100 / (buy_strength + sell_strength)))
        else:  # Neutral
            signal_type = 'NEUTRAL'
            strength = 0
            confidence = 50
        
        # Generate summary of key indicators
        key_indicators = []
        
        # Add trend indicators
        if 'moving_averages' in self.indicators:
            ma = self.indicators['moving_averages']
            if ma['values'].get('uptrend', False):
                key_indicators.append("Bullish Trend on Moving Averages")
            elif ma['values'].get('downtrend', False):
                key_indicators.append("Bearish Trend on Moving Averages")
        
        # Add momentum indicators
        if 'rsi' in self.indicators:
            rsi = self.indicators['rsi']
            rsi_value = rsi['values'].get('rsi')
            if rsi_value:
                if rsi_value < 30:
                    key_indicators.append(f"RSI Oversold ({rsi_value})")
                elif rsi_value > 70:
                    key_indicators.append(f"RSI Overbought ({rsi_value})")
        
        # Add other key indicators
        if 'supertrend' in self.indicators:
            st = self.indicators['supertrend']
            key_indicators.append(f"Supertrend: {st['values'].get('direction', 'Unknown')}")
        
        # Format summary
        if key_indicators:
            summary = "Key indicators: " + " • ".join(key_indicators)
        else:
            summary = "Mixed signals, no clear direction"
        
        return {
            'signal': signal_type,
            'strength': strength,
            'confidence': confidence,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'summary': summary,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_detailed_analysis(self):
        """
        Get a detailed analysis with trade setup information
        
        Returns:
            Dictionary with detailed analysis information
        """
        # Ensure all indicators are calculated
        if not self.indicators:
            self.calculate_all()
        
        # Get overall signal
        overall = self.get_overall_signal()
        
        # Get current price
        current_price = self.df['close'].iloc[-1]
        
        # Get ATR for stop loss calculation
        if 'atr' not in self.indicators:
            self.calculate_atr()
        
        atr_value = self.indicators['atr']['values']['atr']
        
        # Get stop loss and target parameters
        stop_multiplier = self.params.get_signal_param('stop_multiplier')
        target_multiplier = self.params.get_signal_param('target_multiplier')
        
        # Calculate stop loss and target based on signal
        if overall['signal'] == 'BUY' or overall['signal'] == 'STRONG BUY':
            stop_loss = current_price - (atr_value * stop_multiplier)
            target = current_price + (atr_value * target_multiplier)
            
            # Check support/resistance levels for better stop placement
            if 'support_resistance' in self.indicators:
                sr = self.indicators['support_resistance']['values']
                if sr.get('closest_support') and sr['closest_support'] < current_price:
                    # Use support as stop loss if it's within reasonable range
                    potential_stop = sr['closest_support'] - (0.2 * atr_value)
                    if current_price - potential_stop < atr_value * 2:  # Not too far
                        stop_loss = potential_stop
                
                if sr.get('closest_resistance') and sr['closest_resistance'] > current_price:
                    # Use resistance as target if it's within reasonable range
                    potential_target = sr['closest_resistance']
                    if potential_target - current_price > atr_value:  # Not too close
                        target = potential_target
            
        elif overall['signal'] == 'SELL' or overall['signal'] == 'STRONG SELL':
            stop_loss = current_price + (atr_value * stop_multiplier)
            target = current_price - (atr_value * target_multiplier)
            
            # Check support/resistance levels for better stop placement
            if 'support_resistance' in self.indicators:
                sr = self.indicators['support_resistance']['values']
                if sr.get('closest_resistance') and sr['closest_resistance'] > current_price:
                    # Use resistance as stop loss if it's within reasonable range
                    potential_stop = sr['closest_resistance'] + (0.2 * atr_value)
                    if potential_stop - current_price < atr_value * 2:  # Not too far
                        stop_loss = potential_stop
                
                if sr.get('closest_support') and sr['closest_support'] < current_price:
                    # Use support as target if it's within reasonable range
                    potential_target = sr['closest_support']
                    if current_price - potential_target > atr_value:  # Not too close
                        target = potential_target
        else:
            # Neutral signal
            stop_loss = None
            target = None
        
        # Calculate reward-to-risk ratio
        if stop_loss and target:
            risk = abs(current_price - stop_loss)
            reward = abs(target - current_price)
            reward_risk_ratio = round(reward / risk, 2) if risk > 0 else 0
        else:
            reward_risk_ratio = 0
        
        # Calculate stop loss and target percentages
        if stop_loss:
            stop_loss_pct = round(abs(stop_loss - current_price) / current_price * 100, 2)
        else:
            stop_loss_pct = None
            
        if target:
            target_pct = round(abs(target - current_price) / current_price * 100, 2)
        else:
            target_pct = None
        
        # Get support and resistance levels
        if 'support_resistance' in self.indicators:
            support_levels = self.indicators['support_resistance']['values'].get('support_levels', [])
            resistance_levels = self.indicators['support_resistance']['values'].get('resistance_levels', [])
        else:
            support_levels = []
            resistance_levels = []
        
        # Create trade checklist
        checklist = self._create_trade_checklist(overall['signal'])
        
        # Return detailed analysis
        return {
            'signal': overall['signal'],
            'strength': overall['strength'],
            'confidence': overall['confidence'],
            'current_price': round(current_price, 2),
            'entry_price': round(current_price, 2),
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'stop_loss_pct': stop_loss_pct,
            'target': round(target, 2) if target else None,
            'target_pct': target_pct,
            'reward_risk_ratio': reward_risk_ratio,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'checklist': checklist,
            'buy_signals': overall['buy_signals'],
            'sell_signals': overall['sell_signals'],
            'summary': overall['summary'],
            'timestamp': overall['timestamp']
        }
    
    def _create_trade_checklist(self, signal_type):
        """
        Create a trade checklist for validation
        
        Args:
            signal_type: The type of signal (BUY, SELL, etc.)
            
        Returns:
            Dictionary with checklist items
        """
        checklist = {}
        
        # Skip checklist for neutral signals
        if signal_type == 'NEUTRAL':
            return checklist
        
        is_buy = signal_type in ['BUY', 'STRONG BUY']
        
        # 1. Trend alignment
        trend_aligned = False
        if 'moving_averages' in self.indicators:
            ma = self.indicators['moving_averages']['values']
            if is_buy:
                trend_aligned = ma.get('uptrend', False) or ma.get('price_above_sma_long', False)
            else:
                trend_aligned = ma.get('downtrend', False) or ma.get('price_above_sma_long', False) == False
        checklist['trend_aligned'] = trend_aligned
        
        # 2. Volume confirmation
        volume_confirmed = False
        if 'volume' in self.df.columns and self.df['volume'].sum() > 0:
            # Check recent volume
            avg_volume = self.df['volume'].iloc[-20:].mean()
            current_volume = self.df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume * 1.2
        checklist['volume_confirmed'] = volume_confirmed
        
        # 3. Risk-Reward ratio check
        rrr_confirmed = False
        if 'reward_to_risk' in self.indicators:
            rrr = self.indicators['reward_to_risk']['values'].get('rrr', 0)
            min_rrr = self.params.get_signal_param('min_rrr')
            rrr_confirmed = rrr >= min_rrr
        checklist['rrr_confirmed'] = rrr_confirmed
        
        # 4. Multiple timeframe alignment (would need data from multiple timeframes)
        # For now, use a simplified approach based on trend strength
        timeframe_aligned = False
        if 'adx' in self.indicators:
            adx_trend_strength = self.indicators['adx']['values'].get('trend_strength', 'Weak')
            timeframe_aligned = adx_trend_strength == 'Strong'
        checklist['timeframe_aligned'] = timeframe_aligned
        
        # 5. Pattern confirmation
        pattern_confirmed = False
        for signal in self.signals:
            if signal['indicator'].startswith('Chart Pattern') or signal['indicator'].startswith('Candlestick'):
                if (is_buy and signal['signal'] == 'BUY') or (not is_buy and signal['signal'] == 'SELL'):
                    pattern_confirmed = True
                    break
        checklist['pattern_confirmed'] = pattern_confirmed
        
        # 6. Support/Resistance validation
        sr_confirmed = False
        if 'support_resistance' in self.indicators:
            sr = self.indicators['support_resistance']
            if is_buy and sr['signal'] == 1:
                sr_confirmed = True
            elif not is_buy and sr['signal'] == -1:
                sr_confirmed = True
        checklist['sr_confirmed'] = sr_confirmed
        
        return checklist
    
    def backtest_strategy(self, lookback_days=250, backtesting_func=None):
        """
        Perform comprehensive backtesting of the current strategy
        
        Args:
            lookback_days: Number of days to include in backtest
            backtesting_func: Optional custom backtesting function
            
        Returns:
            BacktestEngine instance with results
        """
        # Create a copy of the dataframe for backtesting
        df_backtest = self.df.copy()
        
        # Initialize backtesting engine
        backtest = BacktestEngine(df_backtest, self.params)
        
        # Default strategy function if none provided
        if backtesting_func is None:
            def default_strategy(data):
                # Initialize indicators
                indicators = TechnicalIndicators(data, self.params)
                indicators.calculate_all()
                
                # Get overall signal
                signal = indicators.get_overall_signal()
                
                # Convert signal to numeric (-1, 0, 1)
                if signal['signal'] in ['BUY', 'STRONG BUY']:
                    return 1
                elif signal['signal'] in ['SELL', 'STRONG SELL']:
                    return -1
                else:
                    return 0
                    
            backtesting_func = default_strategy
        
        # Run backtest
        results = backtest.run_backtest(backtesting_func)
        
        # Generate detailed report
        report = backtest.generate_report()
        
        return {
            'backtest_engine': backtest,
            'results': results,
            'report': report
        }


# ===============================================================
# Enhanced Backtesting System
# ===============================================================
class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, df, params=None, initial_capital=100000.0):
        """
        Initialize backtesting engine
        
        Args:
            df: DataFrame with OHLCV data
            params: Optional TradingParameters instance
            initial_capital: Starting capital for backtesting
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Initialize parameters (use defaults if not provided)
        self.params = params or TradingParameters()
        
        # Set up initial backtesting state
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = []
        self.equity_curve = []
        
        # Performance metrics
        self.metrics = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, strategy_func, start_date=None, end_date=None):
        """
        Run backtest using the provided strategy function
        
        Args:
            strategy_func: Function that returns signals (1=buy, -1=sell, 0=hold)
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range if provided
        if start_date:
            self.df = self.df[self.df.index >= pd.to_datetime(start_date)]
        if end_date:
            self.df = self.df[self.df.index <= pd.to_datetime(end_date)]
        
        # Reset backtesting state
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = []
        self.equity_curve = [{'date': self.df.index[0], 'equity': self.initial_capital}]
        
        # Add columns for signals and positions
        self.df['signal'] = 0
        self.df['position'] = 0
        self.df['equity'] = self.initial_capital
        
        # Apply strategy function to get signals
        self.logger.info("Generating signals for backtest")
        for i in range(len(self.df)):
            # Create a subset of data up to this point (to prevent lookahead bias)
            data_subset = self.df.iloc[:i+1].copy()
            
            # Get signal (1=buy, -1=sell, 0=hold)
            try:
                signal = strategy_func(data_subset)
                self.df.loc[self.df.index[i], 'signal'] = signal
            except Exception as e:
                self.logger.warning(f"Error generating signal: {e}")
                self.df.loc[self.df.index[i], 'signal'] = 0
        
        # Process signals and track positions
        self.logger.info("Processing trades for backtest")
        position = 0
        entry_price = 0
        entry_date = None
        stop_loss = 0
        target = 0
        
        for i in range(1, len(self.df)):
            prev_idx = i - 1
            current_idx = i
            
            # Get signal and price data
            signal = self.df['signal'].iloc[current_idx]
            open_price = self.df['open'].iloc[current_idx]
            high_price = self.df['high'].iloc[current_idx]
            low_price = self.df['low'].iloc[current_idx]
            close_price = self.df['close'].iloc[current_idx]
            
            # Determine ATR for position sizing if available
            atr = self.df['atr'].iloc[prev_idx] if 'atr' in self.df.columns else close_price * 0.02
            
            # Update existing position
            if position != 0:
                # Check if stop loss or target hit during the day
                if position == 1:  # Long position
                    # Check if stop loss hit
                    if low_price <= stop_loss:
                        # Close position at stop loss
                        exit_price = stop_loss
                        pnl = (exit_price - entry_price) / entry_price
                        position_size = self.current_capital * 0.1  # 10% position size
                        pnl_amount = position_size * pnl
                        self.current_capital += pnl_amount
                        
                        # Record trade
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': self.df.index[current_idx],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl_pct': pnl * 100,
                            'pnl_amount': pnl_amount,
                            'exit_reason': 'stop_loss'
                        }
                        self.trades.append(trade)
                        
                        # Reset position
                        position = 0
                    
                    # Check if target hit
                    elif high_price >= target:
                        # Close position at target
                        exit_price = target
                        pnl = (exit_price - entry_price) / entry_price
                        position_size = self.current_capital * 0.1  # 10% position size
                        pnl_amount = position_size * pnl
                        self.current_capital += pnl_amount
                        
                        # Record trade
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': self.df.index[current_idx],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl_pct': pnl * 100,
                            'pnl_amount': pnl_amount,
                            'exit_reason': 'target_hit'
                        }
                        self.trades.append(trade)
                        
                        # Reset position
                        position = 0
                
                elif position == -1:  # Short position
                    # Check if stop loss hit
                    if high_price >= stop_loss:
                        # Close position at stop loss
                        exit_price = stop_loss
                        pnl = (entry_price - exit_price) / entry_price
                        position_size = self.current_capital * 0.1  # 10% position size
                        pnl_amount = position_size * pnl
                        self.current_capital += pnl_amount
                        
                        # Record trade
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': self.df.index[current_idx],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl_pct': pnl * 100,
                            'pnl_amount': pnl_amount,
                            'exit_reason': 'stop_loss'
                        }
                        self.trades.append(trade)
                        
                        # Reset position
                        position = 0
                    
                    # Check if target hit
                    elif low_price <= target:
                        # Close position at target
                        exit_price = target
                        pnl = (entry_price - exit_price) / entry_price
                        position_size = self.current_capital * 0.1  # 10% position size
                        pnl_amount = position_size * pnl
                        self.current_capital += pnl_amount
                        
                        # Record trade
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': self.df.index[current_idx],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl_pct': pnl * 100,
                            'pnl_amount': pnl_amount,
                            'exit_reason': 'target_hit'
                        }
                        self.trades.append(trade)
                        
                        # Reset position
                        position = 0
            
            # Check for signal to enter new position
            if position == 0 and signal != 0:
                # Enter new position
                position = signal
                entry_price = open_price  # Use next day's open for entry
                entry_date = self.df.index[current_idx]
                
                # Set stop loss and target based on ATR
                stop_multiplier = self.params.get_signal_param('stop_multiplier')
                target_multiplier = self.params.get_signal_param('target_multiplier')
                
                if position == 1:  # Long position
                    stop_loss = entry_price - (atr * stop_multiplier)
                    target = entry_price + (atr * target_multiplier)
                else:  # Short position
                    stop_loss = entry_price + (atr * stop_multiplier)
                    target = entry_price - (atr * target_multiplier)
            
            # Exit on contrary signal
            elif position != 0 and signal == -position:
                # Close position at open
                exit_price = open_price
                
                if position == 1:  # Long position
                    pnl = (exit_price - entry_price) / entry_price
                else:  # Short position
                    pnl = (entry_price - exit_price) / entry_price
                
                position_size = self.current_capital * 0.1  # 10% position size
                pnl_amount = position_size * pnl
                self.current_capital += pnl_amount
                
                # Record trade
                trade = {
                    'entry_date': entry_date,
                    'exit_date': self.df.index[current_idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl_pct': pnl * 100,
                    'pnl_amount': pnl_amount,
                    'exit_reason': 'signal_reversal'
                }
                self.trades.append(trade)
                
                # Reset position
                position = 0
            
            # Track position
            self.df.loc[self.df.index[current_idx], 'position'] = position
            
            # Update equity curve (mark-to-market)
            if position == 1:
                position_value = position_size * (close_price / entry_price)
                equity = self.current_capital - position_size + position_value
            elif position == -1:
                position_value = position_size * (2 - close_price / entry_price)
                equity = self.current_capital - position_size + position_value
            else:
                equity = self.current_capital
            
            self.df.loc[self.df.index[current_idx], 'equity'] = equity
            self.equity_curve.append({'date': self.df.index[current_idx], 'equity': equity})
        
        # Close any open positions at the end of the test
        if position != 0:
            # Close position at last close
            exit_price = self.df['close'].iloc[-1]
            
            if position == 1:  # Long position
                pnl = (exit_price - entry_price) / entry_price
            else:  # Short position
                pnl = (entry_price - exit_price) / entry_price
            
            position_size = self.current_capital * 0.1  # 10% position size
            pnl_amount = position_size * pnl
            self.current_capital += pnl_amount
            
            # Record trade
            trade = {
                'entry_date': entry_date,
                'exit_date': self.df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl_pct': pnl * 100,
                'pnl_amount': pnl_amount,
                'exit_reason': 'end_of_test'
            }
            self.trades.append(trade)
        
        # Calculate performance metrics
        self.calculate_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': self.metrics
        }
    
    def calculate_metrics(self):
        """Calculate performance metrics for the backtest"""
        # Skip if no trades
        if not self.trades:
            self.metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0
            }
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['pnl_pct'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Profit and loss
        gross_profit = sum([t['pnl_amount'] for t in self.trades if t['pnl_amount'] > 0])
        gross_loss = abs(sum([t['pnl_amount'] for t in self.trades if t['pnl_amount'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Convert equity curve to numpy array for calculations
        equity = np.array([point['equity'] for point in self.equity_curve])
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_drawdown = drawdown.max()
        
        # Annualized return
        start_date = self.df.index[0]
        end_date = self.df.index[-1]
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Daily returns
        daily_returns = np.diff(equity) / equity[:-1]
        
        # Sharpe ratio (annualized)
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (annualized)
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = (np.mean(excess_returns) / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Largest profit and loss
        largest_profit = max([t['pnl_amount'] for t in self.trades]) if self.trades else 0
        largest_loss = min([t['pnl_amount'] for t in self.trades]) if self.trades else 0
        
        # Average profit and loss
        avg_profit = np.mean([t['pnl_amount'] for t in self.trades if t['pnl_amount'] > 0]) if profitable_trades > 0 else 0
        avg_loss = np.mean([t['pnl_amount'] for t in self.trades if t['pnl_amount'] < 0]) if total_trades - profitable_trades > 0 else 0
        
        # Save metrics
        self.metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }
    
    def monte_carlo_simulation(self, num_simulations=1000):
        """
        Run Monte Carlo simulation by randomizing trade order
        
        Args:
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with simulation results
        """
        # Skip if no trades
        if not self.trades:
            return {
                'simulations': [],
                'confidence_intervals': {
                    'return_5pct': 0,
                    'return_50pct': 0,
                    'return_95pct': 0,
                    'drawdown_5pct': 0,
                    'drawdown_50pct': 0,
                    'drawdown_95pct': 0
                }
            }
        
        # Extract trade returns
        trade_returns = np.array([t['pnl_pct'] / 100 for t in self.trades])
        
        # Prepare arrays for results
        final_returns = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        
        # Run simulations
        for i in range(num_simulations):
            # Randomize trade order
            np.random.shuffle(trade_returns)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + trade_returns)
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            
            # Store results
            final_returns[i] = cumulative_returns[-1] - 1
            max_drawdowns[i] = drawdowns.max()
        
        # Calculate confidence intervals
        return_5pct = np.percentile(final_returns, 5)
        return_50pct = np.percentile(final_returns, 50)
        return_95pct = np.percentile(final_returns, 95)
        
        drawdown_5pct = np.percentile(max_drawdowns, 5)
        drawdown_50pct = np.percentile(max_drawdowns, 50)
        drawdown_95pct = np.percentile(max_drawdowns, 95)
        
        return {
            'simulations': {
                'returns': final_returns.tolist(),
                'drawdowns': max_drawdowns.tolist()
            },
            'confidence_intervals': {
                'return_5pct': return_5pct,
                'return_50pct': return_50pct,
                'return_95pct': return_95pct,
                'drawdown_5pct': drawdown_5pct,
                'drawdown_50pct': drawdown_50pct,
                'drawdown_95pct': drawdown_95pct
            }
        }
    
    def walk_forward_optimization(self, strategy_func, param_grid, in_sample_pct=0.6, step_size=0.2):
        """
        Perform walk-forward optimization to reduce overfitting
        
        Args:
            strategy_func: Strategy function that takes params and returns signals
            param_grid: Dictionary with parameter names and values to test
            in_sample_pct: Percentage of data to use for in-sample optimization
            step_size: Size of each walk-forward step as a percentage of data
            
        Returns:
            Dictionary with optimized parameters and out-of-sample results
        """
        # Ensure enough data
        if len(self.df) < 100:
            return {
                'error': 'Not enough data for walk-forward optimization',
                'best_params': {}
            }
        
        # Prepare list for results
        walk_forward_results = []
        
        # Calculate window sizes
        total_size = len(self.df)
        in_sample_size = int(total_size * in_sample_pct)
        step_size_bars = int(total_size * step_size)
        
        # Grid search helper function
        def grid_search(params, df_subset):
            # Recursive function to generate all parameter combinations
            def generate_combinations(keys, current_combo={}):
                if not keys:
                    yield current_combo.copy()
                    return
                
                current_key = keys[0]
                remaining_keys = keys[1:]
                
                for value in params[current_key]:
                    current_combo[current_key] = value
                    yield from generate_combinations(remaining_keys, current_combo)
            
            # Track best parameters
            best_params = None
            best_metric = -float('inf')  # For sharpe ratio or similar
            
            # Generate all parameter combinations
            param_keys = list(params.keys())
            for combo in generate_combinations(param_keys):
                # Create a function with fixed parameters
                def parameterized_strategy(data):
                    return strategy_func(data, **combo)
                
                # Run backtest with these parameters
                backtest = BacktestEngine(df_subset, self.params)
                results = backtest.run_backtest(parameterized_strategy)
                
                # Evaluate metric (e.g., sharpe ratio)
                metric = results['metrics']['sharpe_ratio']
                
                # Update best parameters if better
                if metric > best_metric:
                    best_metric = metric
                    best_params = combo
            
            return best_params, best_metric
        
        # Perform walk-forward analysis
        for i in range(0, total_size - in_sample_size, step_size_bars):
            # Define in-sample and out-of-sample periods
            in_sample_start = i
            in_sample_end = i + in_sample_size
            out_sample_start = in_sample_end
            out_sample_end = min(out_sample_start + step_size_bars, total_size)
            
            # Skip if out-of-sample period is too small
            if out_sample_end - out_sample_start < 20:
                continue
            
            # Get data subsets
            in_sample_df = self.df.iloc[in_sample_start:in_sample_end].copy()
            out_sample_df = self.df.iloc[out_sample_start:out_sample_end].copy()
            
            # Optimize parameters on in-sample data
            best_params, in_sample_metric = grid_search(param_grid, in_sample_df)
            
            # Test optimized parameters on out-of-sample data
            def parameterized_strategy(data):
                return strategy_func(data, **best_params)
            
            # Run backtest on out-of-sample data
            backtest = BacktestEngine(out_sample_df, self.params)
            out_sample_results = backtest.run_backtest(parameterized_strategy)
            
            # Store results
            walk_forward_results.append({
                'period': (self.df.index[in_sample_start], self.df.index[out_sample_end-1]),
                'best_params': best_params,
                'in_sample_metric': in_sample_metric,
                'out_sample_metric': out_sample_results['metrics']['sharpe_ratio'],
                'out_sample_return': out_sample_results['metrics']['total_return'],
                'out_sample_drawdown': out_sample_results['metrics']['max_drawdown']
            })
        
        # Combine results to get final parameter recommendations
        if not walk_forward_results:
            return {
                'error': 'No valid walk-forward periods found',
                'best_params': {}
            }
        
        # Calculate average performance metrics
        avg_out_sample_return = np.mean([r['out_sample_return'] for r in walk_forward_results])
        avg_out_sample_drawdown = np.mean([r['out_sample_drawdown'] for r in walk_forward_results])
        
        # Find most frequent parameter values
        param_keys = list(param_grid.keys())
        recommended_params = {}
        
        for key in param_keys:
            # Count parameter values
            param_counts = {}
            for result in walk_forward_results:
                param_value = result['best_params'][key]
                param_counts[param_value] = param_counts.get(param_value, 0) + 1
            
            # Get most frequent value
            recommended_params[key] = max(param_counts, key=param_counts.get)
        
        return {
            'periods': [(r['period'][0].strftime('%Y-%m-%d'), r['period'][1].strftime('%Y-%m-%d')) for r in walk_forward_results],
            'recommended_params': recommended_params,
            'avg_out_sample_return': avg_out_sample_return,
            'avg_out_sample_drawdown': avg_out_sample_drawdown,
            'all_results': walk_forward_results
        }
    
    def generate_report(self, include_trades=True, include_equity_curve=True):
        """
        Generate comprehensive backtest report
        
        Args:
            include_trades: Whether to include individual trades
            include_equity_curve: Whether to include equity curve data
            
        Returns:
            Dictionary with detailed backtest report
        """
        # Ensure metrics are calculated
        if not self.metrics:
            self.calculate_metrics()
        
        # Format metrics
        formatted_metrics = {
            'Total Trades': self.metrics.get('total_trades', 0),
            'Win Rate': f"{self.metrics.get('win_rate', 0) * 100:.2f}%",
            'Profit Factor': f"{self.metrics.get('profit_factor', 0):.2f}",
            'Total Return': f"{self.metrics.get('total_return', 0) * 100:.2f}%",
            'Annualized Return': f"{self.metrics.get('annualized_return', 0) * 100:.2f}%",
            'Max Drawdown': f"{self.metrics.get('max_drawdown', 0) * 100:.2f}%",
            'Sharpe Ratio': f"{self.metrics.get('sharpe_ratio', 0):.2f}",
            'Sortino Ratio': f"{self.metrics.get('sortino_ratio', 0):.2f}",
            'Calmar Ratio': f"{self.metrics.get('calmar_ratio', 0):.2f}",
            'Average Profit': f"${self.metrics.get('avg_profit', 0):.2f}",
            'Average Loss': f"${abs(self.metrics.get('avg_loss', 0)):.2f}",
            'Largest Profit': f"${self.metrics.get('largest_profit', 0):.2f}",
            'Largest Loss': f"${abs(self.metrics.get('largest_loss', 0)):.2f}",
        }
        
        # Prepare report
        report = {
            'metrics': formatted_metrics,
            'summary': f"Backtest Summary: {self.metrics.get('total_trades', 0)} trades with {self.metrics.get('win_rate', 0) * 100:.1f}% win rate, returning {self.metrics.get('total_return', 0) * 100:.1f}% ({self.metrics.get('annualized_return', 0) * 100:.1f}% annualized) with {self.metrics.get('max_drawdown', 0) * 100:.1f}% max drawdown."
        }
        
        # Include trades if requested
        if include_trades and self.trades:
            # Format trade data
            formatted_trades = []
            for trade in self.trades:
                formatted_trades.append({
                    'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': trade['exit_date'].strftime('%Y-%m-%d'),
                    'direction': 'Long' if trade['position'] == 1 else 'Short',
                    'entry_price': f"${trade['entry_price']:.2f}",
                    'exit_price': f"${trade['exit_price']:.2f}",
                    'pnl_pct': f"{trade['pnl_pct']:.2f}%",
                    'pnl_amount': f"${trade['pnl_amount']:.2f}",
                    'exit_reason': trade['exit_reason'].replace('_', ' ').title()
                })
            
            report['trades'] = formatted_trades
        
        # Include equity curve if requested
        if include_equity_curve and self.equity_curve:
            # Format equity curve data
            formatted_curve = []
            for point in self.equity_curve:
                formatted_curve.append({
                    'date': point['date'].strftime('%Y-%m-%d'),
                    'equity': f"${point['equity']:.2f}"
                })
            
            report['equity_curve'] = formatted_curve
        
        return report

# ===============================================================
# Chart Pattern Detection
# ===============================================================
class ChartPatterns:
    """Detect complex chart patterns in price data"""
    
    def __init__(self, df, params=None):
        """
        Initialize with DataFrame containing OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            params: Optional TradingParameters instance for pattern thresholds
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Initialize parameters (use defaults if not provided)
        self.params = params or TradingParameters()
        
        # Initialize results
        self.patterns = {}
        self.signals = []
        self.logger = logging.getLogger(__name__)
    
    def detect_all_patterns(self):
        """
        Detect all implemented chart patterns
        
        Returns:
            Dictionary with all detected patterns
        """
        # Initialize patterns dictionary
        patterns = {}
        
        # Detect various patterns
        try:
            patterns.update(self.detect_head_and_shoulders())
        except Exception as e:
            self.logger.warning(f"Error detecting head and shoulders patterns: {e}")
        
        try:
            patterns.update(self.detect_double_patterns())
        except Exception as e:
            self.logger.warning(f"Error detecting double top/bottom patterns: {e}")
        
        try:
            patterns.update(self.detect_triple_patterns())
        except Exception as e:
            self.logger.warning(f"Error detecting triple top/bottom patterns: {e}")
        
        try:
            patterns.update(self.detect_wedges())
        except Exception as e:
            self.logger.warning(f"Error detecting wedge patterns: {e}")
        
        try:
            patterns.update(self.detect_rectangle())
        except Exception as e:
            self.logger.warning(f"Error detecting rectangle patterns: {e}")
        
        try:
            patterns.update(self.detect_flags())
        except Exception as e:
            self.logger.warning(f"Error detecting flag patterns: {e}")
        
        try:
            patterns.update(self.detect_cup_and_handle())
        except Exception as e:
            self.logger.warning(f"Error detecting cup and handle pattern: {e}")
        
        try:
            patterns.update(self.detect_rounding_patterns())
        except Exception as e:
            self.logger.warning(f"Error detecting rounding patterns: {e}")
        
        # Store patterns
        self.patterns = patterns
        
        return patterns
    
    def find_swing_points(self, window_size=None):
        """
        Find swing highs and lows in the price data
        
        Args:
            window_size: Size of window to look for local extrema
            
        Returns:
            Dictionary with swing highs and lows
        """
        # Get window size from parameters if not provided
        if window_size is None:
            window_size = self.params.get_chart_pattern_param('swing_high_low_window')
        
        # Ensure we have enough data
        if len(self.df) < window_size * 2 + 1:
            return {
                'swing_highs': [],
                'swing_lows': []
            }
        
        # Find swing highs
        swing_highs = []
        for i in range(window_size, len(self.df) - window_size):
            # Check if current high is local maximum
            if self.df['high'].iloc[i] == self.df['high'].iloc[i-window_size:i+window_size+1].max():
                swing_highs.append((i, self.df.index[i], self.df['high'].iloc[i]))
        
        # Find swing lows
        swing_lows = []
        for i in range(window_size, len(self.df) - window_size):
            # Check if current low is local minimum
            if self.df['low'].iloc[i] == self.df['low'].iloc[i-window_size:i+window_size+1].min():
                swing_lows.append((i, self.df.index[i], self.df['low'].iloc[i]))
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def detect_head_and_shoulders(self):
        """
        Detect Head and Shoulders and Inverse Head and Shoulders patterns
        
        Returns:
            Dictionary with detected head and shoulders patterns
        """
        # Get parameters
        head_tolerance = self.params.get_chart_pattern_param('head_shoulders_head_tolerance')
        shoulder_tolerance = self.params.get_chart_pattern_param('head_shoulders_shoulder_tolerance')
        
        # Find swing highs and lows
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        # Need at least 5 swing points (3 highs and 2 lows)
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return {
                'head_and_shoulders': False,
                'inverse_head_and_shoulders': False
            }
        
        # Head and Shoulders pattern (top reversal)
        h_and_s = False
        
        # Check sequences of 3 swing highs with 2 swing lows in between
        for i in range(len(swing_highs) - 2):
            # Get 3 consecutive swing highs
            left_shoulder_idx, left_shoulder_date, left_shoulder = swing_highs[i]
            head_idx, head_date, head = swing_highs[i+1]
            right_shoulder_idx, right_shoulder_date, right_shoulder = swing_highs[i+2]
            
            # Check if the head is higher than both shoulders
            if (head > left_shoulder and head > right_shoulder):
                # Check if shoulders are approximately at the same level
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff <= shoulder_tolerance:
                    # Find the swing lows between the swing highs
                    neck_line_points = []
                    for low_idx, low_date, low in swing_lows:
                        if left_shoulder_idx < low_idx < head_idx or head_idx < low_idx < right_shoulder_idx:
                            neck_line_points.append((low_idx, low_date, low))
                    
                    # Need at least 2 lows to form a neckline
                    if len(neck_line_points) >= 2:
                        # Check if we've broken the neckline
                        neckline_level = sum(low for _, _, low in neck_line_points) / len(neck_line_points)
                        current_price = self.df['close'].iloc[-1]
                        
                        if current_price < neckline_level:
                            h_and_s = True
                            break
        
        # Inverse Head and Shoulders pattern (bottom reversal)
        inv_h_and_s = False
        
        # Check sequences of 3 swing lows with 2 swing highs in between
        for i in range(len(swing_lows) - 2):
            # Get 3 consecutive swing lows
            left_shoulder_idx, left_shoulder_date, left_shoulder = swing_lows[i]
            head_idx, head_date, head = swing_lows[i+1]
            right_shoulder_idx, right_shoulder_date, right_shoulder = swing_lows[i+2]
            
            # Check if the head is lower than both shoulders
            if (head < left_shoulder and head < right_shoulder):
                # Check if shoulders are approximately at the same level
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff <= shoulder_tolerance:
                    # Find the swing highs between the swing lows
                    neck_line_points = []
                    for high_idx, high_date, high in swing_highs:
                        if left_shoulder_idx < high_idx < head_idx or head_idx < high_idx < right_shoulder_idx:
                            neck_line_points.append((high_idx, high_date, high))
                    
                    # Need at least 2 highs to form a neckline
                    if len(neck_line_points) >= 2:
                        # Check if we've broken the neckline
                        neckline_level = sum(high for _, _, high in neck_line_points) / len(neck_line_points)
                        current_price = self.df['close'].iloc[-1]
                        
                        if current_price > neckline_level:
                            inv_h_and_s = True
                            break
        
        # Set signal if pattern is detected
        if h_and_s:
            self.signals.append({
                'pattern': 'Head and Shoulders',
                'signal': 'SELL',
                'strength': 4
            })
        
        if inv_h_and_s:
            self.signals.append({
                'pattern': 'Inverse Head and Shoulders',
                'signal': 'BUY',
                'strength': 4
            })
        
        return {
            'head_and_shoulders': h_and_s,
            'inverse_head_and_shoulders': inv_h_and_s
        }
    
    def detect_double_patterns(self):
        """
        Detect Double Top and Double Bottom patterns
        
        Returns:
            Dictionary with detected double top/bottom patterns
        """
        # Get parameters
        tolerance = self.params.get_chart_pattern_param('double_pattern_tolerance')
        lookback = self.params.get_chart_pattern_param('double_pattern_lookback')
        
        # Limit lookback to available data
        lookback = min(lookback, len(self.df) - 1)
        
        # Find swing highs and lows
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        # Need at least 2 swing points
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                'double_top': False,
                'double_bottom': False
            }
        
        # Double Top pattern
        double_top = False
        
        # Check recent swing highs for peaks at similar levels
        recent_highs = [high for idx, date, high in swing_highs if idx >= len(self.df) - lookback]
        if len(recent_highs) >= 2:
            # Compare last 2 swing highs
            high1 = recent_highs[-2]
            high2 = recent_highs[-1]
            
            # Check if the highs are at similar levels
            price_diff = abs(high1 - high2) / high1
            if price_diff <= tolerance:
                # Find the low between the two highs
                mid_low = min(self.df['low'].iloc[-lookback:])
                
                # Check if we've broken below the low between the two tops
                current_price = self.df['close'].iloc[-1]
                if current_price < mid_low:
                    double_top = True
        
        # Double Bottom pattern
        double_bottom = False
        
        # Check recent swing lows for troughs at similar levels
        recent_lows = [low for idx, date, low in swing_lows if idx >= len(self.df) - lookback]
        if len(recent_lows) >= 2:
            # Compare last 2 swing lows
            low1 = recent_lows[-2]
            low2 = recent_lows[-1]
            
            # Check if the lows are at similar levels
            price_diff = abs(low1 - low2) / low1
            if price_diff <= tolerance:
                # Find the high between the two lows
                mid_high = max(self.df['high'].iloc[-lookback:])
                
                # Check if we've broken above the high between the two bottoms
                current_price = self.df['close'].iloc[-1]
                if current_price > mid_high:
                    double_bottom = True
        
        # Set signal if pattern is detected
        if double_top:
            self.signals.append({
                'pattern': 'Double Top',
                'signal': 'SELL',
                'strength': 3
            })
        
        if double_bottom:
            self.signals.append({
                'pattern': 'Double Bottom',
                'signal': 'BUY',
                'strength': 3
            })
        
        return {
            'double_top': double_top,
            'double_bottom': double_bottom
        }
    
    def detect_triple_patterns(self):
        """
        Detect Triple Top and Triple Bottom patterns
        
        Returns:
            Dictionary with detected triple top/bottom patterns
        """
        # Get parameters
        tolerance = self.params.get_chart_pattern_param('triple_pattern_tolerance')
        lookback = self.params.get_chart_pattern_param('triple_pattern_lookback')
        
        # Limit lookback to available data
        lookback = min(lookback, len(self.df) - 1)
        
        # Find swing highs and lows
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        # Need at least 3 swing points
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {
                'triple_top': False,
                'triple_bottom': False
            }
        
        # Triple Top pattern
        triple_top = False
        
        # Check recent swing highs for 3 peaks at similar levels
        recent_highs = [high for idx, date, high in swing_highs if idx >= len(self.df) - lookback]
        if len(recent_highs) >= 3:
            # Compare last 3 swing highs
            high1 = recent_highs[-3]
            high2 = recent_highs[-2]
            high3 = recent_highs[-1]
            
            # Check if all 3 highs are at similar levels
            price_diff1 = abs(high1 - high2) / high1
            price_diff2 = abs(high2 - high3) / high2
            price_diff3 = abs(high1 - high3) / high1
            
            if price_diff1 <= tolerance and price_diff2 <= tolerance and price_diff3 <= tolerance:
                # Find the lowest low between the highs
                mid_low = min(self.df['low'].iloc[-lookback:])
                
                # Check if we've broken below the lowest low
                current_price = self.df['close'].iloc[-1]
                if current_price < mid_low:
                    triple_top = True
        
        # Triple Bottom pattern
        triple_bottom = False
        
        # Check recent swing lows for 3 troughs at similar levels
        recent_lows = [low for idx, date, low in swing_lows if idx >= len(self.df) - lookback]
        if len(recent_lows) >= 3:
            # Compare last 3 swing lows
            low1 = recent_lows[-3]
            low2 = recent_lows[-2]
            low3 = recent_lows[-1]
            
            # Check if all 3 lows are at similar levels
            price_diff1 = abs(low1 - low2) / low1
            price_diff2 = abs(low2 - low3) / low2
            price_diff3 = abs(low1 - low3) / low1
            
            if price_diff1 <= tolerance and price_diff2 <= tolerance and price_diff3 <= tolerance:
                # Find the highest high between the lows
                mid_high = max(self.df['high'].iloc[-lookback:])
                
                # Check if we've broken above the highest high
                current_price = self.df['close'].iloc[-1]
                if current_price > mid_high:
                    triple_bottom = True
        
        # Set signal if pattern is detected
        if triple_top:
            self.signals.append({
                'pattern': 'Triple Top',
                'signal': 'SELL',
                'strength': 4
            })
        
        if triple_bottom:
            self.signals.append({
                'pattern': 'Triple Bottom',
                'signal': 'BUY',
                'strength': 4
            })
        
        return {
            'triple_top': triple_top,
            'triple_bottom': triple_bottom
        }
    
    def detect_wedges(self):
        """
        Detect Rising and Falling Wedge patterns
        
        Returns:
            Dictionary with detected wedge patterns
        """
        # Get parameters
        lookback = min(100, len(self.df) - 1)  # Use last 100 candles or less
        
        # Find swing highs and lows
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        # Need at least 3 swing points of each kind
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {
                'rising_wedge': False,
                'falling_wedge': False
            }
        
        # Get recent swing points
        recent_highs = [(idx, date, high) for idx, date, high in swing_highs if idx >= len(self.df) - lookback]
        recent_lows = [(idx, date, low) for idx, date, low in swing_lows if idx >= len(self.df) - lookback]
        
        # Need at least 3 recent swing points of each kind
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            return {
                'rising_wedge': False,
                'falling_wedge': False
            }
        
        # Calculate trendlines on highs and lows
        high_x = [idx for idx, _, _ in recent_highs]
        high_y = [high for _, _, high in recent_highs]
        
        low_x = [idx for idx, _, _ in recent_lows]
        low_y = [low for _, _, low in recent_lows]
        
        # Calculate linear regression lines
        try:
            from scipy import stats
            high_slope, high_intercept, _, _, _ = stats.linregress(high_x, high_y)
            low_slope, low_intercept, _, _, _ = stats.linregress(low_x, low_y)
        except:
            # If scipy is not available, use a simple approach
            # This is not as robust but works for demonstration
            high_slope = (high_y[-1] - high_y[0]) / (high_x[-1] - high_x[0])
            high_intercept = high_y[0] - high_slope * high_x[0]
            
            low_slope = (low_y[-1] - low_y[0]) / (low_x[-1] - low_x[0])
            low_intercept = low_y[0] - low_slope * low_x[0]
        
        # Check if trendlines are converging
        trendlines_converging = ((high_slope < 0 and low_slope < 0 and high_slope < low_slope) or
                              (high_slope > 0 and low_slope > 0 and high_slope > low_slope))
        
        # Check if price is near the vertex of the wedge
        current_idx = len(self.df) - 1
        high_trendline_value = high_slope * current_idx + high_intercept
        low_trendline_value = low_slope * current_idx + low_intercept
        
        wedge_width = abs(high_trendline_value - low_trendline_value)
        avg_price = (self.df['high'].iloc[-1] + self.df['low'].iloc[-1]) / 2
        
        price_near_vertex = wedge_width < avg_price * 0.03  # Within 3% of average price
        
        # Determine wedge type
        rising_wedge = False
        falling_wedge = False
        
        if trendlines_converging and price_near_vertex:
            if high_slope > 0 and low_slope > 0:  # Both trendlines rising
                rising_wedge = True
            elif high_slope < 0 and low_slope < 0:  # Both trendlines falling
                falling_wedge = True
        
        # Check for breakout
        current_price = self.df['close'].iloc[-1]
        
        if rising_wedge and current_price < low_trendline_value:
            # Confirmed bearish breakout
            self.signals.append({
                'pattern': 'Rising Wedge',
                'signal': 'SELL',
                'strength': 4
            })
        elif falling_wedge and current_price > high_trendline_value:
            # Confirmed bullish breakout
            self.signals.append({
                'pattern': 'Falling Wedge',
                'signal': 'BUY',
                'strength': 4
            })
        
        return {
            'rising_wedge': rising_wedge,
            'falling_wedge': falling_wedge
        }
    
    def detect_rectangle(self):
        """
        Detect Rectangle (Trading Range) pattern
        
        Returns:
            Dictionary with detected rectangle pattern
        """
        # Get parameters
        range_lookback = self.params.get_chart_pattern_param('range_lookback')
        range_tolerance = self.params.get_chart_pattern_param('range_tolerance')
        min_touches = self.params.get_chart_pattern_param('range_min_touches')
        
        # Limit lookback to available data
        lookback = min(range_lookback, len(self.df) - 1)
        
        # Get recent data
        recent_data = self.df.iloc[-lookback:]
        
        # Identify potential support and resistance levels
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        swing_lows = swings['swing_lows']
        
        # Need enough swing points
        if len(swing_highs) < min_touches / 2 or len(swing_lows) < min_touches / 2:
            return {'rectangle': False}
        
        # Get recent swing points
        recent_highs = [high for idx, date, high in swing_highs if idx >= len(self.df) - lookback]
        recent_lows = [low for idx, date, low in swing_lows if idx >= len(self.df) - lookback]
        
        # Cluster the highs and lows to find potential resistance and support
        def cluster_levels(levels, tolerance):
            if not levels:
                return []
            
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # Check if this level is close to the current cluster
                if abs(level - current_cluster[0]) / current_cluster[0] <= tolerance:
                    current_cluster.append(level)
                else:
                    # Start a new cluster
                    if len(current_cluster) >= min_touches / 2:
                        # Calculate average level for the cluster
                        avg_level = sum(current_cluster) / len(current_cluster)
                        clusters.append((avg_level, len(current_cluster)))
                    current_cluster = [level]
            
            # Add the last cluster
            if len(current_cluster) >= min_touches / 2:
                avg_level = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_level, len(current_cluster)))
            
            return clusters
        
        # Cluster the highs and lows
        resistance_clusters = cluster_levels(recent_highs, range_tolerance)
        support_clusters = cluster_levels(recent_lows, range_tolerance)
        
        # Check if we have at least one strong support and resistance level
        if not resistance_clusters or not support_clusters:
            return {'rectangle': False}
        
        # Get the strongest levels (most touches)
        resistance_clusters.sort(key=lambda x: x[1], reverse=True)
        support_clusters.sort(key=lambda x: x[1], reverse=True)
        
        resistance_level = resistance_clusters[0][0]
        support_level = support_clusters[0][0]
        
        # Check if these levels form a range
        range_height = resistance_level - support_level
        range_midpoint = (resistance_level + support_level) / 2
        
        if range_height / range_midpoint < 0.15:  # Range is too narrow
            return {'rectangle': False}
        
        # Check total number of touches
        total_touches = resistance_clusters[0][1] + support_clusters[0][1]
        
        # Rectangle pattern confirmed if we have enough touches
        rectangle = total_touches >= min_touches
        
        # Check for breakout
        current_price = self.df['close'].iloc[-1]
        prev_close = self.df['close'].iloc[-2]
        
        if rectangle:
            if current_price > resistance_level and prev_close <= resistance_level:
                # Bullish breakout
                self.signals.append({
                    'pattern': 'Rectangle Breakout',
                    'signal': 'BUY',
                    'strength': 3
                })
            elif current_price < support_level and prev_close >= support_level:
                # Bearish breakout
                self.signals.append({
                    'pattern': 'Rectangle Breakdown',
                    'signal': 'SELL',
                    'strength': 3
                })
        
        return {'rectangle': rectangle}
    
    def detect_flags(self):
        """
        Detect Flag and Pennant patterns
        
        Returns:
            Dictionary with detected flag/pennant patterns
        """
        # Get parameters
        lookback = self.params.get_chart_pattern_param('flag_lookback')
        pole_threshold = self.params.get_chart_pattern_param('flag_pole_threshold')
        consolidation_threshold = self.params.get_chart_pattern_param('flag_consolidation_threshold')
        max_bars = self.params.get_chart_pattern_param('flag_max_bars')
        
        # Limit lookback to available data
        lookback = min(lookback, len(self.df) - 1)
        
        # Need enough data
        if len(self.df) < lookback + max_bars:
            return {
                'bullish_flag': False,
                'bearish_flag': False,
                'bullish_pennant': False,
                'bearish_pennant': False
            }
        
        # Get recent data
        recent_data = self.df.iloc[-lookback:]
        
        # Find strong trend moves (poles)
        bullish_pole = False
        bearish_pole = False
        pole_start_idx = None
        pole_end_idx = None
        
        # Look for a strong price move in a short period
        for i in range(len(recent_data) - max_bars):
            # Check window of bars for a strong move
            window = recent_data.iloc[i:i+max_bars]
            start_price = window['open'].iloc[0]
            end_price = window['close'].iloc[-1]
            
            price_change = (end_price - start_price) / start_price
            
            if price_change > pole_threshold:  # Strong bullish move
                bullish_pole = True
                pole_start_idx = i
                pole_end_idx = i + max_bars - 1
                break
            elif price_change < -pole_threshold:  # Strong bearish move
                bearish_pole = True
                pole_start_idx = i
                pole_end_idx = i + max_bars - 1
                break
        
        if not bullish_pole and not bearish_pole:
            return {
                'bullish_flag': False,
                'bearish_flag': False,
                'bullish_pennant': False,
                'bearish_pennant': False
            }
        
        # Check for consolidation after the pole
        consolidation_start = pole_end_idx + 1
        if consolidation_start >= len(recent_data):
            return {
                'bullish_flag': False,
                'bearish_flag': False,
                'bullish_pennant': False,
                'bearish_pennant': False
            }
        
        # Get consolidation data
        consolidation_data = recent_data.iloc[consolidation_start:]
        
        # Check if consolidation range is small compared to the pole
        pole_range = recent_data['high'].iloc[pole_start_idx:pole_end_idx+1].max() - recent_data['low'].iloc[pole_start_idx:pole_end_idx+1].min()
        consolidation_range = consolidation_data['high'].max() - consolidation_data['low'].min()
        
        consolidation_ratio = consolidation_range / pole_range
        
        if consolidation_ratio > consolidation_threshold:
            return {
                'bullish_flag': False,
                'bearish_flag': False,
                'bullish_pennant': False,
                'bearish_pennant': False
            }
        
        # Calculate trendlines for the consolidation
        consolidation_highs = consolidation_data['high'].values
        consolidation_lows = consolidation_data['low'].values
        consolidation_indices = np.arange(len(consolidation_highs))
        
        # Calculate linear regression lines for highs and lows
        try:
            from scipy import stats
            high_slope, high_intercept, _, _, _ = stats.linregress(consolidation_indices, consolidation_highs)
            low_slope, low_intercept, _, _, _ = stats.linregress(consolidation_indices, consolidation_lows)
        except:
            # Simple alternative if scipy isn't available
            high_slope = (consolidation_highs[-1] - consolidation_highs[0]) / (len(consolidation_highs) - 1)
            high_intercept = consolidation_highs[0]
            
            low_slope = (consolidation_lows[-1] - consolidation_lows[0]) / (len(consolidation_lows) - 1)
            low_intercept = consolidation_lows[0]
        
        # Check if slopes are similar (pennant) or opposites (flag)
        slopes_similar = abs(high_slope - low_slope) < 0.0001  # Nearly parallel
        slopes_opposite = (high_slope * low_slope) < 0  # Opposite signs
        
        # Determine pattern type
        bullish_flag = bullish_pole and slopes_opposite and high_slope < 0
        bearish_flag = bearish_pole and slopes_opposite and high_slope > 0
        
        bullish_pennant = bullish_pole and slopes_similar
        bearish_pennant = bearish_pole and slopes_similar
        
        # Check for breakout
        current_price = self.df['close'].iloc[-1]
        prev_close = self.df['close'].iloc[-2]
        
        if bullish_flag or bullish_pennant:
            # Calculate upper trendline value at current point
            upper_trendline = high_intercept + high_slope * (len(consolidation_highs) - 1)
            
            if current_price > upper_trendline and prev_close <= upper_trendline:
                # Bullish breakout
                self.signals.append({
                    'pattern': 'Bullish Flag' if bullish_flag else 'Bullish Pennant',
                    'signal': 'BUY',
                    'strength': 4
                })
        
        if bearish_flag or bearish_pennant:
            # Calculate lower trendline value at current point
            lower_trendline = low_intercept + low_slope * (len(consolidation_lows) - 1)
            
            if current_price < lower_trendline and prev_close >= lower_trendline:
                # Bearish breakout
                self.signals.append({
                    'pattern': 'Bearish Flag' if bearish_flag else 'Bearish Pennant',
                    'signal': 'SELL',
                    'strength': 4
                })
        
        return {
            'bullish_flag': bullish_flag,
            'bearish_flag': bearish_flag,
            'bullish_pennant': bullish_pennant,
            'bearish_pennant': bearish_pennant
        }
    
    def detect_cup_and_handle(self):
        """
        Detect Cup and Handle pattern
        
        Returns:
            Dictionary with detected cup and handle pattern
        """
        # Get parameters
        cup_depth_threshold = self.params.get_chart_pattern_param('cup_depth_threshold')
        volume_confirmation = self.params.get_chart_pattern_param('cup_volume_confirmation')
        
        # Need enough data for a cup pattern (at least 30 bars)
        if len(self.df) < 30:
            return {'cup_and_handle': False}
        
        # Find swing highs and lows
        swings = self.find_swing_points()
        swing_highs = swings['swing_highs']
        
        # Need at least 2 swing highs for cup formation
        if len(swing_highs) < 2:
            return {'cup_and_handle': False}
        
        # Get recent swing highs
        recent_highs = swing_highs[-5:]  # Look at last 5 swing highs
        
        cup_and_handle = False
        
        # Analyze potential cup formations
        for i in range(len(recent_highs) - 1):
            # Get left and right rims of the cup
            left_rim_idx, left_rim_date, left_rim = recent_highs[i]
            right_rim_idx, right_rim_date, right_rim = recent_highs[i+1]
            
            # Check if rims are at similar levels
            price_diff = abs(left_rim - right_rim) / left_rim
            if price_diff <= 0.03:  # Rims within 3% of each other
                # Check for a rounded bottom between the rims
                cup_section = self.df.iloc[left_rim_idx:right_rim_idx+1]
                
                # Find the lowest point in the cup
                cup_bottom = cup_section['low'].min()
                cup_bottom_idx = cup_section['low'].idxmin()
                
                # Check cup depth (should be 10%-50% of cup height)
                cup_height = left_rim - cup_bottom
                cup_depth_ratio = cup_height / left_rim
                
                if cup_depth_ratio >= cup_depth_threshold and cup_depth_ratio <= 0.5:
                    # Check that the bottom is rounded (not V-shaped)
                    # This is a simplified check
                    mid_idx = (left_rim_idx + right_rim_idx) // 2
                    mid_price = self.df['close'].iloc[mid_idx]
                    
                    is_rounded = abs(mid_price - cup_bottom) / cup_height < 0.3
                    
                    if is_rounded:
                        # Look for a handle after the right rim
                        if right_rim_idx < len(self.df) - 5:
                            handle_section = self.df.iloc[right_rim_idx:right_rim_idx+10]  # 10 bars for handle
                            
                            # Handle should be a small pullback (less than 50% of cup depth)
                            handle_low = handle_section['low'].min()
                            handle_pullback = right_rim - handle_low
                            
                            if handle_pullback / cup_height <= 0.5 and handle_pullback > 0:
                                # Check volume if required
                                volume_ok = True
                                if volume_confirmation and 'volume' in self.df.columns:
                                    # Volume should decline in the cup and increase in the handle
                                    cup_start_volume = self.df['volume'].iloc[left_rim_idx]
                                    cup_bottom_volume = self.df['volume'].iloc[cup_bottom_idx]
                                    handle_volume = handle_section['volume'].mean()
                                    
                                    volume_ok = (cup_bottom_volume < cup_start_volume and 
                                                handle_volume > cup_bottom_volume)
                                
                                if volume_ok:
                                    # Check for breakout above right rim
                                    current_price = self.df['close'].iloc[-1]
                                    prev_close = self.df['close'].iloc[-2]
                                    
                                    if current_price > right_rim and prev_close <= right_rim:
                                        cup_and_handle = True
                                        self.signals.append({
                                            'pattern': 'Cup and Handle',
                                            'signal': 'BUY',
                                            'strength': 4
                                        })
                                        break
        
        return {'cup_and_handle': cup_and_handle}
    
    def detect_rounding_patterns(self):
        """
        Detect Rounding Top and Rounding Bottom patterns
        
        Returns:
            Dictionary with detected rounding patterns
        """
        # Get parameters
        curve_smoothness = self.params.get_chart_pattern_param('rounding_curve_smoothness')
        min_points = self.params.get_chart_pattern_param('rounding_min_points')
        
        # Need enough data
        if len(self.df) < min_points:
            return {
                'rounding_top': False,
                'rounding_bottom': False
            }
        
        # Check for rounding bottom
        rounding_bottom = False
        
        # Get recent lows
        recent_lows = self.df['low'].iloc[-min_points:].values
        
        # Calculate fit to curved bottom
        try:
            # If numpy/scipy are available, use polynomial fit
            x = np.arange(len(recent_lows))
            z = np.polyfit(x, recent_lows, 2)
            p = np.poly1d(z)
            
            # The coefficient of the squared term indicates curvature
            curve_coef = z[0]
            
            # Positive coefficient indicates upward curvature
            is_curved_up = curve_coef > 0
            
            # Calculate fit quality
            fit_values = p(x)
            residuals = recent_lows - fit_values
            fit_quality = 1 - (np.sum(residuals**2) / ((recent_lows - np.mean(recent_lows))**2).sum())
            
            # Good fit and upward curvature indicates rounding bottom
            if is_curved_up and fit_quality > curve_smoothness:
                # Check if price is rising after the bottom
                current_price = self.df['close'].iloc[-1]
                lowest_point = recent_lows.min()
                
                if current_price > lowest_point * 1.03:  # Price at least 3% above bottom
                    rounding_bottom = True
                    self.signals.append({
                        'pattern': 'Rounding Bottom',
                        'signal': 'BUY',
                        'strength': 3
                    })
        except:
            # Simple alternative if numpy/scipy not available
            pass
        
        # Check for rounding top
        rounding_top = False
        
        # Get recent highs
        recent_highs = self.df['high'].iloc[-min_points:].values
        
        # Calculate fit to curved top
        try:
            # If numpy/scipy are available, use polynomial fit
            x = np.arange(len(recent_highs))
            z = np.polyfit(x, recent_highs, 2)
            p = np.poly1d(z)
            
            # The coefficient of the squared term indicates curvature
            curve_coef = z[0]
            
            # Negative coefficient indicates downward curvature
            is_curved_down = curve_coef < 0
            
            # Calculate fit quality
            fit_values = p(x)
            residuals = recent_highs - fit_values
            fit_quality = 1 - (np.sum(residuals**2) / ((recent_highs - np.mean(recent_highs))**2).sum())
            
            # Good fit and downward curvature indicates rounding top
            if is_curved_down and fit_quality > curve_smoothness:
                # Check if price is falling after the top
                current_price = self.df['close'].iloc[-1]
                highest_point = recent_highs.max()
                
                if current_price < highest_point * 0.97:  # Price at least 3% below top
                    rounding_top = True
                    self.signals.append({
                        'pattern': 'Rounding Top',
                        'signal': 'SELL',
                        'strength': 3
                    })
        except:
            # Simple alternative if numpy/scipy not available
            pass
        
        return {
            'rounding_top': rounding_top,
            'rounding_bottom': rounding_bottom
        }
    
    def get_signals(self):
        """
        Get all chart pattern signals
        
        Returns:
            List of all detected pattern signals
        """
        # Detect all patterns if not already detected
        if not self.patterns:
            self.detect_all_patterns()
        
        return self.signals
    
    def get_latest_patterns(self):
        """
        Get the latest detected patterns
        
        Returns:
            Dictionary of patterns found
        """
        # Detect all patterns if not already detected
        if not self.patterns:
            self.detect_all_patterns()
        
        # Return patterns with True values
        return {pattern: True for pattern, value in self.patterns.items() if value}
    
    def get_pattern_signals(self):
        """
        Get signals from detected patterns
        
        Returns:
            Dictionary with buy and sell signals
        """
        # Get signals
        signals = self.get_signals()
        
        # Separate buy and sell signals
        buy_signals = []
        sell_signals = []
        
        for signal in signals:
            if signal['signal'] == 'BUY':
                buy_signals.append({
                    'pattern': signal['pattern'],
                    'strength': signal['strength']
                })
            elif signal['signal'] == 'SELL':
                sell_signals.append({
                    'pattern': signal['pattern'],
                    'strength': signal['strength']
                })
        
        return {
            'buy': buy_signals,
            'sell': sell_signals
        }

# ===============================================================
# Trading Signal and Strategy Logic
# ===============================================================
class TradingSignalBot:
    """Main bot class for generating trading signals"""

    async def run_backtest(self, instrument_key, days=250, strategy=None):
        """
        Run backtest for a specific instrument
        
        Args:
            instrument_key: Instrument to backtest
            days: Number of days of historical data to use
            strategy: Optional custom strategy function
            
        Returns:
            Backtest report
        """
        try:
            # Fetch historical data
            self.logger.info(f"Fetching historical data for backtest of {instrument_key}")
            df = await self.fetch_historical_data(instrument_key, days=days)
            
            # Ensure we have enough data
            if len(df) < 50:
                self.logger.warning(f"Not enough historical data for {instrument_key}")
                return {
                    'instrument_key': instrument_key,
                    'error': 'Not enough historical data for backtesting'
                }
                
            # Calculate indicators
            indicators = TechnicalIndicators(df, self.params)
            
            # Run backtest
            self.logger.info(f"Running backtest for {instrument_key}")
            backtest_results = indicators.backtest_strategy(lookback_days=days, backtesting_func=strategy)
            
            # Get instrument details for report
            instrument_details = None
            try:
                if self.client:
                    instrument_details = self.client.get_instrument_details(instrument_key)
            except Exception as e:
                self.logger.warning(f"Error getting instrument details: {e}")
                
            # Get stock info
            stock_info = get_stock_info_by_key(instrument_key, self.stock_info_cache)
            
            # Enhance report with instrument details
            report = backtest_results['report']
            report['instrument_key'] = instrument_key
            report['stock_name'] = stock_info.get('name', 'Unknown')
            report['stock_symbol'] = stock_info.get('symbol', instrument_key)
            
            # Add monte carlo simulation results
            monte_carlo = backtest_results['backtest_engine'].monte_carlo_simulation(num_simulations=500)
            report['monte_carlo'] = {
                'median_return': f"{monte_carlo['confidence_intervals']['return_50pct'] * 100:.2f}%",
                'return_range': f"{monte_carlo['confidence_intervals']['return_5pct'] * 100:.2f}% to {monte_carlo['confidence_intervals']['return_95pct'] * 100:.2f}%",
                'median_drawdown': f"{monte_carlo['confidence_intervals']['drawdown_50pct'] * 100:.2f}%",
                'drawdown_range': f"{monte_carlo['confidence_intervals']['drawdown_5pct'] * 100:.2f}% to {monte_carlo['confidence_intervals']['drawdown_95pct'] * 100:.2f}%",
            }
            
            self.logger.info(f"Backtest completed for {instrument_key}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error in backtest for {instrument_key}: {e}")
            self.logger.error(traceback.format_exc())
            
            return {
                'instrument_key': instrument_key,
                'error': str(e)
            }
    
    def __init__(self, config=None):
        """
        Initialize the trading signal bot
        
        Args:
            config: Configuration dictionary
        """
        # Set up logging
        self.logger = setup_logging()
        self.logger.info(f"Initializing Trading Signal Bot - Version 4.0.0")
        
        # Load configuration
        self.config = config or {}
        
        # Initialize trading parameters
        self.params = TradingParameters(self.config)
        
        # Initialize Upstox client
        try:
            self.client = UpstoxClient(self.config)
            self.logger.info("Upstox client initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Upstox client: {e}")
            self.client = None
        
        # Cache for storing stock information
        self.stock_info_cache = {}
        
        # Cache for storing historical data
        self.data_cache = {}
        
        # Initialize results storage
        self.results = {}
        
        self.logger.info("Trading Signal Bot initialized successfully")
    
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
        # Check if data is in cache
        cache_key = f"{instrument_key}_{interval}_{days}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Authenticate client if needed
        if self.client is None:
            self.logger.error("Upstox client not available")
            raise APIConnectionError("Upstox client not available")
        
        if not self.client.authenticate():
            self.logger.error("Failed to authenticate with Upstox")
            raise APIConnectionError("Failed to authenticate with Upstox")
        
        # Calculate date range
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            # Fetch data from Upstox
            self.logger.info(f"Fetching historical data for {instrument_key}, interval {interval}")
            df = self.client.get_historical_data(instrument_key, interval, from_date, to_date)
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise DataFetchError(instrument_key, cause=e)
    
    async def generate_signals(self, instrument_key):
        """
        Generate trading signals for a given instrument
        
        Args:
            instrument_key: Instrument identifier
            
        Returns:
            Dictionary with trading signals and analysis
        """
        try:
            # Fetch historical data
            self.logger.info(f"Generating signals for {instrument_key}")
            df = await self.fetch_historical_data(instrument_key)
            
            # Ensure we have enough data
            if len(df) < 30:
                self.logger.warning(f"Not enough historical data for {instrument_key}")
                return {
                    'instrument_key': instrument_key,
                    'error': 'Not enough historical data',
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Calculate technical indicators
            self.logger.info(f"Calculating technical indicators for {instrument_key}")
            indicators = TechnicalIndicators(df, self.params)
            indicator_signals = indicators.calculate_all()
            
            # Detect candlestick patterns
            self.logger.info(f"Detecting candlestick patterns for {instrument_key}")
            patterns = CandlestickPatterns(df, self.params)
            pattern_signals = patterns.get_pattern_signals()
            
            # Detect chart patterns
            self.logger.info(f"Detecting chart patterns for {instrument_key}")
            chart = ChartPatterns(df, self.params)
            chart_signals = chart.get_pattern_signals()
            
            # Get overall signal
            overall = indicators.get_overall_signal()
            
            # Get detailed analysis with trade setup
            detailed = indicators.get_detailed_analysis()
            
            # Get instrument details
            instrument_details = None
            try:
                if self.client:
                    instrument_details = self.client.get_instrument_details(instrument_key)
            except Exception as e:
                self.logger.warning(f"Error getting instrument details: {e}")
            
            # Get stock info (name, industry, etc.)
            stock_info = get_stock_info_by_key(instrument_key, self.stock_info_cache)
            
            # Combine all signals and analysis
            result = {
                'instrument_key': instrument_key,
                'stock_name': stock_info.get('name', 'Unknown'),
                'stock_symbol': stock_info.get('symbol', instrument_key),
                'industry': stock_info.get('industry', 'Unknown'),
                'current_price': detailed['current_price'],
                'signal': overall['signal'],
                'signal_strength': overall['strength'],
                'confidence': overall['confidence'],
                'indicator_signals': indicator_signals,
                'candlestick_patterns': pattern_signals,
                'chart_patterns': chart_signals,
                'trade_setup': {
                    'entry_price': detailed['entry_price'],
                    'stop_loss': detailed['stop_loss'],
                    'stop_loss_pct': detailed['stop_loss_pct'],
                    'target': detailed['target'],
                    'target_pct': detailed['target_pct'],
                    'reward_risk_ratio': detailed['reward_risk_ratio'],
                    'support_levels': detailed['support_levels'],
                    'resistance_levels': detailed['resistance_levels'],
                    'checklist': detailed['checklist']
                },
                'summary': overall['summary'],
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add trend analysis
            trend_analysis = ""
            if 'moving_averages' in indicators.indicators:
                ma = indicators.indicators['moving_averages']['values']
                
                if ma.get('uptrend', False):
                    trend_analysis += "Strong uptrend detected: price above key moving averages. "
                elif ma.get('downtrend', False):
                    trend_analysis += "Strong downtrend detected: price below key moving averages. "
                elif ma.get('price_above_sma_long', False):
                    trend_analysis += "Long-term trend is bullish but short-term momentum may be weakening. "
                elif not ma.get('price_above_sma_long', True):
                    trend_analysis += "Long-term trend is bearish but short-term momentum may be improving. "
                else:
                    trend_analysis += "Mixed trend signals with no clear direction. "
            
            result['trend_analysis'] = trend_analysis
            
            self.results[instrument_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {instrument_key}: {e}")
            self.logger.error(traceback.format_exc())
            
            return {
                'instrument_key': instrument_key,
                'error': str(e),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    async def generate_all_signals(self):
        """
        Generate signals for all stocks in the configuration
        
        Returns:
            Dictionary with results for all stocks
        """
        # Get list of stocks from config
        stock_list = self.config.get('STOCK_LIST', [])
        
        if not stock_list:
            self.logger.warning("No stocks defined in configuration")
            return {}
        
        self.logger.info(f"Generating signals for {len(stock_list)} stocks")
        
        # Process each stock
        all_results = {}
        
        for instrument_key in stock_list:
            try:
                result = await self.generate_signals(instrument_key)
                all_results[instrument_key] = result
            except Exception as e:
                self.logger.error(f"Error processing {instrument_key}: {e}")
                all_results[instrument_key] = {
                    'instrument_key': instrument_key,
                    'error': str(e),
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return all_results
    
    async def send_signal_alert(self, result):
        """
        Send a signal alert via Telegram
        
        Args:
            result: Signal result dictionary
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        # Skip if telegram alerts are disabled
        if not self.config.get('ENABLE_TELEGRAM_ALERTS', False):
            return False
        
        # Skip if no signal or weak signal
        signal = result.get('signal', 'NEUTRAL')
        if signal == 'NEUTRAL':
            return False
        
        # Get message template
        template = self.config.get('SIGNAL_MESSAGE_TEMPLATE', '')
        if not template:
            template = """
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
        
        try:
            # Format signal strength as stars
            signal_strength = result.get('signal_strength', 0)
            signal_strength_stars = "⭐" * signal_strength
            
            # Format primary indicators
            primary_indicators = []
            
            # Add important indicator signals
            for indicator in ['Moving Averages', 'MACD', 'RSI', 'Supertrend']:
                for signal_item in result.get('indicator_signals', {}).get('buy_signals', []):
                    if signal_item['indicator'] == indicator:
                        primary_indicators.append(f"• {indicator}: {signal_item['name']} (BUY)")
                
                for signal_item in result.get('indicator_signals', {}).get('sell_signals', []):
                    if signal_item['indicator'] == indicator:
                        primary_indicators.append(f"• {indicator}: {signal_item['name']} (SELL)")
            
            if not primary_indicators:
                primary_indicators.append("No significant indicator signals")
            
            # Format patterns
            patterns = []
            
            # Add candlestick patterns
            for pattern in result.get('candlestick_patterns', {}).get('buy', []):
                patterns.append(f"• {pattern['pattern']} (BUY)")
            
            for pattern in result.get('candlestick_patterns', {}).get('sell', []):
                patterns.append(f"• {pattern['pattern']} (SELL)")
            
            # Add chart patterns
            for pattern in result.get('chart_patterns', {}).get('buy', []):
                patterns.append(f"• {pattern['pattern']} (BUY)")
            
            for pattern in result.get('chart_patterns', {}).get('sell', []):
                patterns.append(f"• {pattern['pattern']} (SELL)")
            
            if not patterns:
                patterns.append("No significant patterns detected")
            
            # Create trade setup summary
            trade_setup = result.get('trade_setup', {})
            
            # Create buy/sell summary
            buy_sell_summary = ""
            if signal in ['BUY', 'STRONG BUY']:
                buy_sell_summary = f"⬆️ *BUY RECOMMENDATION* ⬆️\nConfidence: {result.get('confidence', 0)}%"
            elif signal in ['SELL', 'STRONG SELL']:
                buy_sell_summary = f"⬇️ *SELL RECOMMENDATION* ⬇️\nConfidence: {result.get('confidence', 0)}%"
            
            # Format timestamp
            timestamp = result.get('timestamp', '')
            timestamp_short = timestamp.split()[0] if timestamp else ''
            
            # Format the message
            message = template.format(
                stock_name=escape_telegram_markdown(result.get('stock_name', 'Unknown')),
                stock_symbol=escape_telegram_markdown(result.get('stock_symbol', '')),
                current_price=escape_telegram_markdown(str(result.get('current_price', 'N/A'))),
                industry=escape_telegram_markdown(result.get('industry', 'Unknown')),
                signal_type=escape_telegram_markdown(signal),
                signal_strength_stars=escape_telegram_markdown(signal_strength_stars),
                primary_indicators=escape_telegram_markdown("\n".join(primary_indicators)),
                patterns=escape_telegram_markdown("\n".join(patterns)),
                entry_price=escape_telegram_markdown(str(trade_setup.get('entry_price', 'N/A'))),
                stop_loss=escape_telegram_markdown(str(trade_setup.get('stop_loss', 'N/A'))),
                stop_loss_pct=escape_telegram_markdown(str(trade_setup.get('stop_loss_pct', 'N/A'))),
                target_price=escape_telegram_markdown(str(trade_setup.get('target', 'N/A'))),
                target_pct=escape_telegram_markdown(str(trade_setup.get('target_pct', 'N/A'))),
                risk_reward_ratio=escape_telegram_markdown(str(trade_setup.get('reward_risk_ratio', 'N/A'))),
                trend_strength=escape_telegram_markdown(result.get('trend_analysis', 'N/A')),
                buy_sell_summary=escape_telegram_markdown(buy_sell_summary),
                timestamp_short=escape_telegram_markdown(timestamp_short)
            )
            
            # Send the message
            success = await send_telegram_message(message, self.config, self.logger)
            
            if success:
                self.logger.info(f"Signal alert sent for {result.get('stock_symbol', 'Unknown')}")
            else:
                self.logger.warning(f"Failed to send signal alert for {result.get('stock_symbol', 'Unknown')}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending signal alert: {e}")
            return False
    
    async def send_daily_report(self, results):
        """
        Send a daily summary report via Telegram
        
        Args:
            results: Dictionary of all signal results
            
        Returns:
            True if report was sent successfully, False otherwise
        """
        # Skip if telegram alerts are disabled
        if not self.config.get('ENABLE_DAILY_REPORT', False):
            return False
        
        try:
            # Count signals by type
            buy_count = 0
            strong_buy_count = 0
            sell_count = 0
            strong_sell_count = 0
            neutral_count = 0
            error_count = 0
            
            # Lists for tracking stocks by signal
            buy_stocks = []
            sell_stocks = []
            
            for key, result in results.items():
                signal = result.get('signal', 'NEUTRAL')
                
                if 'error' in result:
                    error_count += 1
                    continue
                
                if signal == 'STRONG BUY':
                    strong_buy_count += 1
                    buy_stocks.append(result.get('stock_symbol', 'Unknown'))
                elif signal == 'BUY':
                    buy_count += 1
                    buy_stocks.append(result.get('stock_symbol', 'Unknown'))
                elif signal == 'STRONG SELL':
                    strong_sell_count += 1
                    sell_stocks.append(result.get('stock_symbol', 'Unknown'))
                elif signal == 'SELL':
                    sell_count += 1
                    sell_stocks.append(result.get('stock_symbol', 'Unknown'))
                else:
                    neutral_count += 1
            
            # Prepare message
            message = f"""
📊 *DAILY TRADING SIGNALS SUMMARY* 📊

*{datetime.datetime.now().strftime('%Y-%m-%d')}*

*Signal Breakdown:*
✅ Strong Buy: {strong_buy_count}
✅ Buy: {buy_count}
➖ Neutral: {neutral_count}
❌ Sell: {sell_count}
❌ Strong Sell: {strong_sell_count}
⚠️ Errors: {error_count}

*Top Buy Signals:*
{', '.join(buy_stocks[:5])}

*Top Sell Signals:*
{', '.join(sell_stocks[:5])}

*Market Analysis:*
{"Bullish bias detected" if (buy_count + strong_buy_count) > (sell_count + strong_sell_count) else "Bearish bias detected" if (sell_count + strong_sell_count) > (buy_count + strong_buy_count) else "Mixed signals, no clear direction"}

⏰ Generated: {datetime.datetime.now().strftime('%H:%M:%S')}
"""
            
            # Send the message
            success = await send_telegram_message(message, self.config, self.logger)
            
            if success:
                self.logger.info("Daily report sent successfully")
            else:
                self.logger.warning("Failed to send daily report")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending daily report: {e}")
            return False
    
    async def run(self):
        """
        Run the trading signal bot
        
        Returns:
            Dictionary with all results
        """
        try:
            self.logger.info("Starting trading signal bot run")
            
            # Generate signals for all stocks
            all_results = await self.generate_all_signals()
            
            # Send alerts for strong signals
            for _, result in all_results.items():
                signal = result.get('signal', 'NEUTRAL')
                
                # Skip if neutral or error
                if signal == 'NEUTRAL' or 'error' in result:
                    continue
                
                # Send alert for significant signals
                await self.send_signal_alert(result)
            
            # Send daily report
            await self.send_daily_report(all_results)
            
            self.logger.info("Completed trading signal bot run")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error running trading signal bot: {e}")
            self.logger.error(traceback.format_exc())
            raise

# ===============================================================
# Class for Technical Analysis
# ===============================================================
class TechnicalAnalysis:
    """Unified class combining all technical analysis functions"""
    
    def __init__(self, df, params=None):
        """
        Initialize with DataFrame containing OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            params: Optional TradingParameters instance
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Initialize parameters (use defaults if not provided)
        self.params = params or TradingParameters()
        
        # Initialize analysis components
        self.indicators = TechnicalIndicators(self.df, self.params)
        self.candlesticks = CandlestickPatterns(self.df, self.params)
        self.chart_patterns = ChartPatterns(self.df, self.params)
        
        # Initialize results
        self.results = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def run_analysis(self):
        """
        Run complete technical analysis
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Calculate technical indicators
            self.logger.info("Calculating technical indicators")
            indicator_results = self.indicators.calculate_all()
            
            # Detect candlestick patterns
            self.logger.info("Detecting candlestick patterns")
            candlestick_patterns = self.candlesticks.get_latest_patterns()
            
            # Detect chart patterns
            self.logger.info("Detecting chart patterns")
            chart_patterns = self.chart_patterns.get_latest_patterns()
            
            # Get signals from each component
            indicator_signals = self.indicators.get_signals()
            candlestick_signals = self.candlesticks.get_pattern_signals()
            chart_pattern_signals = self.chart_patterns.get_pattern_signals()
            
            # Get overall signal
            overall = self.indicators.get_overall_signal()
            detailed = self.indicators.get_detailed_analysis()
            
            # Combine all results
            self.results = {
                'indicators': indicator_results,
                'candlestick_patterns': candlestick_patterns,
                'chart_patterns': chart_patterns,
                'indicator_signals': indicator_signals,
                'candlestick_signals': candlestick_signals,
                'chart_pattern_signals': chart_pattern_signals,
                'overall_signal': overall,
                'detailed_analysis': detailed,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error running technical analysis: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def get_trade_recommendation(self):
        """
        Get actionable trade recommendation
        
        Returns:
            Dictionary with trade recommendation
        """
        # Ensure analysis has been run
        if not self.results:
            self.run_analysis()
        
        # Get detailed analysis
        detailed = self.results.get('detailed_analysis', {})
        
        # Get overall signal
        signal = detailed.get('signal', 'NEUTRAL')
        
        # Determine action
        action = "NO ACTION"
        if signal in ['BUY', 'STRONG BUY']:
            action = "BUY"
        elif signal in ['SELL', 'STRONG SELL']:
            action = "SELL"
        
        # Get current price
        current_price = detailed.get('current_price')
        
        # Get stop loss and target
        stop_loss = detailed.get('stop_loss')
        target = detailed.get('target')
        
        # Get risk-reward ratio
        rrr = detailed.get('reward_risk_ratio')
        
        # Check if recommendation meets minimum criteria
        valid_recommendation = (
            action != "NO ACTION" and
            stop_loss is not None and
            target is not None and
            rrr >= self.params.get_signal_param('min_rrr')
        )
        
        # Prepare recommendation
        recommendation = {
            'action': action,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward_ratio': rrr,
            'valid': valid_recommendation,
            'confidence': detailed.get('confidence', 0),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return recommendation
