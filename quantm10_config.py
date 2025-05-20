"""
Configuration manager for QuantM10

Handles loading configuration from environment variables, files,
and provides validation of configuration values.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API credentials and connection settings"""
    # Upstox credentials
    upstox_api_key: str = field(default_factory=lambda: os.environ.get("UPSTOX_API_KEY", ""))
    upstox_api_secret: str = field(default_factory=lambda: os.environ.get("UPSTOX_API_SECRET", ""))
    upstox_redirect_uri: str = field(default_factory=lambda: os.environ.get("UPSTOX_REDIRECT_URI", "https://localhost"))
    
    # Telegram configuration
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    enable_telegram_alerts: bool = True
    enable_daily_report: bool = True


@dataclass
class ScheduleConfig:
    """Scheduling configuration"""
    market_open: str = "09:15"
    mid_day: str = "12:30"
    pre_close: str = "15:00"
    post_market: str = "16:15"
    run_on_weekends: bool = False
    run_on_startup: bool = True


@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    short_term_lookback: int = 120
    medium_term_lookback: int = 250
    long_term_lookback: int = 500
    intervals: Dict[str, str] = field(default_factory=lambda: {
        "short_term": "1D",
        "long_term": "1W"
    })
    minimum_signal_strength: int = 6


@dataclass
class TechnicalIndicatorConfig:
    """Technical indicator parameters"""
    moving_averages: Dict[str, int] = field(default_factory=dict)
    macd: Dict[str, Any] = field(default_factory=dict)
    rsi: Dict[str, Any] = field(default_factory=dict)
    stochastic: Dict[str, Any] = field(default_factory=dict)
    bollinger_bands: Dict[str, Any] = field(default_factory=dict)
    supertrend: Dict[str, Any] = field(default_factory=dict)
    parabolic_sar: Dict[str, Any] = field(default_factory=dict)
    aroon: Dict[str, Any] = field(default_factory=dict)
    adx: Dict[str, Any] = field(default_factory=dict)
    atr: Dict[str, Any] = field(default_factory=dict)
    roc: Dict[str, Any] = field(default_factory=dict)
    obv: Dict[str, Any] = field(default_factory=dict)
    vwap: Dict[str, Any] = field(default_factory=dict)
    cpr: Dict[str, Any] = field(default_factory=dict)
    stochastic_rsi: Dict[str, Any] = field(default_factory=dict)
    williams_r: Dict[str, Any] = field(default_factory=dict)
    ultimate_oscillator: Dict[str, Any] = field(default_factory=dict)
    cmf: Dict[str, Any] = field(default_factory=dict)
    vix_analysis: Dict[str, Any] = field(default_factory=dict)
    divergence: Dict[str, Any] = field(default_factory=dict)
    support_resistance: Dict[str, Any] = field(default_factory=dict)
    fibonacci_retracement: Dict[str, Any] = field(default_factory=dict)
    volume_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternConfig:
    """Candlestick and chart pattern parameters"""
    candlestick_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chart_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SignalConfig:
    """Signal generation parameters"""
    min_rrr: float = 1.5
    target_multiplier: float = 1.5
    stop_multiplier: float = 1.0
    min_signal_strength: int = 3
    pattern_strength_weights: Dict[str, int] = field(default_factory=dict)
    indicator_strength_weights: Dict[str, int] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtesting parameters"""
    lookback_days: int = 250
    test_signals: bool = True
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0
    max_open_positions: int = 5
    include_commissions: bool = True
    commission_per_trade: float = 0.05
    monte_carlo_simulations: int = 500
    run_for_all_stocks: bool = False
    walk_forward_optimization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    indicators: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    stock_list: List[str] = field(default_factory=list)
    templates: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Manages application configuration with validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration YAML file
        """
        self.config = AppConfig()
        
        # Load from YAML file if provided
        if config_file:
            self._load_from_file(config_file)
            
        # Validate configuration
        self._validate_config()
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update configuration
            if 'api' in config_data:
                self.config.api = APIConfig(**config_data['api'])
            
            if 'schedule' in config_data:
                self.config.schedule = ScheduleConfig(**config_data['schedule'])
            
            if 'analysis' in config_data:
                self.config.analysis = AnalysisConfig(**config_data['analysis'])
            
            if 'indicators' in config_data:
                self.config.indicators = TechnicalIndicatorConfig(**config_data['indicators'])
            
            if 'patterns' in config_data:
                self.config.patterns = PatternConfig(**config_data['patterns'])
            
            if 'signals' in config_data:
                self.config.signals = SignalConfig(**config_data['signals'])
            
            if 'backtest' in config_data:
                self.config.backtest = BacktestConfig(**config_data['backtest'])
            
            if 'stock_list' in config_data:
                self.config.stock_list = config_data['stock_list']
            
            if 'templates' in config_data:
                self.config.templates = config_data['templates']
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        # Check for required API credentials
        if not self.config.api.upstox_api_key:
            logger.warning("Upstox API key is missing - set UPSTOX_API_KEY environment variable")
        
        if not self.config.api.upstox_api_secret:
            logger.warning("Upstox API secret is missing - set UPSTOX_API_SECRET environment variable")
        
        # Check if Telegram is enabled but credentials are missing
        if self.config.api.enable_telegram_alerts and not (
            self.config.api.telegram_bot_token and self.config.api.telegram_chat_id
        ):
            logger.warning("Telegram alerts enabled but credentials are missing")
        
        # Validate numerical parameters
        if self.config.signals.min_rrr <= 0:
            logger.warning(f"Invalid minimum RRR: {self.config.signals.min_rrr}, must be > 0")
            self.config.signals.min_rrr = 1.5
        
        if self.config.backtest.initial_capital <= 0:
            logger.warning(f"Invalid initial capital: {self.config.backtest.initial_capital}, must be > 0")
            self.config.backtest.initial_capital = 100000.0
    
    def get_config(self) -> AppConfig:
        """Get the application configuration"""
        return self.config


# Default configuration manager instance
config_manager = ConfigManager()