"""
Main trading bot module for QuantM10

Combines all components to generate trading signals and alerts.
"""
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

from quantm10.config import APIConfig, AppConfig, ConfigManager
from quantm10.exceptions import (
    QuantM10Error, AuthenticationError, DataFetchError, 
    EmptyDataError, SignalGenerationError, NotificationError
)
from quantm10.data.upstox_provider import UpstoxProvider
from quantm10.analysis.indicators import TechnicalIndicators
from quantm10.analysis.patterns import CandlestickPatterns, ChartPatterns
from quantm10.signal.generator import SignalGenerator
from quantm10.backtest.engine import BacktestEngine
from quantm10.notification.telegram import TelegramNotifier
from quantm10.utils.logging import get_logger
from quantm10.utils.database import save_signal, save_backtest_result

logger = get_logger(__name__)


class TradingBot:
    """Main trading bot for signal generation and notification"""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], AppConfig]] = None):
        """
        Initialize trading bot
        
        Args:
            config: Configuration (dict or AppConfig object)
        """
        # Load configuration
        if isinstance(config, dict):
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config
            
            # Update config with provided dict
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        elif isinstance(config, AppConfig):
            self.config = config
        else:
            # Load default configuration
            self.config_manager = ConfigManager()
            self.config = self.config_manager.config
        
        # Initialize components
        self.data_provider = UpstoxProvider(self.config.api)
        self.notifier = None
        
        # Initialize notification if enabled
        if (self.config.api.enable_telegram_alerts and 
            self.config.api.telegram_bot_token and 
            self.config.api.telegram_chat_id):
            self.notifier = TelegramNotifier(
                self.config.api.telegram_bot_token,
                self.config.api.telegram_chat_id
            )
    
    async def initialize(self) -> None:
        """
        Initialize components that require async initialization
        
        Raises:
            QuantM10Error: If initialization fails
        """
        try:
            # Initialize notifier if available
            if self.notifier:
                await self.notifier.initialize()
            
            logger.info("Trading bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {str(e)}")
            raise QuantM10Error(f"Initialization failed: {str(e)}")
    
    async def get_historical_data(self, instrument_key: str, 
                                 interval: str = "1D", 
                                 days: int = 250) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            instrument_key: Instrument identifier
            interval: Timeframe for candlesticks
            days: Number of days of data
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataFetchError: If data fetch fails
        """
        logger.info(f"Fetching {days} days of {interval} data for {instrument_key}")
        
        try:
            # Get data from provider
            df = await self.data_provider.get_historical_data(
                instrument_key=instrument_key,
                interval=interval,
                days=days
            )
            
            logger.info(f"Retrieved {len(df)} data points for {instrument_key}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            raise DataFetchError(instrument_key, str(e))
    
    async def get_stock_info(self, instrument_key: str) -> Dict[str, Any]:
        """
        Get stock information
        
        Args:
            instrument_key: Instrument identifier
            
        Returns:
            Dictionary with stock information
            
        Raises:
            DataFetchError: If data fetch fails
        """
        try:
            # Get details from provider
            return self.data_provider.get_instrument_details(instrument_key)
        except Exception as e:
            logger.error(f"Failed to fetch stock info: {str(e)}")
            raise DataFetchError(instrument_key, str(e))
    
    async def generate_signal(self, instrument_key: str) -> Dict[str, Any]:
        """
        Generate trading signal for an instrument
        
        Args:
            instrument_key: Instrument identifier
            
        Returns:
            Dictionary with signal information
            
        Raises:
            SignalGenerationError: If signal generation fails
        """
        try:
            logger.info(f"Generating signal for {instrument_key}")
            
            # Get historical data
            df = await self.get_historical_data(
                instrument_key=instrument_key,
                interval=self.config.analysis.intervals.get('short_term', '1D'),
                days=self.config.analysis.short_term_lookback
            )
            
            # Get stock info
            stock_info = await self.get_stock_info(instrument_key)
            
            # Calculate technical indicators
            indicators = TechnicalIndicators(df, self.config)
            indicators.calculate_all()
            indicator_signals = indicators.get_overall_signal()
            
            # Detect candlestick patterns
            candlestick_patterns = CandlestickPatterns(df, self.config)
            candlestick_patterns.detect_all_patterns()
            candlestick_signals = candlestick_patterns.get_overall_signal()
            
            # Detect chart patterns
            chart_patterns = ChartPatterns(df, self.config)
            chart_patterns.detect_all_patterns()
            chart_signals = chart_patterns.get_overall_signal()
            
            # Generate combined signal
            signal_generator = SignalGenerator(df, self.config)
            signal = signal_generator.generate_signals(
                indicator_signals,
                candlestick_signals,
                chart_signals
            )
            
            # Add trade checklist
            signal['checklist'] = signal_generator.create_trade_checklist(signal)
            
            # Add stock info
            signal['stock_info'] = stock_info
            
            # Save signal to database
            if signal.get('signal_type') != 'NEUTRAL':
                save_signal(
                    instrument_key=instrument_key,
                    symbol=stock_info.get('symbol', ''),
                    name=stock_info.get('name', ''),
                    exchange=stock_info.get('exchange', ''),
                    signal_type=signal.get('signal_type', 'NEUTRAL'),
                    entry_price=signal.get('entry_price', 0),
                    stop_loss=signal.get('stop_loss'),
                    target_price=signal.get('target_price'),
                    strength=signal.get('signal_strength', 0),
                    indicators=indicator_signals,
                    patterns={
                        'candlestick': candlestick_signals,
                        'chart': chart_signals
                    }
                )
            
            logger.info(f"Generated {signal.get('signal_text', 'NEUTRAL')} signal for {instrument_key}")
            return signal
        
        except Exception as e:
            logger.error(f"Signal generation failed for {instrument_key}: {str(e)}")
            raise SignalGenerationError(f"Failed for {instrument_key}: {str(e)}")
    
    async def run_backtest(self, instrument_key: str, days: int = 250) -> Dict[str, Any]:
        """
        Run backtest for an instrument
        
        Args:
            instrument_key: Instrument identifier
            days: Number of days for backtest
            
        Returns:
            Dictionary with backtest results
            
        Raises:
            BacktestError: If backtest fails
        """
        try:
            logger.info(f"Running backtest for {instrument_key} with {days} days of data")
            
            # Get historical data
            df = await self.get_historical_data(
                instrument_key=instrument_key,
                interval=self.config.analysis.intervals.get('short_term', '1D'),
                days=days
            )
            
            # Get stock info
            stock_info = await self.get_stock_info(instrument_key)
            
            # Define strategy function
            def strategy(data: pd.DataFrame, **kwargs) -> int:
                """
                Trading strategy based on indicators and patterns
                
                Args:
                    data: Price data
                    
                Returns:
                    Signal (1 for buy, -1 for sell, 0 for neutral)
                """
                # Use only data up to current index
                current_data = data.copy()
                
                # Calculate indicators
                indicators = TechnicalIndicators(current_data, self.config)
                indicators.calculate_all()
                indicator_signals = indicators.get_overall_signal()
                
                # Detect patterns
                candlestick_patterns = CandlestickPatterns(current_data, self.config)
                candlestick_patterns.detect_all_patterns()
                pattern_signals = candlestick_patterns.get_overall_signal()
                
                # Determine signal
                indicator_strength = indicator_signals.get('strength', 0)
                pattern_strength = pattern_signals.get('strength', 0)
                
                # Weighted combination
                combined_strength = (indicator_strength * 0.7) + (pattern_strength * 0.3)
                
                if combined_strength >= 2:
                    return 1  # Buy
                elif combined_strength <= -2:
                    return -1  # Sell
                else:
                    return 0  # Neutral
            
            # Initialize backtest engine
            backtest = BacktestEngine(df, self.config, self.config.backtest.initial_capital)
            
            # Run backtest
            results = backtest.run_backtest(strategy)
            
            # Run Monte Carlo simulation
            monte_carlo_results = backtest.monte_carlo_simulation(
                self.config.backtest.monte_carlo_simulations
            )
            
            # Combine results
            results['monte_carlo'] = monte_carlo_results
            results['stock_name'] = stock_info.get('name', '')
            results['stock_symbol'] = stock_info.get('symbol', '')
            results['instrument_key'] = instrument_key
            results['start_date'] = df.index[0].strftime('%Y-%m-%d')
            results['end_date'] = df.index[-1].strftime('%Y-%m-%d')
            
            # Save to database
            save_backtest_result(
                instrument_key=instrument_key,
                symbol=stock_info.get('symbol', ''),
                name=stock_info.get('name', ''),
                exchange=stock_info.get('exchange', ''),
                start_date=df.index[0],
                end_date=df.index[-1],
                initial_capital=self.config.backtest.initial_capital,
                strategy_name='default',
                params={},
                metrics={
                    'total_return': results.get('total_return', '0%'),
                    'sharpe_ratio': results.get('sharpe_ratio', '0'),
                    'max_drawdown': results.get('max_drawdown', '0%'),
                    'win_rate': results.get('win_rate', '0%'),
                    'total_trades': results.get('total_trades', 0)
                },
                trades=results.get('trades', [])
            )
            
            logger.info(f"Backtest completed for {instrument_key}")
            return results
        
        except Exception as e:
            logger.error(f"Backtest failed for {instrument_key}: {str(e)}")
            return {'error': str(e)}
    
    async def send_signal_alert(self, signal: Dict[str, Any]) -> bool:
        """
        Send signal alert via notification service
        
        Args:
            signal: Signal data
            
        Returns:
            True if alert was sent successfully
        """
        if not self.notifier or not self.config.api.enable_telegram_alerts:
            logger.info("Notifications disabled, skipping alert")
            return False
        
        # Only send alerts for actionable signals
        if signal.get('signal_type', 'NEUTRAL') == 'NEUTRAL' or not signal.get('actionable', False):
            logger.info("Signal not actionable, skipping alert")
            return False
        
        try:
            # Get template from config
            template = self.config.templates.get('signal_message', None)
            
            # Send alert
            return await self.notifier.send_signal_alert(
                signal_data=signal,
                stock_info=signal.get('stock_info', {}),
                template=template
            )
        
        except Exception as e:
            logger.error(f"Failed to send signal alert: {str(e)}")
            return False
    
    async def send_backtest_report(self, backtest_results: Dict[str, Any]) -> bool:
        """
        Send backtest report via notification service
        
        Args:
            backtest_results: Backtest results
            
        Returns:
            True if report was sent successfully
        """
        if not self.notifier or not self.config.api.enable_telegram_alerts:
            logger.info("Notifications disabled, skipping backtest report")
            return False
        
        # Skip if there was an error
        if 'error' in backtest_results:
            logger.info(f"Backtest had error, skipping report: {backtest_results['error']}")
            return False
        
        try:
            # Get template from config
            template = self.config.templates.get('backtest_report', None)
            
            # Extract stock info
            stock_info = {
                'name': backtest_results.get('stock_name', ''),
                'symbol': backtest_results.get('stock_symbol', '')
            }
            
            # Send report
            return await self.notifier.send_backtest_report(
                backtest_results=backtest_results,
                stock_info=stock_info,
                template=template
            )
        
        except Exception as e:
            logger.error(f"Failed to send backtest report: {str(e)}")
            return False
    
    async def send_daily_report(self, results: List[Dict[str, Any]]) -> bool:
        """
        Send daily summary report of all signals
        
        Args:
            results: List of signal results
            
        Returns:
            True if report was sent successfully
        """
        if not self.notifier or not self.config.api.enable_telegram_alerts or not self.config.api.enable_daily_report:
            logger.info("Daily reports disabled, skipping")
            return False
        
        try:
            return await self.notifier.send_daily_report(results)
        
        except Exception as e:
            logger.error(f"Failed to send daily report: {str(e)}")
            return False
    
    async def analyze_all_stocks(self) -> List[Dict[str, Any]]:
        """
        Analyze all stocks in the configured list
        
        Returns:
            List of signal results
        """
        logger.info(f"Analyzing {len(self.config.stock_list)} stocks")
        
        results = []
        for instrument_key in self.config.stock_list:
            try:
                signal = await self.generate_signal(instrument_key)
                
                # Add to results if signal is actionable
                if signal.get('actionable', False):
                    results.append(signal)
                    
                    # Send alert for individual signal
                    await self.send_signal_alert(signal)
                    
                    # Run backtest if configured
                    if self.config.backtest.run_for_all_stocks:
                        backtest_results = await self.run_backtest(instrument_key)
                        
                        # Send backtest report if configured
                        await self.send_backtest_report(backtest_results)
            
            except Exception as e:
                logger.error(f"Failed to analyze {instrument_key}: {str(e)}")
        
        # Send daily report if enabled
        if results and self.config.api.enable_daily_report:
            await self.send_daily_report(results)
        
        return results
    
    async def run(self) -> None:
        """
        Run the trading bot for a complete cycle
        """
        logger.info("Starting trading bot run")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Analyze all stocks
            results = await self.analyze_all_stocks()
            
            logger.info(f"Trading bot run completed with {len(results)} actionable signals")
        
        except Exception as e:
            logger.error(f"Trading bot run failed: {str(e)}")
            raise