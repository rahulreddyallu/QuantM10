"""
Core interfaces for QuantM10

Defines the contract between different components of the system
using Python's Protocol classes for structural typing.
"""
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from datetime import datetime
import pandas as pd


@runtime_checkable
class DataProvider(Protocol):
    """Interface for market data providers"""
    
    async def get_historical_data(self, 
                                instrument_key: str, 
                                interval: str, 
                                from_date: Optional[datetime] = None,
                                to_date: Optional[datetime] = None,
                                days: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            instrument_key: Identifier for the instrument
            interval: Timeframe for candlesticks (e.g. '1D', '1H')
            from_date: Start date for data fetch
            to_date: End date for data fetch
            days: Number of days of data to fetch (alternative to date range)
            
        Returns:
            DataFrame with OHLCV data
        """
        ...
    
    def get_instrument_details(self, instrument_key: str) -> Dict[str, Any]:
        """
        Get details about an instrument
        
        Args:
            instrument_key: Identifier for the instrument
            
        Returns:
            Dictionary with instrument details
        """
        ...


@runtime_checkable
class TechnicalAnalyzer(Protocol):
    """Interface for technical analysis providers"""
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators
        
        Args:
            data: OHLCV price data
            
        Returns:
            Dictionary with indicator values and signals
        """
        ...
    
    def get_signals(self) -> Dict[str, Any]:
        """
        Get trading signals from calculated indicators
        
        Returns:
            Dictionary with signal information
        """
        ...


@runtime_checkable
class PatternDetector(Protocol):
    """Interface for pattern detection"""
    
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect patterns in price data
        
        Args:
            data: OHLCV price data
            
        Returns:
            Dictionary with detected patterns
        """
        ...
    
    def get_signals(self) -> Dict[str, Any]:
        """
        Get trading signals from detected patterns
        
        Returns:
            Dictionary with signal information
        """
        ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Interface for signal generation"""
    
    def generate_signals(self, 
                        indicators: Dict[str, Any], 
                        patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals
        
        Args:
            indicators: Dictionary with indicator values and signals
            patterns: Dictionary with detected patterns
            
        Returns:
            Dictionary with final trading signals
        """
        ...


@runtime_checkable
class BacktestEngine(Protocol):
    """Interface for backtesting"""
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy: Any, 
                    **params) -> Dict[str, Any]:
        """
        Run backtest for a strategy
        
        Args:
            data: OHLCV price data
            strategy: Trading strategy function or object
            params: Additional parameters for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        ...
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        ...


@runtime_checkable
class NotificationService(Protocol):
    """Interface for notification services"""
    
    async def send_message(self, message: str) -> bool:
        """
        Send notification message
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        ...
    
    async def send_signal_alert(self, signal_data: Dict[str, Any]) -> bool:
        """
        Send trading signal alert
        
        Args:
            signal_data: Signal data to include in alert
            
        Returns:
            True if alert was sent successfully
        """
        ...