"""
Exception hierarchy for QuantM10

Provides specific exceptions for different error scenarios
to improve error handling and reporting.
"""
from typing import Optional, Any


class QuantM10Error(Exception):
    """Base exception for all QuantM10 errors"""
    pass


class ConfigurationError(QuantM10Error):
    """Error in configuration values or loading configuration"""
    pass


class APIError(QuantM10Error):
    """Base class for API-related errors"""
    def __init__(self, message: str, service: str, status_code: Optional[int] = None):
        self.service = service
        self.status_code = status_code
        super().__init__(f"{service} API Error: {message} (Status: {status_code})")


class ConnectionError(APIError):
    """Error establishing connection to external service"""
    def __init__(self, service: str, reason: str):
        super().__init__(f"Connection failed: {reason}", service)


class AuthenticationError(APIError):
    """Error authenticating with external service"""
    def __init__(self, service: str, details: str = ""):
        super().__init__(f"Authentication failed: {details}", service, 401)


class DataError(QuantM10Error):
    """Base class for data-related errors"""
    pass


class DataFetchError(DataError):
    """Error fetching data from external source"""
    def __init__(self, source: str, symbol: Optional[str] = None, details: str = ""):
        message = f"Failed to fetch data from {source}"
        if symbol:
            message += f" for {symbol}"
        if details:
            message += f": {details}"
        super().__init__(message)


class EmptyDataError(DataError):
    """No data was returned from source"""
    def __init__(self, source: str, symbol: Optional[str] = None):
        message = f"No data returned from {source}"
        if symbol:
            message += f" for {symbol}"
        super().__init__(message)


class InvalidDataError(DataError):
    """Data structure or content is invalid"""
    def __init__(self, details: str):
        super().__init__(f"Invalid data: {details}")


class AnalysisError(QuantM10Error):
    """Base class for analysis-related errors"""
    pass


class IndicatorError(AnalysisError):
    """Error calculating technical indicator"""
    def __init__(self, indicator: str, details: str = ""):
        message = f"Error calculating {indicator}"
        if details:
            message += f": {details}"
        super().__init__(message)


class PatternDetectionError(AnalysisError):
    """Error detecting chart or candlestick pattern"""
    def __init__(self, pattern_type: str, details: str = ""):
        message = f"Error detecting {pattern_type} pattern"
        if details:
            message += f": {details}"
        super().__init__(message)


class SignalGenerationError(AnalysisError):
    """Error generating trading signal"""
    def __init__(self, details: str = ""):
        message = f"Error generating trading signal"
        if details:
            message += f": {details}"
        super().__init__(message)


class BacktestError(QuantM10Error):
    """Error during backtesting"""
    def __init__(self, details: str = ""):
        message = f"Backtest error"
        if details:
            message += f": {details}"
        super().__init__(message)


class NotificationError(QuantM10Error):
    """Error sending notification"""
    def __init__(self, channel: str, details: str = ""):
        message = f"Failed to send notification via {channel}"
        if details:
            message += f": {details}"
        super().__init__(message)