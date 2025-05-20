"""
Upstox data provider for QuantM10

Handles authentication, historical data retrieval, and instrument details
from the Upstox API.
"""
import logging
import asyncio
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import upstox_client
from upstox_client.api_client import ApiClient
from upstox_client.api.login_api import LoginApi
from upstox_client.api.market_quote_api import MarketQuoteApi
from upstox_client.api.history_api import HistoryApi
from upstox_client.models.ohlc import Ohlc as OHLCInterval

from quantm10.config import APIConfig
from quantm10.exceptions import AuthenticationError, ConnectionError, DataFetchError, EmptyDataError
from quantm10.utils.cache import cached_dataframe
from quantm10.utils.logging import get_logger

logger = get_logger(__name__)


class UpstoxProvider:
    """Data provider for Upstox API"""
    
    def __init__(self, config: APIConfig):
        """
        Initialize Upstox API client
        
        Args:
            config: API configuration
        """
        self.config = config
        self.api_key = config.upstox_api_key
        self.api_secret = config.upstox_api_secret
        self.redirect_uri = config.upstox_redirect_uri
        
        # Track authentication state
        self.authenticated = False
        self.client = None
        self.last_auth_time = 0
        self.token_expiry = 0
        
        # Initialize API clients
        self.market_client = None
        self.history_client = None
        
        # Cache for instrument details
        self.instrument_cache = {}
    
    def authenticate(self) -> None:
        """
        Authenticate with Upstox API
        
        Raises:
            AuthenticationError: If authentication fails
        """
        logger.info("Authenticating with Upstox API")
        
        try:
            # Create API client with configured API key
            configuration = upstox_client.Configuration()
            configuration.api_key['api_key'] = self.api_key
            self.client = ApiClient(configuration)
            
            # Create API instances
            login_api = LoginApi(self.client)
            
            # Authenticate (in a real implementation, we would need to handle the OAuth flow)
            # This is a simplified version assuming we already have a valid access token
            # In production, you would need to implement the full OAuth flow
            
            # Set token in configuration
            self.client.configuration.access_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
            
            # Update authentication state
            self.authenticated = True
            self.last_auth_time = time.time()
            self.token_expiry = self.last_auth_time + 86400  # Assume 24h validity
            
            # Initialize API clients
            self.market_client = MarketQuoteApi(self.client)
            self.history_client = HistoryApi(self.client)
            
            logger.info("Successfully authenticated with Upstox API")
        
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise AuthenticationError("Upstox", str(e))
    
    def _check_authentication(self) -> None:
        """
        Check if we're authenticated and refresh if needed
        
        Raises:
            AuthenticationError: If authentication fails
        """
        current_time = time.time()
        
        # If not authenticated or token is expired, authenticate
        if not self.authenticated or current_time > self.token_expiry - 300:  # 5 min buffer
            self.authenticate()
    
    @cached_dataframe(expires_after=3600)  # Cache for 1 hour
    async def get_historical_data(self, 
                                instrument_key: str, 
                                interval: str = "1D", 
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
            
        Raises:
            DataFetchError: If data fetch fails
            EmptyDataError: If no data is returned
        """
        logger.info(f"Fetching historical data for {instrument_key} with interval {interval}")
        
        try:
            # Check authentication
            self._check_authentication()
            
            # Calculate date range if days is provided
            if days is not None and from_date is None and to_date is None:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days)
            
            # Set default to_date if not provided
            if to_date is None:
                to_date = datetime.now()
            
            # Set default from_date if not provided
            if from_date is None:
                from_date = to_date - timedelta(days=30)  # Default to 30 days
            
            # Convert interval to Upstox format
            interval_map = {
                "1m": OHLCInterval.MINUTE_1,
                "5m": OHLCInterval.MINUTE_5,
                "10m": OHLCInterval.MINUTE_10,
                "30m": OHLCInterval.MINUTE_30,
                "1h": OHLCInterval.HOUR_1,
                "1d": OHLCInterval.DAY_1,
                "1w": OHLCInterval.WEEK_1,
                "1M": OHLCInterval.MONTH_1,
                "1D": OHLCInterval.DAY_1,
                "1W": OHLCInterval.WEEK_1,
                "day": OHLCInterval.DAY_1,
                "week": OHLCInterval.WEEK_1,
                "month": OHLCInterval.MONTH_1
            }
            
            upstox_interval = interval_map.get(interval.lower(), OHLCInterval.DAY_1)
            
            # Format dates for API
            from_date_str = from_date.strftime("%Y-%m-%d")
            to_date_str = to_date.strftime("%Y-%m-%d")
            
            # Make API request
            response = await asyncio.to_thread(
                self.history_client.get_history_data,
                instrument_key=instrument_key,
                interval=upstox_interval,
                from_date=from_date_str,
                to_date=to_date_str
            )
            
            # Check if data was returned
            if not response or not response.data or len(response.data) == 0:
                raise EmptyDataError(instrument_key)
            
            # Convert to DataFrame
            data = []
            for candle in response.data:
                data.append({
                    'timestamp': pd.Timestamp(candle.timestamp).to_pydatetime(),
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Retrieved {len(df)} historical data points for {instrument_key}")
            
            return df
        
        except EmptyDataError:
            # Re-raise empty data errors
            raise
        
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {instrument_key}: {str(e)}")
            raise DataFetchError(instrument_key, str(e))
    
    def get_instrument_details(self, instrument_key: str) -> Dict[str, Any]:
        """
        Get details about an instrument
        
        Args:
            instrument_key: Identifier for the instrument
            
        Returns:
            Dictionary with instrument details
            
        Raises:
            DataFetchError: If data fetch fails
        """
        # Check if we have this instrument in cache
        if instrument_key in self.instrument_cache:
            return self.instrument_cache[instrument_key]
        
        logger.info(f"Fetching instrument details for {instrument_key}")
        
        try:
            # Check authentication
            self._check_authentication()
            
            # Parse instrument key to extract exchange and token
            parts = instrument_key.split('|')
            
            if len(parts) != 2:
                raise ValueError(f"Invalid instrument key format: {instrument_key}")
            
            exchange, token = parts
            
            # Make API request
            response = self.market_client.get_instrument_info(
                instrument_key=instrument_key
            )
            
            # Extract relevant details
            details = {
                'instrument_key': instrument_key,
                'exchange': exchange,
                'token': token,
                'symbol': response.symbol if hasattr(response, 'symbol') else '',
                'name': response.name if hasattr(response, 'name') else '',
                'instrument_type': response.instrument_type if hasattr(response, 'instrument_type') else '',
                'expiry': response.expiry if hasattr(response, 'expiry') else None,
                'lot_size': response.lot_size if hasattr(response, 'lot_size') else 1
            }
            
            # Cache result
            self.instrument_cache[instrument_key] = details
            
            return details
        
        except Exception as e:
            logger.error(f"Failed to fetch instrument details for {instrument_key}: {str(e)}")
            
            # Return basic details that we can extract from the key
            parts = instrument_key.split('|')
            if len(parts) == 2:
                exchange, token = parts
                return {
                    'instrument_key': instrument_key,
                    'exchange': exchange,
                    'token': token,
                    'symbol': token,
                    'name': token
                }
            
            raise DataFetchError(instrument_key, str(e))