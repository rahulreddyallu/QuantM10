"""
Caching utilities for QuantM10

Provides decorators and functions for caching expensive operations.
"""
import functools
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def hash_args(*args: Any, **kwargs: Any) -> str:
    """
    Create a hash of function arguments for cache key
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        MD5 hash of arguments as string
    """
    # Convert arguments to JSON-compatible format
    args_str = json.dumps((args, kwargs), sort_keys=True, default=str)
    return hashlib.md5(args_str.encode('utf-8')).hexdigest()


def cached(expires_after: Optional[int] = 3600) -> Callable:
    """
    Cache function results to disk
    
    Args:
        expires_after: Cache expiry time in seconds (None for no expiry)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key from function name and arguments
            func_name = func.__name__
            arg_hash = hash_args(*args, **kwargs)
            cache_key = f"{func_name}_{arg_hash}"
            cache_path = CACHE_DIR / f"{cache_key}.pkl"
            
            # Check if cache file exists and is not expired
            if cache_path.exists():
                # Check expiry if specified
                if expires_after is not None:
                    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    if datetime.now() - mtime > timedelta(seconds=expires_after):
                        logger.debug(f"Cache expired for {func_name}")
                        # Remove expired cache
                        os.remove(cache_path)
                    else:
                        # Load from cache
                        try:
                            with open(cache_path, 'rb') as f:
                                result = pickle.load(f)
                            logger.debug(f"Loaded from cache: {func_name}")
                            return result
                        except Exception as e:
                            logger.warning(f"Failed to load cache: {str(e)}")
                else:
                    # No expiry, load from cache
                    try:
                        with open(cache_path, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Loaded from cache: {func_name}")
                        return result
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {str(e)}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Saved to cache: {func_name}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator


def cached_dataframe(expires_after: Optional[int] = 3600) -> Callable:
    """
    Specialized cache decorator for pandas DataFrames
    with optimization for reading/writing
    
    Args:
        expires_after: Cache expiry time in seconds (None for no expiry)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            # Generate cache key from function name and arguments
            func_name = func.__name__
            arg_hash = hash_args(*args, **kwargs)
            cache_key = f"{func_name}_{arg_hash}"
            cache_path = CACHE_DIR / f"{cache_key}.parquet"
            
            # Check if cache file exists and is not expired
            if cache_path.exists():
                # Check expiry if specified
                if expires_after is not None:
                    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    if datetime.now() - mtime > timedelta(seconds=expires_after):
                        logger.debug(f"DataFrame cache expired for {func_name}")
                        # Remove expired cache
                        os.remove(cache_path)
                    else:
                        # Load from cache using parquet for efficiency
                        try:
                            df = pd.read_parquet(cache_path)
                            logger.debug(f"Loaded DataFrame from cache: {func_name}")
                            return df
                        except Exception as e:
                            logger.warning(f"Failed to load DataFrame cache: {str(e)}")
                else:
                    # No expiry, load from cache
                    try:
                        df = pd.read_parquet(cache_path)
                        logger.debug(f"Loaded DataFrame from cache: {func_name}")
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to load DataFrame cache: {str(e)}")
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Ensure result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                logger.warning(f"Result is not a DataFrame, not caching: {func_name}")
                return result
            
            # Save to cache using parquet for efficiency
            try:
                result.to_parquet(cache_path, index=True)
                logger.debug(f"Saved DataFrame to cache: {func_name}")
            except Exception as e:
                logger.warning(f"Failed to save DataFrame cache: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator


def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cache files
    
    Args:
        pattern: Optional pattern to match filenames (None to clear all)
        
    Returns:
        Number of files deleted
    """
    count = 0
    for cache_file in CACHE_DIR.glob("*.pkl" if pattern is None else f"*{pattern}*.pkl"):
        try:
            os.remove(cache_file)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
    
    for cache_file in CACHE_DIR.glob("*.parquet" if pattern is None else f"*{pattern}*.parquet"):
        try:
            os.remove(cache_file)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
    
    logger.info(f"Cleared {count} cache files")
    return count