"""
Logging utilities for QuantM10

Configures logging for different components with consistent formatting.
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None
) -> logging.Logger:
    """
    Configure logging for the application
    
    Args:
        level: Default logging level
        log_file: Log file path (defaults to timestamped file in logs directory)
        module_levels: Dictionary mapping module names to specific log levels
        
    Returns:
        Root logger instance
    """
    # Generate default log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOGS_DIR / f"quantm10_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific levels for modules if provided
    if module_levels:
        for module, module_level in module_levels.items():
            logging.getLogger(module).setLevel(module_level)
    
    # Log startup message
    root_logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    root_logger.info(f"Log file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)