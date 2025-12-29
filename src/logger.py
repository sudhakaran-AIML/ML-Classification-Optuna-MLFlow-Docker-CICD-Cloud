"""
Centralized logging utility for the ML application.
Provides consistent logging across all modules with rotation and multiple handlers.
"""

import logging
import logging.config
import os
from pathlib import Path
import yaml


def setup_logger(name: str, config_path: str = "config/logging_config.yaml") -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Name of the logger (typically __name__ from calling module)
        config_path: Path to the logging configuration YAML file
    
    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Load logging configuration
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration if config file not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.warning(f"Logging config file not found at {config_path}. Using basic configuration.")
    
    logger = logging.getLogger(name)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Name of the logger
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    Usage: class MyClass(LoggerMixin): ...
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(name)


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Exception occurred during testing")
    
    print("Logging test complete. Check logs/ directory for output files.")
