"""
Utility functions for the heart disease ML pipeline.

This module provides helper functions for file operations, logging setup,
and other common utilities used throughout the project.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union
import sys


def setup_logging(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Path object of the validated file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file is not readable
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if not os.access(path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")
    
    return path


def validate_directory_path(dir_path: Union[str, Path], create_if_missing: bool = False) -> Path:
    """
    Validate that a directory path exists and is accessible.
    
    Args:
        dir_path: Path to the directory to validate
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Path object of the validated directory
        
    Raises:
        FileNotFoundError: If the directory doesn't exist and create_if_missing is False
        PermissionError: If the directory is not accessible
    """
    path = Path(dir_path)
    
    if not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")
    
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError(f"Directory is not accessible: {dir_path}")
    
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # Start from current file and go up until we find a marker file
    current_path = Path(__file__).parent
    
    # Look for common project root markers
    markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt', '.kiro']
    
    while current_path != current_path.parent:
        for marker in markers:
            if (current_path / marker).exists():
                return current_path
        current_path = current_path.parent
    
    # If no markers found, return the parent of src directory
    return Path(__file__).parent.parent


def ensure_directory_exists(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Path object of the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.85 for 85%)
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with optional suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length of the result
        suffix: Suffix to add if truncation occurs
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix