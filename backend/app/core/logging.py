"""
core/logging.py
===============
Structured logging configuration for the copilot backend.

Uses Python's standard logging with Rich formatting for beautiful console output
in development, and plain JSON-friendly format for production.
"""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configure application-wide logging."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Remove existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "chromadb"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Call this at the top of each module:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
