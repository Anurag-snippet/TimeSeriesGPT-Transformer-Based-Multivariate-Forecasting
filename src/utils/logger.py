"""Structured logging utility for experiments and pipelines."""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "TimeSeriesIntelligence", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized console/file logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
