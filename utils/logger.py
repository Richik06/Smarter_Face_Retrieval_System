"""
Centralised logging setup.
Call setup_logger(__name__) in every module to get a properly configured logger.
"""

import logging
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def setup_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name* that is configured exactly once.
    Uses lru_cache so repeated calls for the same name return the same instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger