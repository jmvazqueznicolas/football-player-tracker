"""Centralized logging configuration using loguru."""

from __future__ import annotations

import os
import sys

from loguru import logger


def setup_logging(level: str | None = None) -> None:
    """Configure loguru with sensible defaults.

    Reads LOG_LEVEL from environment or uses provided level. INFO by default.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )


def get_logger():
    """Return the global loguru logger."""
    return logger
