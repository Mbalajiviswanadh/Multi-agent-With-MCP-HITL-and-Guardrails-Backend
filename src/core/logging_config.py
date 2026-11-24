"""
src/core/logging_config.py
--------------------------
Centralised logger factory.  Every module in the project calls
`get_logger(__name__)` and gets a consistently formatted logger that
writes to both the console and the persistent log file.
"""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "TripMateAI") -> logging.Logger:
    """
    Return a logger for *name*.  Handlers are added only once; subsequent
    calls with the same name reuse the existing handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    console_fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.getenv("LOG_FILE", "agent_runs.log")
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", log_file, exc)

    return logger
