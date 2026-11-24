"""
src/memory/checkpointer.py
--------------------------
PostgresSaver wired to the production connection pool.
Provides the graph-level checkpointing that enables thread resumption
(Human-in-the-Loop) and short-term message persistence.
"""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.memory.checkpointer")

_checkpointer: PostgresSaver | None = None
_conn: psycopg.Connection | None = None


def get_checkpointer() -> PostgresSaver:
    """
    Return the module-level PostgresSaver, creating it on first call.
    Uses a dedicated synchronous connection required by LangGraph's
    PostgresSaver (psycopg_pool async is not yet supported).
    """
    global _checkpointer, _conn
    if _checkpointer is None:
        logger.info("[CHECKPOINTER] Initialising PostgresSaver...")
        _conn = psycopg.connect(
            settings.db_url_with_ssl,
            autocommit=True,
            row_factory=dict_row,
        )
        _checkpointer = PostgresSaver(_conn)
        _checkpointer.setup()
        logger.info("[CHECKPOINTER] PostgresSaver ready")
    return _checkpointer


def close_checkpointer() -> None:
    global _conn, _checkpointer
    if _conn is not None:
        _conn.close()
        _conn = None
        _checkpointer = None
        logger.info("[CHECKPOINTER] Connection closed")
