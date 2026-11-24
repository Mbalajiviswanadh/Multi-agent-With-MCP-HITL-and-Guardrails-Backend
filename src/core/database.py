"""
src/core/database.py
--------------------
Production-grade PostgreSQL connection management using psycopg_pool.
Replaces the single global `psycopg.connect()` with a ConnectionPool
that handles up to `db_pool_max` concurrent requests without exhausting
the database.
"""
from __future__ import annotations

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.database")

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """
    Return the module-level connection pool, creating it on first call.
    Cheap to call repeatedly — creation happens only once.
    """
    global _pool
    if _pool is None:
        logger.info(
            "[DB] Creating connection pool (min=%d, max=%d)",
            settings.db_pool_min,
            settings.db_pool_max,
        )
        _pool = ConnectionPool(
            conninfo=settings.db_url_with_ssl,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
        logger.info("[DB] Connection pool ready")
    return _pool


def close_pool() -> None:
    """Call this during application shutdown to cleanly release connections."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("[DB] Connection pool closed")
