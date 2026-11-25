"""
src/memory/profile_store.py
----------------------------
Long-Term Memory Layer — CRUD for the user_profiles table.

This module provides:
  - load_user_profile(user_id)  → dict  (loaded at start of every conversation)
  - upsert_user_profile(user_id, fields)  → None  (updated after itinerary approval)

The supervisor agent injects the loaded profile into TravelState so all
specialist agents have context about the user (home airport, diet, budget tier, etc.)
without re-querying the DB on every node.
"""
from __future__ import annotations

from typing import Any

from src.core.database import get_pool
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.memory.profile")

# SQL to run once at startup (safe with IF NOT EXISTS)
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id             VARCHAR(64) PRIMARY KEY,
    name           VARCHAR(128),
    email          VARCHAR(128) UNIQUE,
    password_hash  VARCHAR(255),
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id                 VARCHAR(64) PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    home_airport            VARCHAR(10) DEFAULT 'HYD',
    preferred_travel_style  VARCHAR(64),
    dietary_preferences     TEXT[],
    frequent_airlines       TEXT[],
    hotel_preferences       TEXT[],
    custom_notes            TEXT,
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS threads (
    id          VARCHAR(64) PRIMARY KEY,
    user_id     VARCHAR(64) REFERENCES users(id) ON DELETE CASCADE,
    title       VARCHAR(255),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
"""


def run_schema_migrations() -> None:
    """Create user-facing tables if they don't exist. Safe to call on every startup."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(_SCHEMA_SQL)
    logger.info("[PROFILE] Schema migrations complete (users, user_profiles, threads)")


def ensure_user_exists(user_id: str, name: str = "", email: str = "") -> None:
    """Insert the user row if it doesn't already exist (upsert-safe)."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO users (id, name, email)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (user_id, name or user_id, email),
        )


def load_user_profile(user_id: str) -> dict[str, Any]:
    """
    Load the user's account info and long-term profile in a single fast query.
    Returns an empty dict if the user does not exist.
    """
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT u.id AS user_id, u.name, u.email,
                   p.home_airport, p.preferred_travel_style,
                   p.dietary_preferences, p.frequent_airlines,
                   p.hotel_preferences, p.custom_notes, p.updated_at
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = %s
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        logger.debug("[PROFILE] No user/profile found for user_id=%s", user_id)
        return {}

    profile = dict(row)
    logger.info(
        "[PROFILE] Loaded account+profile for user_id=%s (name=%s, airport=%s)",
        user_id,
        profile.get("name"),
        profile.get("home_airport"),
    )
    return profile


def upsert_user_profile(user_id: str, fields: dict[str, Any]) -> None:
    """
    Persist learned user preferences after a trip is planned.
    Only the fields explicitly provided in `fields` are updated.
    """
    if not fields:
        return

    ensure_user_exists(user_id)

    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, home_airport, preferred_travel_style,
                dietary_preferences, frequent_airlines,
                hotel_preferences, custom_notes, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                home_airport           = COALESCE(EXCLUDED.home_airport, user_profiles.home_airport),
                preferred_travel_style = COALESCE(EXCLUDED.preferred_travel_style, user_profiles.preferred_travel_style),
                dietary_preferences    = COALESCE(EXCLUDED.dietary_preferences, user_profiles.dietary_preferences),
                frequent_airlines      = COALESCE(EXCLUDED.frequent_airlines, user_profiles.frequent_airlines),
                hotel_preferences      = COALESCE(EXCLUDED.hotel_preferences, user_profiles.hotel_preferences),
                custom_notes           = COALESCE(EXCLUDED.custom_notes, user_profiles.custom_notes),
                updated_at             = NOW()
            """,
            (
                user_id,
                fields.get("home_airport"),
                fields.get("preferred_travel_style"),
                fields.get("dietary_preferences"),
                fields.get("frequent_airlines"),
                fields.get("hotel_preferences"),
                fields.get("custom_notes"),
            ),
        )
    logger.info("[PROFILE] Upserted profile for user_id=%s fields=%s", user_id, list(fields.keys()))
