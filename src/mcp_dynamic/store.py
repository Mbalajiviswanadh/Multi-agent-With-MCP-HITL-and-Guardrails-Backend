"""
src/mcp_dynamic/store.py
------------------------
PostgreSQL persistence for per-user connected MCP servers (`user_mcp_connections`).
Stores encrypted OAuth tokens / API keys and cached tool schemas.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.core.database import get_pool
from src.core.encryption import decrypt_json, encrypt_json
from src.core.logging_config import get_logger
from src.memory.profile_store import ensure_user_exists

logger = get_logger("TripMateAI.mcp_dynamic.store")

_MCP_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_mcp_connections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(64) NOT NULL,
    server_slug     VARCHAR(64) NOT NULL,
    display_name    VARCHAR(128) NOT NULL,
    transport       VARCHAR(32) NOT NULL DEFAULT 'streamable_http',
    url             TEXT DEFAULT '',
    auth_type       VARCHAR(32) NOT NULL DEFAULT 'none',
    encrypted_auth  TEXT DEFAULT '',
    cached_tools    JSONB DEFAULT '[]'::jsonb,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    status          VARCHAR(32) NOT NULL DEFAULT 'connected',
    last_error      TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_mcp_user_id
    ON user_mcp_connections (user_id);
"""


def run_mcp_schema_migrations() -> None:
    """Create user_mcp_connections table if it doesn't exist."""
    pool = get_pool()
    try:
        with pool.connection() as conn:
            conn.execute(_MCP_SCHEMA_SQL)
        logger.info("[MCP_STORE] Schema ready (user_mcp_connections)")
    except Exception as exc:
        logger.warning("[MCP_STORE] Schema check notice: %s", exc)


def upsert_mcp_connection(
    *,
    user_id: str,
    server_slug: str,
    display_name: str,
    transport: str,
    url: str,
    auth_type: str,
    auth_payload: dict[str, Any],
    cached_tools: list[dict[str, Any]],
    status: str = "connected",
    last_error: str = "",
    conn_id: str | None = None,
) -> dict[str, Any]:
    """Insert or update a user's MCP connection with encrypted credentials."""
    ensure_user_exists(user_id)
    enc = encrypt_json(auth_payload)
    tools_json = json.dumps(cached_tools or [])
    pool = get_pool()

    with pool.connection() as conn:
        if conn_id:
            row = conn.execute(
                """
                UPDATE user_mcp_connections
                SET display_name = %s,
                    transport = %s,
                    url = %s,
                    auth_type = %s,
                    encrypted_auth = %s,
                    cached_tools = %s::jsonb,
                    status = %s,
                    last_error = %s,
                    enabled = TRUE,
                    updated_at = NOW()
                WHERE id = %s AND user_id = %s
                RETURNING id, user_id, server_slug, display_name, transport, url,
                          auth_type, cached_tools, enabled, status, last_error, created_at, updated_at
                """,
                (
                    display_name, transport, url, auth_type, enc,
                    tools_json, status, last_error, conn_id, user_id,
                ),
            ).fetchone()
            if row:
                return _public_row(dict(row))

        # For built-in OAuth integrations (google_drive / google_calendar), replace existing slug for that user
        if server_slug in ("google_drive", "google_calendar"):
            existing = conn.execute(
                "SELECT id FROM user_mcp_connections WHERE user_id = %s AND server_slug = %s LIMIT 1",
                (user_id, server_slug),
            ).fetchone()
            if existing:
                row = conn.execute(
                    """
                    UPDATE user_mcp_connections
                    SET display_name = %s,
                        transport = %s,
                        url = %s,
                        auth_type = %s,
                        encrypted_auth = %s,
                        cached_tools = %s::jsonb,
                        status = %s,
                        last_error = %s,
                        enabled = TRUE,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, user_id, server_slug, display_name, transport, url,
                              auth_type, cached_tools, enabled, status, last_error, created_at, updated_at
                    """,
                    (
                        display_name, transport, url, auth_type, enc,
                        tools_json, status, last_error, existing["id"],
                    ),
                ).fetchone()
                return _public_row(dict(row))

        new_id = str(uuid.uuid4())
        row = conn.execute(
            """
            INSERT INTO user_mcp_connections
                (id, user_id, server_slug, display_name, transport, url,
                 auth_type, encrypted_auth, cached_tools, enabled, status, last_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, TRUE, %s, %s)
            RETURNING id, user_id, server_slug, display_name, transport, url,
                      auth_type, cached_tools, enabled, status, last_error, created_at, updated_at
            """,
            (
                new_id, user_id, server_slug, display_name, transport, url,
                auth_type, enc, tools_json, status, last_error,
            ),
        ).fetchone()
        return _public_row(dict(row))


def list_user_mcp_connections(user_id: str) -> list[dict[str, Any]]:
    """Return all MCP connections for a user (secrets stripped)."""
    if not user_id or user_id == "anonymous":
        return []
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, server_slug, display_name, transport, url,
                   auth_type, cached_tools, enabled, status, last_error, created_at, updated_at
            FROM user_mcp_connections
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_public_row(dict(r)) for r in rows]


def get_enabled_user_mcps(user_id: str, include_secrets: bool = False) -> list[dict[str, Any]]:
    """Return enabled, connected MCP servers for a user."""
    if not user_id or user_id == "anonymous":
        return []
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, server_slug, display_name, transport, url,
                   auth_type, encrypted_auth, cached_tools, enabled, status
            FROM user_mcp_connections
            WHERE user_id = %s AND enabled = TRUE AND status = 'connected'
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()

    results = []
    for r in rows:
        item = dict(r)
        item["id"] = str(item["id"])
        if include_secrets:
            item["auth"] = decrypt_json(item.pop("encrypted_auth", ""))
        else:
            item.pop("encrypted_auth", None)
        results.append(item)
    return results


def update_mcp_tokens(conn_id: str, auth_payload: dict[str, Any]) -> None:
    """Update encrypted OAuth tokens after a token refresh."""
    enc = encrypt_json(auth_payload)
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE user_mcp_connections SET encrypted_auth = %s, updated_at = NOW() WHERE id = %s",
            (enc, conn_id),
        )


def toggle_mcp_connection(user_id: str, conn_id: str, enabled: bool) -> dict[str, Any] | None:
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """
            UPDATE user_mcp_connections
            SET enabled = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING id, user_id, server_slug, display_name, transport, url,
                      auth_type, cached_tools, enabled, status, last_error, created_at, updated_at
            """,
            (enabled, conn_id, user_id),
        ).fetchone()
    return _public_row(dict(row)) if row else None


def delete_mcp_connection(user_id: str, conn_id: str) -> bool:
    pool = get_pool()
    with pool.connection() as conn:
        cur = conn.execute(
            "DELETE FROM user_mcp_connections WHERE id = %s AND user_id = %s",
            (conn_id, user_id),
        )
    return cur.rowcount > 0


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    row["id"] = str(row["id"])
    for dt_key in ("created_at", "updated_at"):
        if row.get(dt_key) is not None:
            row[dt_key] = row[dt_key].isoformat()
    row.pop("encrypted_auth", None)
    return row
