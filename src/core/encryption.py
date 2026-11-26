"""
src/core/encryption.py
----------------------
Symmetric Fernet (AES-128-CBC + HMAC-SHA256) encryption for storing user
MCP OAuth tokens and API keys safely in PostgreSQL.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import settings


def _get_fernet() -> Fernet:
    raw_key = (settings.mcp_encryption_key or "").strip()
    if raw_key:
        return Fernet(raw_key.encode("utf-8"))
    # Deterministic 32-byte Fernet key derived from jwt_secret_key when
    # MCP_ENCRYPTION_KEY is not explicitly set in .env
    digest = hashlib.sha256(settings.jwt_secret_key.encode("utf-8")).digest()
    derived_key = base64.urlsafe_b64encode(digest)
    return Fernet(derived_key)


def encrypt_json(payload: dict[str, Any]) -> str:
    """Encrypt a credentials dict into a URL-safe Fernet token string."""
    if not payload:
        return ""
    raw = json.dumps(payload).encode("utf-8")
    return _get_fernet().encrypt(raw).decode("utf-8")


def decrypt_json(token: str) -> dict[str, Any]:
    """Decrypt a Fernet token string back into a credentials dict."""
    if not token:
        return {}
    try:
        raw = _get_fernet().decrypt(token.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (InvalidToken, ValueError, TypeError):
        return {}
