"""
src/mcp_dynamic/oauth.py
------------------------
OAuth 2.1 + PKCE authorization flows for:
  1. Built-in Google integrations (Google Drive, Google Calendar) using your
     Google Cloud OAuth Client ID & Secret.
  2. Standard OAuth-protected Remote MCP servers (auto-discovers
     `/.well-known/oauth-protected-resource` & `/.well-known/oauth-authorization-server`
     and performs RFC 7591 Dynamic Client Registration + PKCE).
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from src.core.config import settings
from src.core.logging_config import get_logger
from src.mcp_dynamic.store import update_mcp_tokens

logger = get_logger("TripMateAI.mcp_dynamic.oauth")

# In-memory short-lived PKCE state store (TTL 10 minutes)
_PENDING_STATES: dict[str, dict[str, Any]] = {}
_STATE_TTL_SECONDS = 600


def _prune_expired_states() -> None:
    now = time.time()
    expired = [k for k, v in _PENDING_STATES.items() if now - v.get("created_at", 0) > _STATE_TTL_SECONDS]
    for k in expired:
        _PENDING_STATES.pop(k, None)


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def callback_url() -> str:
    base = (settings.oauth_redirect_base_url or "http://localhost:8000").rstrip("/")
    return f"{base}/api/mcp/oauth/callback"


# =========================================================
# 1. Google OAuth 2.0 (Drive & Calendar)
# =========================================================

def start_google_oauth(
    *,
    user_id: str,
    server_slug: str,
    display_name: str,
    scopes: list[str],
) -> str:
    """Build Google OAuth consent URL and store PKCE state."""
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise ValueError(
            "Google OAuth is not configured on the backend. "
            "Add GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET to .env."
        )

    _prune_expired_states()
    state = secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()

    _PENDING_STATES[state] = {
        "kind": "google_oauth",
        "user_id": user_id,
        "server_slug": server_slug,
        "display_name": display_name,
        "scopes": scopes,
        "code_verifier": verifier,
        "created_at": time.time(),
    }

    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": callback_url(),
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


# =========================================================
# 2. Standard Remote MCP OAuth 2.1 Discovery + PKCE
# =========================================================

def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def discover_mcp_oauth_metadata(mcp_url: str) -> dict[str, Any] | None:
    """
    Probe a remote MCP URL's `.well-known` metadata endpoints per the
    MCP Authorization specification. Returns None if server doesn't advertise OAuth.
    """
    base = _origin(mcp_url)
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        auth_server_base = base
        try:
            prm = client.get(f"{base}/.well-known/oauth-protected-resource")
            if prm.status_code == 200:
                prm_data = prm.json()
                servers = prm_data.get("authorization_servers") or []
                if servers:
                    auth_server_base = servers[0].rstrip("/")
        except Exception:
            pass

        try:
            as_meta = client.get(f"{auth_server_base}/.well-known/oauth-authorization-server")
            if as_meta.status_code == 200:
                return as_meta.json()
        except Exception:
            pass
    return None


def start_remote_mcp_oauth(
    *,
    user_id: str,
    mcp_url: str,
    transport: str,
    display_name: str,
) -> str:
    """Discover remote MCP OAuth metadata, register client (RFC 7591), return authorize URL."""
    meta = discover_mcp_oauth_metadata(mcp_url)
    if not meta or not meta.get("authorization_endpoint") or not meta.get("token_endpoint"):
        raise ValueError(
            "This MCP server requires authentication, but does not expose OAuth 2.1 metadata. "
            "Please provide an API Key or Bearer Token instead."
        )

    client_id = ""
    client_secret = ""
    reg_endpoint = meta.get("registration_endpoint")
    if reg_endpoint:
        try:
            with httpx.Client(timeout=10.0) as client:
                reg_res = client.post(
                    reg_endpoint,
                    json={
                        "client_name": "TripMate AI",
                        "redirect_uris": [callback_url()],
                        "grant_types": ["authorization_code", "refresh_token"],
                        "response_types": ["code"],
                        "token_endpoint_auth_method": "none",
                    },
                )
                if reg_res.status_code in (200, 201):
                    reg_data = reg_res.json()
                    client_id = reg_data.get("client_id", "")
                    client_secret = reg_data.get("client_secret", "")
        except Exception as exc:
            logger.warning("[MCP_OAUTH] Dynamic client registration failed: %s", exc)

    if not client_id:
        raise ValueError(
            "Remote MCP server OAuth does not support Dynamic Client Registration. "
            "Please connect using an API Key / Bearer Token."
        )

    _prune_expired_states()
    state = secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()

    _PENDING_STATES[state] = {
        "kind": "remote_oauth",
        "user_id": user_id,
        "server_slug": "custom_remote",
        "display_name": display_name,
        "mcp_url": mcp_url,
        "transport": transport,
        "token_endpoint": meta["token_endpoint"],
        "client_id": client_id,
        "client_secret": client_secret,
        "code_verifier": verifier,
        "created_at": time.time(),
    }

    params = {
        "client_id": client_id,
        "redirect_uri": callback_url(),
        "response_type": "code",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    scopes = meta.get("scopes_supported")
    if isinstance(scopes, list) and scopes:
        params["scope"] = " ".join(scopes[:6])

    sep = "&" if "?" in meta["authorization_endpoint"] else "?"
    return f"{meta['authorization_endpoint']}{sep}{urlencode(params)}"


# =========================================================
# 3. Callback Token Exchange & Refresh
# =========================================================

def exchange_oauth_callback(state: str, code: str) -> dict[str, Any]:
    """Exchange authorization code for access + refresh tokens."""
    _prune_expired_states()
    pending = _PENDING_STATES.pop(state, None)
    if not pending:
        raise ValueError("Invalid or expired OAuth state. Please try connecting again.")

    if pending["kind"] == "google_oauth":
        token_url = "https://oauth2.googleapis.com/token"
        form = {
            "client_id": settings.google_oauth_client_id,
            "client_secret": settings.google_oauth_client_secret,
            "code": code,
            "code_verifier": pending["code_verifier"],
            "grant_type": "authorization_code",
            "redirect_uri": callback_url(),
        }
    else:
        token_url = pending["token_endpoint"]
        form = {
            "client_id": pending["client_id"],
            "code": code,
            "code_verifier": pending["code_verifier"],
            "grant_type": "authorization_code",
            "redirect_uri": callback_url(),
        }
        if pending.get("client_secret"):
            form["client_secret"] = pending["client_secret"]

    with httpx.Client(timeout=15.0) as client:
        res = client.post(token_url, data=form)
        if res.status_code >= 400:
            raise ValueError(f"OAuth token exchange failed ({res.status_code}): {res.text[:250]}")
        tok = res.json()

    expires_in = int(tok.get("expires_in", 3600))
    auth_payload = {
        "access_token": tok.get("access_token", ""),
        "refresh_token": tok.get("refresh_token", ""),
        "expires_at": time.time() + expires_in - 60,
        "token_endpoint": token_url,
        "client_id": pending.get("client_id", ""),
        "client_secret": pending.get("client_secret", ""),
    }

    return {
        "kind": pending["kind"],
        "user_id": pending["user_id"],
        "server_slug": pending["server_slug"],
        "display_name": pending["display_name"],
        "mcp_url": pending.get("mcp_url", ""),
        "transport": pending.get("transport", "builtin_google"),
        "auth_payload": auth_payload,
    }


def ensure_fresh_access_token(conn: dict[str, Any]) -> str:
    """Return a valid access_token for an OAuth connection, refreshing if expired."""
    auth = conn.get("auth") or {}
    access_token = auth.get("access_token", "")
    expires_at = float(auth.get("expires_at", 0) or 0)

    if access_token and (expires_at == 0 or time.time() < expires_at):
        return access_token

    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return access_token

    is_google = conn.get("auth_type") == "google_oauth"
    token_url = auth.get("token_endpoint") or (
        "https://oauth2.googleapis.com/token" if is_google else ""
    )
    if not token_url:
        return access_token

    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": settings.google_oauth_client_id if is_google else auth.get("client_id", ""),
    }
    secret = settings.google_oauth_client_secret if is_google else auth.get("client_secret", "")
    if secret:
        form["client_secret"] = secret

    try:
        with httpx.Client(timeout=12.0) as client:
            res = client.post(token_url, data=form)
            if res.status_code < 400:
                tok = res.json()
                new_access = tok.get("access_token", access_token)
                expires_in = int(tok.get("expires_in", 3600))
                auth["access_token"] = new_access
                if tok.get("refresh_token"):
                    auth["refresh_token"] = tok["refresh_token"]
                auth["expires_at"] = time.time() + expires_in - 60
                update_mcp_tokens(conn["id"], auth)
                logger.info("[MCP_OAUTH] Refreshed access token for %s", conn.get("server_slug"))
                return new_access
    except Exception as exc:
        logger.warning("[MCP_OAUTH] Token refresh failed for %s: %s", conn.get("server_slug"), exc)

    return access_token
