"""
src/api/routers/mcp_servers.py
------------------------------
REST + OAuth endpoints for user-connected dynamic MCP servers:
  GET    /api/mcp/catalog                — List curated MCP catalog & instructions
  GET    /api/mcp/servers                — List user's connected MCP servers & tools
  POST   /api/mcp/connect                — Connect URL+API key OR start OAuth flow
  GET    /api/mcp/oauth/callback         — OAuth2/PKCE callback -> save encrypted tokens
  PATCH  /api/mcp/servers/{conn_id}      — Toggle enabled state
  DELETE /api/mcp/servers/{conn_id}      — Disconnect and delete credentials
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.core.logging_config import get_logger
from src.mcp_dynamic.catalog import (
    MCP_CATALOG,
    get_catalog_entry,
    validate_remote_mcp_url,
)
from src.mcp_dynamic.client import discover_remote_mcp_tools
from src.mcp_dynamic.oauth import (
    exchange_oauth_callback,
    start_google_oauth,
    start_remote_mcp_oauth,
)
from src.mcp_dynamic.store import (
    delete_mcp_connection,
    list_user_mcp_connections,
    toggle_mcp_connection,
    upsert_mcp_connection,
)

logger = get_logger("TripMateAI.api.mcp")
router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class ConnectMcpRequest(BaseModel):
    server_slug: str = "custom_remote"
    display_name: str = ""
    url: str = ""
    transport: str = "streamable_http"  # streamable_http | sse
    api_key: str = ""
    header_name: str = "Authorization"


class ToggleMcpRequest(BaseModel):
    enabled: bool


@router.get("/catalog")
def get_mcp_catalog(current_user: dict = Depends(get_current_user)):
    """Return available MCP integrations + setup instructions."""
    return {"catalog": MCP_CATALOG}


@router.get("/servers")
def get_user_mcp_servers(current_user: dict = Depends(get_current_user)):
    """List connected MCP servers for the logged-in user (secrets stripped)."""
    user_id = str(current_user["id"])
    return {"servers": list_user_mcp_connections(user_id)}


@router.post("/connect")
async def connect_mcp_server(
    req: ConnectMcpRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Connect either:
      - Built-in Google OAuth server (google_drive, google_calendar) -> returns authorize_url
      - Remote MCP URL + optional API key -> discovers tools & saves immediately
      - Remote OAuth 2.1 MCP URL -> discovers .well-known metadata & returns authorize_url
    """
    user_id = str(current_user["id"])
    slug = (req.server_slug or "custom_remote").strip()

    # 1. Built-in Google OAuth integrations
    if slug in ("google_drive", "google_calendar"):
        entry = get_catalog_entry(slug)
        if not entry:
            raise HTTPException(status_code=404, detail="Unknown integration.")
        try:
            auth_url = start_google_oauth(
                user_id=user_id,
                server_slug=slug,
                display_name=req.display_name.strip() or entry["name"],
                scopes=entry["scopes"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "auth_required", "authorize_url": auth_url}

    # 2. Remote MCP server (URL + optional API key, or OAuth 2.1)
    try:
        clean_url = validate_remote_mcp_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    transport = req.transport if req.transport in ("streamable_http", "sse") else "streamable_http"
    display_name = req.display_name.strip() or clean_url.split("//", 1)[-1].split("/", 1)[0]

    headers: dict[str, str] = {}
    auth_type = "none"
    auth_payload: dict[str, Any] = {}

    if req.api_key.strip():
        auth_type = "api_key"
        h_name = req.header_name.strip() or "Authorization"
        val = req.api_key.strip()
        if h_name.lower() == "authorization" and not val.lower().startswith("bearer "):
            val = f"Bearer {val}"
        headers[h_name] = val
        auth_payload = {"api_key": req.api_key.strip(), "header_name": h_name}

    try:
        tools = await discover_remote_mcp_tools(clean_url, transport=transport, headers=headers)
        conn = upsert_mcp_connection(
            user_id=user_id,
            server_slug=slug,
            display_name=display_name,
            transport=transport,
            url=clean_url,
            auth_type=auth_type,
            auth_payload=auth_payload,
            cached_tools=tools,
        )
        logger.info("[MCP_API] Connected remote MCP '%s' (%d tools) for user=%s", display_name, len(tools), user_id)
        return {"status": "connected", "connection": conn}

    except Exception as exc:
        err_msg = str(exc)
        # If no API key was supplied and server looks auth-gated, try OAuth 2.1 discovery
        if not req.api_key.strip() and any(k in err_msg.lower() for k in ("401", "unauthorized", "auth", "forbidden")):
            try:
                auth_url = start_remote_mcp_oauth(
                    user_id=user_id,
                    mcp_url=clean_url,
                    transport=transport,
                    display_name=display_name,
                )
                return {"status": "auth_required", "authorize_url": auth_url}
            except ValueError as oauth_exc:
                raise HTTPException(status_code=400, detail=str(oauth_exc))

        logger.warning("[MCP_API] Failed to connect '%s': %s", clean_url, err_msg)
        raise HTTPException(
            status_code=400,
            detail=f"Could not connect to MCP server: {err_msg[:260]}",
        )


@router.get("/oauth/callback", response_class=HTMLResponse)
async def mcp_oauth_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
):
    """OAuth2 redirect target — exchanges code, discovers tools, notifies popup opener."""
    if error or not code or not state:
        msg = error or "Authorization was cancelled or missing parameters."
        return _popup_html(success=False, message=msg)

    try:
        exchanged = exchange_oauth_callback(state=state, code=code)
        slug = exchanged["server_slug"]

        if exchanged["kind"] == "google_oauth":
            entry = get_catalog_entry(slug)
            tools = entry["tools_preview"] if entry else []
            auth_type = "google_oauth"
        else:
            tok = exchanged["auth_payload"].get("access_token", "")
            tools = await discover_remote_mcp_tools(
                exchanged["mcp_url"],
                transport=exchanged["transport"],
                headers={"Authorization": f"Bearer {tok}"} if tok else None,
            )
            auth_type = "remote_oauth"

        conn = upsert_mcp_connection(
            user_id=exchanged["user_id"],
            server_slug=slug,
            display_name=exchanged["display_name"],
            transport=exchanged["transport"],
            url=exchanged["mcp_url"],
            auth_type=auth_type,
            auth_payload=exchanged["auth_payload"],
            cached_tools=tools,
        )
        logger.info("[MCP_OAUTH] Connected '%s' (%d tools) for user=%s", conn["display_name"], len(tools), exchanged["user_id"])
        return _popup_html(success=True, message=f"Connected {conn['display_name']}!", connection=conn)

    except Exception as exc:
        logger.error("[MCP_OAUTH] Callback error: %s", exc)
        return _popup_html(success=False, message=str(exc))


@router.patch("/servers/{conn_id}")
def toggle_server(
    conn_id: str,
    req: ToggleMcpRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    updated = toggle_mcp_connection(user_id, conn_id, req.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    return {"connection": updated}


@router.delete("/servers/{conn_id}")
def remove_server(
    conn_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["id"])
    deleted = delete_mcp_connection(user_id, conn_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    return {"deleted": True, "id": conn_id}


def _popup_html(success: bool, message: str, connection: dict[str, Any] | None = None) -> HTMLResponse:
    payload = json.dumps({
        "type": "mcp_oauth_result",
        "success": success,
        "message": message,
        "connection": connection,
    })
    safe_msg = message.replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>TripMate MCP Auth</title></head>
<body style="font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0f172a;color:#f8fafc;">
  <div style="text-align:center;max-width:360px;padding:24px;border-radius:12px;background:#1e293b;">
    <h2 style="margin:0 0 8px;">{"Connected!" if success else "Connection Failed"}</h2>
    <p style="margin:0 0 16px;color:#cbd5e1;font-size:14px;">{safe_msg}</p>
    <p style="font-size:12px;color:#94a3b8;">You can close this window.</p>
  </div>
  <script>
    try {{
      if (window.opener) {{
        window.opener.postMessage({payload}, "*");
      }}
    }} catch (e) {{}}
    setTimeout(function() {{ window.close(); }}, 1200);
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
