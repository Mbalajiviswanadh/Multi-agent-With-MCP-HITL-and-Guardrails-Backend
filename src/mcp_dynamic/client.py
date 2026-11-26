"""
src/mcp_dynamic/client.py
-------------------------
Discovers tools from user-connected remote MCP servers and builds
per-request LangChain tools for:
  - Remote MCP servers (`streamable_http` / `sse`) via MultiServerMCPClient
  - Built-in Google Drive & Google Calendar tools backed by Google REST APIs
    using the user's encrypted & auto-refreshed Google OAuth2 token.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger
from src.mcp_dynamic.oauth import ensure_fresh_access_token
from src.mcp_dynamic.store import get_enabled_user_mcps

logger = get_logger("TripMateAI.mcp_dynamic.client")


def _serialize_tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "name": getattr(tool, "name", "unknown"),
        "description": (getattr(tool, "description", "") or "")[:240],
    }


async def discover_remote_mcp_tools(
    url: str,
    transport: str = "streamable_http",
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform a 1-time handshake with a remote MCP server to verify connectivity
    and cache its available tool schemas.
    """
    server_cfg: dict[str, Any] = {
        "transport": transport if transport in ("streamable_http", "sse") else "streamable_http",
        "url": url,
    }
    if headers:
        server_cfg["headers"] = headers

    client = MultiServerMCPClient({"user_server": server_cfg})
    tools = await client.get_tools()
    return [_serialize_tool_schema(t) for t in tools]


# =========================================================
# Built-in Google Drive & Calendar Tools (OAuth2 REST)
# =========================================================

class _CreateDocInput(BaseModel):
    title: str = Field(description="Title of the Google Doc to create (e.g. 'Goa 3-Day Trip Plan').")
    content: str = Field(description="Full text/markdown content to write into the Google Doc.")


class _ListDriveInput(BaseModel):
    query: str = Field(default="", description="Optional search term for file name.")


class _CreateEventInput(BaseModel):
    summary: str = Field(description="Event title (e.g. 'Goa 3-Day Vacation').")
    start_date: str = Field(description="Start date or ISO datetime (e.g. '2026-12-05' or '2026-12-05T09:00:00Z').")
    end_date: str = Field(description="End date or ISO datetime (e.g. '2026-12-08' or '2026-12-08T18:00:00Z').")
    description: str = Field(default="", description="Trip summary, hotel, or flight notes.")
    location: str = Field(default="", description="Destination city or venue.")


class _ListEventsInput(BaseModel):
    max_results: int = Field(default=5, description="Maximum number of upcoming events to list.")


def _build_google_drive_tools(conn: dict[str, Any]) -> list[StructuredTool]:
    def create_google_doc(title: str, content: str) -> str:
        token = ensure_fresh_access_token(conn)
        if not token:
            return "Error: Google Drive OAuth token is missing or expired. Please reconnect Google Drive."
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with httpx.Client(timeout=20.0) as client:
            # 1. Create empty Google Doc
            res = client.post(
                "https://docs.googleapis.com/v1/documents",
                headers=headers,
                json={"title": title},
            )
            if res.status_code >= 400:
                return f"Google Docs API error ({res.status_code}): {res.text[:250]}"
            doc = res.json()
            doc_id = doc["documentId"]

            # 2. Insert itinerary text
            if content:
                upd = client.post(
                    f"https://docs.googleapis.com/v1/documents/{doc_id}:batchUpdate",
                    headers=headers,
                    json={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
                )
                if upd.status_code >= 400:
                    return f"Created doc '{title}' (https://docs.google.com/document/d/{doc_id}/edit) but failed to write body: {upd.text[:200]}"

        return f"Created Google Doc '{title}': https://docs.google.com/document/d/{doc_id}/edit"

    def list_drive_files(query: str = "") -> str:
        token = ensure_fresh_access_token(conn)
        if not token:
            return "Error: Google Drive OAuth token is missing. Please reconnect Google Drive."
        params: dict[str, Any] = {
            "pageSize": 10,
            "fields": "files(id,name,mimeType,webViewLink)",
        }
        if query:
            safe_q = query.replace("'", "\\'")
            params["q"] = f"name contains '{safe_q}' and trashed = false"
        else:
            params["q"] = "trashed = false"

        with httpx.Client(timeout=15.0) as client:
            res = client.get(
                "https://www.googleapis.com/drive/v3/files",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            if res.status_code >= 400:
                return f"Google Drive API error ({res.status_code}): {res.text[:250]}"
            files = res.json().get("files", [])
            if not files:
                return "No matching files found in your Google Drive."
            return "\n".join(f"- {f['name']} ({f.get('webViewLink', f['id'])})" for f in files)

    return [
        StructuredTool.from_function(
            func=create_google_doc,
            name="create_google_doc",
            description="Create a new Google Doc in the user's Google Drive with the given title and trip plan content.",
            args_schema=_CreateDocInput,
        ),
        StructuredTool.from_function(
            func=list_drive_files,
            name="list_drive_files",
            description="List or search recent files in the user's Google Drive.",
            args_schema=_ListDriveInput,
        ),
    ]


def _build_google_calendar_tools(conn: dict[str, Any]) -> list[StructuredTool]:
    def create_calendar_event(
        summary: str,
        start_date: str,
        end_date: str,
        description: str = "",
        location: str = "",
    ) -> str:
        token = ensure_fresh_access_token(conn)
        if not token:
            return "Error: Google Calendar OAuth token is missing. Please reconnect Google Calendar."

        is_date_only = "T" not in start_date
        body: dict[str, Any] = {
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"date": start_date[:10]} if is_date_only else {"dateTime": start_date},
            "end": {"date": end_date[:10]} if is_date_only else {"dateTime": end_date},
        }
        with httpx.Client(timeout=15.0) as client:
            res = client.post(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            if res.status_code >= 400:
                return f"Google Calendar API error ({res.status_code}): {res.text[:250]}"
            ev = res.json()
            return f"Created Calendar Event '{ev.get('summary', summary)}': {ev.get('htmlLink', '')}"

    def list_upcoming_events(max_results: int = 5) -> str:
        token = ensure_fresh_access_token(conn)
        if not token:
            return "Error: Google Calendar OAuth token is missing."
        with httpx.Client(timeout=15.0) as client:
            res = client.get(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events",
                headers={"Authorization": f"Bearer {token}"},
                params={"maxResults": max_results, "singleEvents": True, "orderBy": "startTime"},
            )
            if res.status_code >= 400:
                return f"Google Calendar API error ({res.status_code}): {res.text[:250]}"
            items = res.json().get("items", [])
            if not items:
                return "No upcoming events found."
            lines = []
            for it in items:
                start = it.get("start", {}).get("dateTime") or it.get("start", {}).get("date", "")
                lines.append(f"- {it.get('summary', '(No title)')} on {start}")
            return "\n".join(lines)

    return [
        StructuredTool.from_function(
            func=create_calendar_event,
            name="create_calendar_event",
            description="Create a trip event on the user's primary Google Calendar.",
            args_schema=_CreateEventInput,
        ),
        StructuredTool.from_function(
            func=list_upcoming_events,
            name="list_upcoming_events",
            description="List upcoming events on the user's primary Google Calendar.",
            args_schema=_ListEventsInput,
        ),
    ]


# =========================================================
# Per-Request Dynamic Tool Loader
# =========================================================

async def load_user_mcp_tools_async(user_id: str) -> tuple[list[Any], dict[str, str]]:
    """
    Load all enabled MCP tools for `user_id` lazily at execution time.
    Returns (tools_list, tool_name_to_server_label_map).
    """
    connections = get_enabled_user_mcps(user_id, include_secrets=True)
    if not connections:
        return [], {}

    all_tools: list[Any] = []
    tool_origin: dict[str, str] = {}
    remote_servers: dict[str, dict[str, Any]] = {}
    remote_label_by_key: dict[str, str] = {}

    for conn in connections:
        slug = conn["server_slug"]
        label = conn["display_name"]

        if slug == "google_drive":
            for t in _build_google_drive_tools(conn):
                all_tools.append(t)
                tool_origin[t.name] = label
        elif slug == "google_calendar":
            for t in _build_google_calendar_tools(conn):
                all_tools.append(t)
                tool_origin[t.name] = label
        elif conn.get("url"):
            server_key = f"mcp_{conn['id'][:8]}"
            cfg: dict[str, Any] = {
                "transport": conn.get("transport") or "streamable_http",
                "url": conn["url"],
            }
            auth = conn.get("auth") or {}
            headers: dict[str, str] = {}
            if conn.get("auth_type") == "api_key" and auth.get("api_key"):
                header_name = auth.get("header_name") or "Authorization"
                val = auth["api_key"]
                if header_name.lower() == "authorization" and not val.lower().startswith("bearer "):
                    val = f"Bearer {val}"
                headers[header_name] = val
            elif conn.get("auth_type") in ("oauth2", "remote_oauth"):
                tok = ensure_fresh_access_token(conn)
                if tok:
                    headers["Authorization"] = f"Bearer {tok}"

            if headers:
                cfg["headers"] = headers
            remote_servers[server_key] = cfg
            remote_label_by_key[server_key] = label

    if remote_servers:
        try:
            client = MultiServerMCPClient(remote_servers)
            remote_tools = await client.get_tools()
            for t in remote_tools:
                all_tools.append(t)
                # Best-effort origin label
                tool_origin[t.name] = next(iter(remote_label_by_key.values()), "Remote MCP")
        except Exception as exc:
            logger.error("[MCP_CLIENT] Failed loading remote MCP tools for user=%s: %s", user_id, exc)

    return all_tools, tool_origin


def load_user_mcp_tools(user_id: str) -> tuple[list[Any], dict[str, str]]:
    """Synchronous wrapper for `load_user_mcp_tools_async`."""
    return asyncio.run(load_user_mcp_tools_async(user_id))
