"""
src/mcp_dynamic/catalog.py
--------------------------
Curated MCP integration catalog + SSRF-safe remote URL validator.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

MCP_CATALOG: list[dict[str, Any]] = [
    {
        "slug": "google_drive",
        "name": "Google Drive",
        "category": "Productivity",
        "description": "Save finalized trip itineraries as formatted Google Docs in your Drive.",
        "auth_mode": "google_oauth",
        "transport": "builtin_google",
        "scopes": [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/documents",
        ],
        "tools_preview": [
            {"name": "create_google_doc", "description": "Create a formatted Google Doc in Drive."},
            {"name": "list_drive_files", "description": "Search recent files in your Google Drive."},
        ],
        "setup_hint": "Requires GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in backend .env.",
    },
    {
        "slug": "google_calendar",
        "name": "Google Calendar",
        "category": "Scheduling",
        "description": "Add trip dates, flight reminders, and itinerary schedules to your Google Calendar.",
        "auth_mode": "google_oauth",
        "transport": "builtin_google",
        "scopes": [
            "https://www.googleapis.com/auth/calendar.events",
        ],
        "tools_preview": [
            {"name": "create_calendar_event", "description": "Create a trip or flight event on your primary Calendar."},
            {"name": "list_upcoming_events", "description": "Check upcoming events on your Google Calendar."},
        ],
        "setup_hint": "Requires GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in backend .env.",
    },
    {
        "slug": "custom_remote",
        "name": "Custom Remote MCP",
        "category": "Custom / OAuth 2.1",
        "description": "Connect any remote MCP server (streamable_http or sse) via API Key, Bearer token, or OAuth 2.1.",
        "auth_mode": "remote",
        "transport": "streamable_http",
        "scopes": [],
        "tools_preview": [],
        "setup_hint": "Enter any HTTPS MCP endpoint URL (e.g. https://mcp.example.com/mcp) and optional API key.",
    },
]


def get_catalog_entry(slug: str) -> dict[str, Any] | None:
    for item in MCP_CATALOG:
        if item["slug"] == slug:
            return item
    return None


def validate_remote_mcp_url(raw_url: str) -> str:
    """
    Validate that a user-supplied remote MCP URL is HTTP/HTTPS and does not
    target private/loopback/link-local addresses (SSRF protection).
    """
    url = (raw_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("MCP URL must start with http:// or https:// and include a valid host.")

    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError("Loopback/localhost URLs are blocked for security.")

    # Resolve IP and block private/reserved/link-local ranges (e.g. cloud metadata 169.254.169.254)
    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
            ):
                raise ValueError(f"Private or reserved IP addresses are not allowed ({ip_str}).")
    except socket.gaierror:
        # Allow DNS resolution failures to be caught by the actual HTTP handshake error
        pass

    return url
