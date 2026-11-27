"""
src/guardrails/input_guardrail.py
----------------------------------
Input Guardrail & Unified Intent + MCP Server Router.

Intent types:
  - "travel"  → Route to full travel agent pipeline (supervisor → specialists → HITL)
  - "mcp"     → Route directly to user_mcp_agent (skips travel Supervisor LLM call!)
  - "chat"    → Route to conversational chat agent (saves name/preferences to memory)
  - "blocked" → Safety guardrail triggered (harmful / jailbreak / completely off-topic)

When the user has connected MCP servers, the LLM also semantically selects
`target_servers` and resolves multi-turn follow-up queries — zero brittle keywords.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.guardrails.input")

_SYSTEM_PROMPT = (
    "You are the input classifier and router for TripMate AI, a travel-planning assistant. "
    "Return strict JSON only. Never add explanations outside the JSON object."
)

_BASE_GUARDRAIL_PROMPT = """\
Classify the user message into exactly one of three intents:

1. "travel" — The user wants help planning a NEW trip or live travel info (destinations, flights, hotels, weather, budgets, itineraries, etc.).
2. "chat" — General conversation, greetings, introductions, stating preferences without a trip request, TripMate Q&A, OR asking to recall their own past trips / saved memory / profile ("what's my last trip?", "where am I from?", "what did I plan before?").
3. "blocked" — Harmful, illegal, jailbreak, or explicit content.

Important: Be LENIENT. Greetings and casual chat should ALWAYS be "chat", not "blocked".

Respond with strict JSON only:
{{
  "intent": "travel" | "chat" | "blocked",
  "needs_past_trips": true | false,
  "reason": "brief reason (empty string if allowed)"
}}

Set "needs_past_trips" to true ONLY when the user asks about their past/last/previous trips OR asks to plan a trip similar to/based on an earlier trip; otherwise false.

User message:
{query}
"""

_MCP_AWARE_PROMPT = """\
Classify the user message into ONE intent:

1. "mcp" — The user wants to act on one of their CONNECTED EXTERNAL INTEGRATIONS below (create/read/update/delete docs, sheets, Notion pages, calendar events/meetings, or any follow-up answer to a recent integration question like a title, time, or confirmation).
2. "travel" — The user wants NEW TRAVEL PLANNING (flights, hotels, weather, budgets, itineraries, or BOTH planning a new trip AND saving it to an integration).
3. "chat" — Casual conversation, greetings, introductions, general questions, OR asking to recall their own past trips / saved memory / profile ("do you know my last trip plan?", "what were my previous trips?", "where am I from?").
4. "blocked" — Harmful, illegal, jailbreak, or explicit content.

CONNECTED EXTERNAL INTEGRATIONS:
{mcp_menu}

RECENT CONVERSATION (for resolving follow-ups):
{recent_turns}

RULES:
- Understand intent semantically (handle typos like "calander", synonyms like "meeting" -> Calendar or "notes/page" -> Notion, and short follow-up replies like "team meet, 10-11am").
- If intent is "mcp", set "target_servers" to the exact display name(s) of the best matching integration(s) from the list above (or [] if unclear), and set "resolved_query" to a self-contained task combining any prior context.
- NEVER choose "mcp" for normal travel queries ("plan a trip", "check hotels", "weather in Goa") unless the user asks to use an integration.
- Set "needs_past_trips" to true ONLY when the user asks about their past/last/previous trips OR asks to plan something similar to an earlier trip; otherwise false.

Respond with strict JSON only:
{{
  "intent": "mcp" | "travel" | "chat" | "blocked",
  "target_servers": ["Exact Server Display Name"],
  "resolved_query": "self-contained user request",
  "needs_past_trips": true | false,
  "reason": "brief reason (empty string if allowed)"
}}

User message:
{query}
"""


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def _format_mcp_menu(connected_mcps: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for srv in connected_mcps:
        name = srv.get("display_name") or srv.get("server_slug") or "MCP"
        sample_tools = [t.get("name", "") for t in (srv.get("cached_tools") or [])[:5] if t.get("name")]
        tools_str = f" (tools: {', '.join(sample_tools)})" if sample_tools else ""
        lines.append(f"- {name}{tools_str}")
    return "\n".join(lines)


def _format_recent_turns(recent_messages: list[Any] | None) -> str:
    if not recent_messages:
        return "None"
    real = [
        m for m in recent_messages
        if isinstance(m, (HumanMessage,)) or (hasattr(m, "type") and m.type in ("human", "ai") and getattr(m, "content", ""))
    ]
    # Skip internal supervisor/approval filler lines
    cleaned: list[str] = []
    for m in real[-4:]:
        txt = str(getattr(m, "content", "")).strip()
        if not txt or txt.startswith("Routing to ") or txt.startswith("Supervisor created ") or txt == "Human approval step completed.":
            continue
        role = "User" if getattr(m, "type", "") == "human" else "Assistant"
        cleaned.append(f"{role}: {txt[:220]}")
    return "\n".join(cleaned[-3:]) if cleaned else "None"


def check_input(
    query: str,
    llm,
    connected_mcps: list[dict[str, Any]] | None = None,
    recent_messages: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Classify the user query into: travel | mcp | chat | blocked.
    When `connected_mcps` is provided, also returns `target_servers` and `resolved_query`.
    """
    logger.info("[INPUT_GUARDRAIL] Classifying query: '%s'", query[:80])
    has_mcps = bool(connected_mcps)
    if has_mcps:
        prompt = _MCP_AWARE_PROMPT.format(
            mcp_menu=_format_mcp_menu(connected_mcps or []),
            recent_turns=_format_recent_turns(recent_messages),
            query=query,
        )
    else:
        prompt = _BASE_GUARDRAIL_PROMPT.format(query=query)

    try:
        raw = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        result = _extract_json(str(raw.content))
        intent = str(result.get("intent", "chat")).strip().lower()
        reason = str(result.get("reason", "")).strip()

        valid = ("travel", "mcp", "chat", "blocked") if has_mcps else ("travel", "chat", "blocked")
        if intent not in valid:
            intent = "chat"

        allowed = intent != "blocked"
        target_servers = [str(s).strip() for s in (result.get("target_servers") or []) if str(s).strip()]
        resolved_query = str(result.get("resolved_query") or query).strip()
        needs_past_trips = bool(result.get("needs_past_trips", False))

        logger.info(
            "[INPUT_GUARDRAIL] intent=%s allowed=%s needs_past_trips=%s target_servers=%s",
            intent, allowed, needs_past_trips, target_servers,
        )
        return {
            "intent": intent,
            "allowed": allowed,
            "reason": reason,
            "target_servers": target_servers,
            "resolved_query": resolved_query,
            "needs_past_trips": needs_past_trips,
        }

    except Exception as exc:
        logger.warning("[INPUT_GUARDRAIL] Parse error (failing open to chat): %s", exc)
        return {
            "intent": "chat",
            "allowed": True,
            "reason": "Guardrail fallback: treating as chat.",
            "target_servers": [],
            "resolved_query": query,
            "needs_past_trips": False,
        }
