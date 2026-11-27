"""
src/agents/state.py
-------------------
Upgraded TravelState that supports all three memory tiers:
  - Short-term: messages + context_summary (sliding window)
  - Long-term:  user_profile loaded from user_profiles table
  - Semantic:   similar_past_trips retrieved via pgvector cosine search
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class TravelState(TypedDict, total=False):
    # ── Identity ─────────────────────────────────────────────────────────
    user_id: str           # Logged-in user (links to users table)
    thread_id: str         # Chat session ID (links to threads table)

    # ── Short-Term Working Memory ─────────────────────────────────────────
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    resolved_query: str        # Coreference-resolved / disambiguated query
    # Condensed summary of old messages — prevents token overflow across
    # long multi-turn sessions without losing context
    context_summary: str

    # ── Long-Term User Profile (loaded at graph start) ───────────────────
    user_profile: dict[str, Any]
    # e.g. {"home_airport": "HYD", "preferred_travel_style": "budget",
    #        "dietary_preferences": ["vegetarian"], "frequent_airlines": ["IndiGo"],
    #        "hotel_preferences": ["near city center"], "custom_notes": "..."}

    # ── Semantic / Episodic Memory (pgvector retrieval) ──────────────────
    similar_past_trips: list[dict[str, Any]]
    # List of past itineraries closest to the current query, e.g.:
    # [{"destination": "Goa", "summary": "...", "metadata": {...}}, ...]

    # ── Supervisor + Guardrail ────────────────────────────────────────────
    guardrail_allowed: bool
    guardrail_reason: str
    query_intent: str          # "travel" | "mcp" | "chat" | "blocked"
    needs_past_trips: bool
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # ── Specialist Results ────────────────────────────────────────────────
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str

    # ── Itinerary & HITL ─────────────────────────────────────────────────
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str

    # ── Final Output ──────────────────────────────────────────────────────
    final_response: str
    llm_calls: int

    # ── Dynamic User MCP Integrations ─────────────────────────────────────
    connected_mcp_tools: list[dict[str, Any]]
    target_mcp_servers: list[str]
    user_mcp_results: str
    mcp_trace: dict[str, Any]


def empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def initial_state(user_query: str, user_id: str = "anonymous", thread_id: str = "") -> dict[str, Any]:
    """Return a clean initial state dict for a new travel planning run."""
    from langchain_core.messages import HumanMessage
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "resolved_query": user_query,
        "context_summary": "",
        "user_profile": {},
        "similar_past_trips": [],
        "guardrail_allowed": True,
        "guardrail_reason": "",
        "query_intent": "travel",
        "needs_past_trips": False,
        "selected_agents": [],
        "trip_constraints": empty_constraints(),
        "supervisor_reasoning": "",
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "budget_results": "",
        "itinerary": "",
        "approval_request": "",
        "approved": False,
        "human_feedback": "",
        "final_response": "",
        "llm_calls": 0,
        "connected_mcp_tools": [],
        "target_mcp_servers": [],
        "user_mcp_results": "",
        "mcp_trace": {},
    }
