"""
src/agents/supervisor.py
------------------------
Supervisor Router & Constraint Extractor.
Now has 3 routing paths:
  1. "chat"    → Conversational chat agent (no travel agents called)
  2. "travel"  → Full planning pipeline (specialist agents + HITL)
  3. "blocked" → Safety guardrail triggered

Uses the extracted input_guardrail to classify the request, then:
  - For travel: calls the LLM to dynamically select specialist agents and extract constraints.
  - For chat:   passes through to chat_agent (no extra LLM call needed).
  - For blocked: returns rejection message.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import TravelState, empty_constraints
from src.core.logging_config import get_logger
from src.memory.thread_memory import format_recent_history, merge_constraints
from src.guardrails.input_guardrail import check_input
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.agents.supervisor")

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
    "user_mcp_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
    "user_mcp_agent",
]


def _json_from_llm(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(text[start : end + 1])


def supervisor_agent(state: TravelState) -> dict:
    start_time = time.time()
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    logger.info("[SUPERVISOR] -> Classifying request with guardrail...")

    # 1. Run Intent Classifier (chat / travel / blocked)
    guardrail_result = check_input(query, resilient_llm)
    intent = guardrail_result.get("intent", "chat")
    allowed = guardrail_result["allowed"]
    guardrail_reason = guardrail_result.get("reason", "")
    llm_calls += 1

    # ── BLOCKED ─────────────────────────────────────────────────────────
    if not allowed or intent == "blocked":
        user_response = (
            "I'm sorry, but I can't assist with that request. I'm TripMate AI, your dedicated "
            "travel planning companion! Feel free to ask me about planning trips, discovering destinations, "
            "finding flights, hotels, weather, or local activities."
        )
        internal_reason = guardrail_reason or "Request flagged by safety guardrails as outside travel planning."
        elapsed = time.time() - start_time
        logger.info("[SUPERVISOR] X BLOCKED (%.1fs): %s", elapsed, internal_reason)
        return {
            "guardrail_allowed": False,
            "query_intent": "blocked",
            "guardrail_reason": internal_reason,
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": internal_reason,
            "final_response": user_response,
            "messages": [AIMessage(content=user_response)],
            "llm_calls": llm_calls,
        }

    connected_mcps = state.get("connected_mcp_tools") or []

    def _is_explicit_mcp_request(q: str) -> bool:
        """True ONLY when the user explicitly mentions a connected integration or file/calendar action."""
        ql = q.lower()
        # 1. Explicit connected server names/slugs (e.g. 'notion', 'google drive', 'calendar')
        for srv in connected_mcps:
            disp = str(srv.get("display_name", "")).lower().strip()
            slug = str(srv.get("server_slug", "")).lower().strip()
            if disp and re.search(rf"\b{re.escape(disp)}\b", ql):
                return True
            for tok in re.split(r"[\s_-]+", f"{disp} {slug}"):
                if len(tok) >= 4 and tok not in ("google", "custom", "remote", "server"):
                    if re.search(rf"\b{re.escape(tok)}\b", ql):
                        return True

        # 2. Explicit external artifact phrases
        explicit_phrases = (
            r"\bgoogle\s+docs?\b", r"\bgoogle\s+sheets?\b", r"\bgoogle\s+drive\b",
            r"\bgoogle\s+calendar\b", r"\bspreadsheets?\b", r"\bexcel\b", r"\bmcp\b",
        )
        if any(re.search(p, ql) for p in explicit_phrases):
            return True

        # 3. Action verb + external file/doc/sheet/calendar noun together (never bare verbs like 'check'/'create')
        has_action = bool(re.search(r"\b(create|make|save|export|add|write|put|update|append|delete|remove|read|list|check|open|show)\b", ql))
        has_artifact = bool(re.search(r"\b(doc|docs|document|documents|sheet|sheets|file|files|calendar)\b", ql))
        return has_action and has_artifact

    is_mcp_req = bool(connected_mcps) and _is_explicit_mcp_request(query)

    if is_mcp_req:
        if intent == "chat":
            logger.info("[SUPERVISOR] Overriding 'chat' -> 'travel' (explicit MCP action detected)")
            intent = "travel"

        # Only short-circuit straight to user_mcp_agent when the user is NOT also asking
        # to plan a new trip or search flights/hotels/weather in the same message.
        also_planning_trip = bool(re.search(
            r"\b(plan\s+(a|my|the)\s+.*\btrip\b|flights?|hotels?|weather|forecast|itinerary\s+for)\b",
            query.lower(),
        ))
        if not also_planning_trip:
            elapsed = time.time() - start_time
            logger.info("[SUPERVISOR] OK TRAVEL intent (%.1fs) | Agents: [user_mcp] | Query: '%s'", elapsed, query)
            return {
                "guardrail_allowed": True,
                "query_intent": "travel",
                "guardrail_reason": "",
                "selected_agents": ["user_mcp_agent"],
                "trip_constraints": state.get("trip_constraints") or empty_constraints(),
                "resolved_query": query,
                "supervisor_reasoning": "User explicitly requested an action on their connected MCP integration.",
                "messages": [AIMessage(content="Routing to user_mcp_agent.")],
                "llm_calls": llm_calls,
            }

    # ── CHAT ─────────────────────────────────────────────────────────────
    if intent == "chat":
        elapsed = time.time() - start_time
        logger.info("[SUPERVISOR] -> CHAT intent detected (%.1fs) | Routing to chat_agent", elapsed)
        return {
            "guardrail_allowed": True,
            "query_intent": "chat",
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": "Casual conversation — routing to chat agent.",
            "messages": [AIMessage(content="Routing to chat agent.")],
            "llm_calls": llm_calls,
        }

    # ── TRAVEL ───────────────────────────────────────────────────────────
    # Plan specialist agents & extract trip constraints with query disambiguation
    existing_constraints = state.get("trip_constraints") or empty_constraints()
    context_summary = state.get("context_summary", "")

    recent_history_str = format_recent_history(state.get("messages", []))

    if connected_mcps:
        mcp_lines = []
        for srv in connected_mcps:
            tool_names = [t.get("name", "") for t in (srv.get("cached_tools") or []) if t.get("name")]
            mcp_lines.append(
                f"  - {srv.get('display_name', srv.get('server_slug'))}: tools={tool_names}"
            )
        connected_mcp_str = "\n".join(mcp_lines)
        mcp_agent_line = (
            "- user_mcp_agent: ONLY when the user explicitly asks to use one of their "
            "connected external integrations (e.g., save/create doc in Google Drive, add event to "
            "Google Calendar, check Drive/Calendar, or use a connected custom MCP tool).\n"
        )
        mcp_rule = (
            "5. EXTERNAL MCP TOOLS: If the user asks to save/export/schedule/query using one of their "
            "CONNECTED USER MCP INTEGRATIONS below, include \"user_mcp_agent\" in selected_agents. "
            "NEVER select user_mcp_agent if the user did not ask to use an external integration.\n"
        )
    else:
        connected_mcp_str = "None connected"
        mcp_agent_line = ""
        mcp_rule = ""

    supervisor_prompt = f"""\
You are the supervisor of a multi-agent travel-planning system.
Analyze the user request and choose ONLY the specialist agents that are strictly needed.

Available agents:
- flight_agent: ONLY when user asks for flights, airlines, routes, airfare, or flight booking advice.
- hotel_agent: ONLY when user asks for hotels, accommodation, resorts, neighborhoods, or places to stay.
- weather_agent: ONLY when user asks for weather, climate, season, forecast, or packing advice.
- budget_agent: ONLY when user asks about budget feasibility, cost estimation, or price limits.
- itinerary_agent: ONLY when user asks for a multi-day trip plan, day-by-day itinerary, schedule, tour, or full travel planning.
{mcp_agent_line}
CONNECTED USER MCP INTEGRATIONS:
{connected_mcp_str}

ACTIVE CONVERSATION CONTEXT:
Summary: {context_summary or "None"}
Recent Turns:
{recent_history_str}

CURRENT ACTIVE TRIP CONSTRAINTS:
{json.dumps(existing_constraints, indent=2)}

CRITICAL ROUTING & DISAMBIGUATION RULES:
1. SPECIFIC INQUIRIES: If the user is only asking for one specific topic (e.g. "hotels near Guntur railway station", "flights from Delhi to Dubai", "weather in Shimla in December"), select ONLY that specific agent (e.g. ["hotel_agent"]). DO NOT include itinerary_agent unless the user explicitly requested an itinerary or schedule!
2. COMPLETE TRIP PLANNING: If the user asks for a complete trip, vacation, holiday, or multi-day itinerary (e.g. "Plan a 3-day trip to Goa with flights and hotel under 20k"), select all relevant specialists plus "itinerary_agent".
3. QUERY DISAMBIGUATION & COREFERENCE RESOLUTION:
   - If the user request contains relative words (e.g. "there", "that hotel", "the second option", "fly back on Sunday", "what about weather?"), resolve them using the recent conversation history and active trip constraints.
   - Output the resolved standalone query in "resolved_query".
4. CONSTRAINT INHERITANCE:
   - Inherit active constraints (destination, origin, duration, budget, travel_style) unless the user explicitly changes or overrides them.
{mcp_rule}

Return strict JSON only using this schema:
{{
  "selected_agents": ["hotel_agent"],
  "resolved_query": "self-contained disambiguated user query",
  "trip_constraints": {{
    "destination": "inherited or newly specified destination",
    "origin": "inherited or newly specified origin",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": "brief explanation of why these agents were selected"
}}

User request:
{query}
"""

    try:
        raw = resilient_llm.invoke(
            [
                SystemMessage(content="You route work to travel specialist agents. Return strict JSON only."),
                HumanMessage(content=supervisor_prompt),
            ]
        )
        parsed = _json_from_llm(str(raw.content))
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]
        # Never run user_mcp_agent unless the user explicitly asked for an external MCP integration
        if not is_mcp_req:
            selected_agents = [name for name in selected_agents if name != "user_mcp_agent"]

        # If no agents were selected, intelligently default based on query keywords
        if not selected_agents:
            q_lower = query.lower()
            if any(k in q_lower for k in ["hotel", "stay", "resort", "lodge", "room"]):
                selected_agents = ["hotel_agent"]
            elif any(k in q_lower for k in ["flight", "airline", "fly", "airfare", "airport"]):
                selected_agents = ["flight_agent"]
            elif any(k in q_lower for k in ["weather", "climate", "temperature", "rain", "forecast"]):
                selected_agents = ["weather_agent"]
            elif any(k in q_lower for k in ["budget", "cost", "afford", "price", "expensive"]):
                selected_agents = ["budget_agent"]
            else:
                selected_agents = ["itinerary_agent"]

        constraints = merge_constraints(existing_constraints, parsed.get("trip_constraints"))

        resolved_query = str(parsed.get("resolved_query") or query).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        logger.warning("[SUPERVISOR] Planning fallback: %s", exc)
        selected_agents = ["itinerary_agent"]
        constraints = existing_constraints or empty_constraints()
        resolved_query = query
        reasoning = "Supervisor parsing fallback: routed to default travel planning."

    elapsed = time.time() - start_time
    agent_names = ", ".join(name.replace("_agent", "") for name in selected_agents)
    logger.info("[SUPERVISOR] OK TRAVEL intent (%.1fs) | Agents: [%s] | Query: '%s'", elapsed, agent_names, resolved_query)

    return {
        "guardrail_allowed": True,
        "query_intent": "travel",
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "resolved_query": resolved_query,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: TravelState) -> dict:
    natural_response = state.get("final_response") or (
        "I'm sorry, but I can't assist with that request. I'm TripMate AI, your dedicated "
        "travel planning companion! Feel free to ask me about planning trips, discovering destinations, "
        "finding flights, hotels, weather, or local activities."
    )
    logger.info("[GUARDRAIL] Delivered natural blocked response to user.")
    return {
        "final_response": natural_response,
        "messages": [AIMessage(content=natural_response)],
    }
