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
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
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

    supervisor_prompt = f"""\
You are the supervisor of a multi-agent travel-planning system.
Analyze the user request and choose ONLY the specialist agents that are strictly needed.

Available agents:
- flight_agent: ONLY when user asks for flights, airlines, routes, airfare, or flight booking advice.
- hotel_agent: ONLY when user asks for hotels, accommodation, resorts, neighborhoods, or places to stay.
- weather_agent: ONLY when user asks for weather, climate, season, forecast, or packing advice.
- budget_agent: ONLY when user asks about budget feasibility, cost estimation, or price limits.
- itinerary_agent: ONLY when user asks for a multi-day trip plan, day-by-day itinerary, schedule, tour, or full travel planning.

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
