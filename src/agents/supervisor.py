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
        reason = guardrail_reason or (
            "I'm sorry, I can't help with that. I'm TripMate AI, "
            "your travel planning assistant. Ask me about trips, flights, hotels, or destinations!"
        )
        elapsed = time.time() - start_time
        logger.info("[SUPERVISOR] X BLOCKED (%.1fs): %s", elapsed, reason)
        return {
            "guardrail_allowed": False,
            "query_intent": "blocked",
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=reason)],
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
    # Plan specialist agents & extract trip constraints
    supervisor_prompt = f"""\
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
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

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        logger.warning("[SUPERVISOR] Planning fallback: %s", exc)
        selected_agents = AGENT_ORDER.copy()
        constraints = empty_constraints()
        reasoning = "Supervisor parsing fallback: selected full travel workflow."

    elapsed = time.time() - start_time
    agent_names = ", ".join(name.replace("_agent", "") for name in selected_agents)
    logger.info("[SUPERVISOR] OK TRAVEL intent (%.1fs) | Agents: [%s]", elapsed, agent_names)

    return {
        "guardrail_allowed": True,
        "query_intent": "travel",
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


def guardrail_blocked_agent(state: TravelState) -> dict:
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "I'm sorry, I can't help with that. I'm TripMate AI — ask me about trips, flights, hotels, or destinations!"
    )
    logger.info("[GUARDRAIL] Delivered blocked response to user.")
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }
