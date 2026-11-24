"""
src/agents/graph.py
-------------------
Assembles the complete LangGraph StateGraph from modular agents,
applies node-level RetryPolicies, wires PostgreSQL checkpointing,
and provides run_travel_agent / resume_travel_agent runners.

Memory Architecture (3 Tiers + Mem0):
  Tier 1 (Short-Term):  LangGraph PostgresSaver — thread states, HITL pausing
  Tier 2 (Long-Term):   Mem0 — self-evolving user fact memory (chat + trip prefs)
  Tier 3 (Episodic):    pgvector trip_memories_vector — semantic trip recall

Routing:
  supervisor → chat_agent     (intent = "chat")    → END  (fast, no MCP tools)
  supervisor → guardrail_blocked (intent="blocked")→ END
  supervisor → flight → hotel → weather → budget
             → itinerary → human_approval → final  → END
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from src.agents.state import TravelState, empty_constraints, initial_state
from src.agents.supervisor import (
    supervisor_agent,
    guardrail_blocked_agent,
    AGENT_ORDER,
)
from src.agents.chat import chat_agent
from src.agents.specialists.flight import flight_agent
from src.agents.specialists.hotel import hotel_agent
from src.agents.specialists.weather import weather_agent
from src.agents.specialists.budget import budget_agent
from src.agents.itinerary import itinerary_agent
from src.agents.final import human_approval_agent, final_agent

from src.core.logging_config import get_logger
from src.memory.checkpointer import get_checkpointer
from src.memory.mem0_service import (
    mem0_search,
    mem0_add,
    mem0_format_for_context,
)
from src.memory.vector_store import store_trip_embedding, find_similar_trips
from src.reliability.retry_policy import MCP_RETRY_POLICY, LLM_RETRY_POLICY

logger = get_logger("TripMateAI.agents.graph")

# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "chat_agent":        "chat_agent",
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent":      "flight_agent",
    "hotel_agent":       "hotel_agent",
    "weather_agent":     "weather_agent",
    "budget_agent":      "budget_agent",
    "itinerary_agent":   "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    """3-way routing: chat | blocked | travel pipeline."""
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    intent = state.get("query_intent", "travel")
    if intent == "chat":
        return "chat_agent"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                return next_agent
        return "itinerary_agent"
    return route


# =========================
# Build Graph
# =========================

def build_travel_graph():
    graph_builder = StateGraph(TravelState)

    # ── Nodes ───────────────────────────────────────────────────────────────
    graph_builder.add_node("supervisor",        supervisor_agent,       retry=LLM_RETRY_POLICY)
    graph_builder.add_node("chat_agent",        chat_agent,             retry=LLM_RETRY_POLICY)
    graph_builder.add_node("guardrail_blocked", guardrail_blocked_agent)
    graph_builder.add_node("flight_agent",      flight_agent,           retry=MCP_RETRY_POLICY)
    graph_builder.add_node("hotel_agent",       hotel_agent,            retry=MCP_RETRY_POLICY)
    graph_builder.add_node("weather_agent",     weather_agent,          retry=MCP_RETRY_POLICY)
    graph_builder.add_node("budget_agent",      budget_agent,           retry=LLM_RETRY_POLICY)
    graph_builder.add_node("itinerary_agent",   itinerary_agent,        retry=LLM_RETRY_POLICY)
    graph_builder.add_node("human_approval",    human_approval_agent)
    graph_builder.add_node("final_agent",       final_agent,            retry=LLM_RETRY_POLICY)

    # ── Entry point & 3-way routing ─────────────────────────────────────────
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

    # Chat & blocked → END immediately (no travel pipeline)
    graph_builder.add_edge("chat_agent",        END)
    graph_builder.add_edge("guardrail_blocked", END)

    # Specialist chain
    graph_builder.add_conditional_edges("flight_agent",  route_after_agent("flight_agent"),  ROUTE_MAP)
    graph_builder.add_conditional_edges("hotel_agent",   route_after_agent("hotel_agent"),   ROUTE_MAP)
    graph_builder.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
    graph_builder.add_conditional_edges("budget_agent",  route_after_agent("budget_agent"),  ROUTE_MAP)

    # Itinerary → HITL → Final → END
    graph_builder.add_edge("itinerary_agent", "human_approval")
    graph_builder.add_edge("human_approval",  "final_agent")
    graph_builder.add_edge("final_agent",     END)

    checkpointer = get_checkpointer()
    return graph_builder.compile(checkpointer=checkpointer)


# Compiled graph singleton
travel_graph = build_travel_graph()


# =========================
# Serialization Helper
# =========================

def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get("itinerary", "")

    return {
        "thread_id":            thread_id,
        "answer":               answer,
        "query_intent":         result.get("query_intent", "travel"),
        "requires_approval":    interrupt_payload is not None,
        "approval_request":     (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results":       result.get("flight_results", ""),
        "hotel_results":        result.get("hotel_results", ""),
        "weather_results":      result.get("weather_results", ""),
        "budget_results":       result.get("budget_results", ""),
        "itinerary":            (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents":      result.get("selected_agents", []),
        "trip_constraints":     result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed":    result.get("guardrail_allowed", True),
        "guardrail_reason":     result.get("guardrail_reason", ""),
        "approved":             result.get("approved"),
        "human_feedback":       result.get("human_feedback", ""),
        "llm_calls":            result.get("llm_calls", 0),
    }


# =========================
# Execution Runners
# =========================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Start a new conversation turn with 3-tier memory + Mem0 support."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    logger.info("=" * 60)
    logger.info("[START] Turn | Thread: %s | User: %s", thread_id, user_id)
    logger.info("[QUERY] \"%s\"", user_input)
    logger.info("-" * 60)
    start_time = time.time()

    # ── Tier 2: Load Mem0 memories (replaces manual SQL profile read) ───────
    user_profile: dict[str, Any] = {}
    mem0_context = ""
    if user_id and user_id != "anonymous":
        memories = mem0_search(user_input, user_id=user_id, top_k=5)
        mem0_context = mem0_format_for_context(memories)
        # Map to legacy user_profile dict for specialist agents that use it
        if memories:
            user_profile["mem0_context"] = mem0_context

    # ── Tier 3: Fetch semantically similar past trips from pgvector ──────────
    similar_trips = (
        find_similar_trips(user_id, user_input, top_k=2)
        if user_id and user_id != "anonymous"
        else []
    )

    state_input = initial_state(user_input, user_id=user_id, thread_id=thread_id)
    state_input["user_profile"]       = user_profile
    state_input["similar_past_trips"] = similar_trips

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = travel_graph.invoke(state_input, config=config)
        elapsed = time.time() - start_time
        serialized = serialize_result(result, thread_id)
        logger.info("-" * 60)
        intent = serialized.get("query_intent", "travel")
        if intent == "chat":
            logger.info("[CHAT] Reply sent (%.1fs) | Thread: %s", elapsed, thread_id)
        elif serialized.get("requires_approval"):
            logger.info("[PAUSED] Awaiting human approval (%.1fs) | Thread: %s", elapsed, thread_id)
        else:
            logger.info("[FINISHED] Completed (%.1fs) | Thread: %s", elapsed, thread_id)
        logger.info("=" * 60)
        return serialized
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FAILED] Thread %s after %.1fs: %s", thread_id, elapsed, exc)
        logger.info("=" * 60)
        raise


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Resume a paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    logger.info("=" * 60)
    logger.info("[RESUME] Thread: %s | Approved: %s", thread_id, approved)
    if feedback:
        logger.info("[FEEDBACK] \"%s\"", feedback)
    logger.info("-" * 60)
    start_time = time.time()

    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = travel_graph.invoke(
            Command(resume={"approved": approved, "feedback": feedback.strip()}),
            config=config,
        )
        elapsed = time.time() - start_time
        serialized = serialize_result(result, thread_id)

        # ── Tier 2: Store trip facts in Mem0 ─────────────────────────────────
        # Mem0 self-evolves: if the user gave feedback ("add veg options"),
        # it stores that as a new user preference, overwriting conflicting old facts.
        if user_id and user_id != "anonymous":
            constraints = result.get("trip_constraints", {})
            memory_messages: list[dict] = []

            # Build a summary of what was planned for memory extraction
            if constraints.get("destination"):
                memory_messages.append({
                    "role": "user",
                    "content": (
                        f"I just planned a trip to {constraints.get('destination')} "
                        f"({constraints.get('duration', 'N/A')}). "
                        f"Budget: {constraints.get('budget', 'N/A')}. "
                        f"Style: {constraints.get('travel_style', 'N/A')}. "
                        f"Preferences: {constraints.get('special_preferences', [])}."
                    ),
                })
            if feedback:
                memory_messages.append({
                    "role": "user",
                    "content": f"Trip feedback: {feedback}",
                })

            if memory_messages:
                mem0_add(memory_messages, user_id=user_id, run_id=thread_id)

        # ── Tier 3: Store finalized itinerary in pgvector ───────────────────
        if serialized.get("final_response") and user_id and user_id != "anonymous":
            destination = (
                result.get("trip_constraints", {}).get("destination")
                or "Unknown Destination"
            )
            store_trip_embedding(
                user_id=user_id,
                thread_id=thread_id,
                destination=destination,
                final_itinerary=serialized["final_response"],
            )

        logger.info("-" * 60)
        logger.info("[COMPLETE] Finished (%.1fs) | Thread: %s", elapsed, thread_id)
        logger.info("=" * 60)
        return serialized

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FAILED] Resume Thread %s after %.1fs: %s", thread_id, elapsed, exc)
        logger.info("=" * 60)
        raise
