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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

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
from src.agents.specialists.user_mcp_agent import user_mcp_agent
from src.agents.itinerary import itinerary_agent
from src.agents.final import human_approval_agent, final_agent

from src.core.logging_config import get_logger
from src.mcp_dynamic.store import get_enabled_user_mcps
from src.memory.checkpointer import get_checkpointer
from src.memory.mem0_service import (
    mem0_search,
    mem0_get_user_context,
    mem0_add,
    mem0_format_for_context,
)
from src.memory.profile_store import load_user_profile
from src.memory.thread_memory import finalize_memory_trace, load_thread_memory
from src.memory.vector_store import store_trip_embedding
from src.reliability.llm_factory import resilient_llm
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
    "final_agent":       "final_agent",
    "user_mcp_agent":    "user_mcp_agent",
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
    return selected[0] if selected else "final_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent == "user_mcp_agent":
                continue  # Run user_mcp_agent AFTER final_agent finishes
            if next_agent in selected:
                return next_agent
        if "itinerary_agent" in selected:
            return "itinerary_agent"
        return "final_agent"
    return route


def route_after_final(state: TravelState) -> str:
    """If user_mcp_agent was also selected (e.g. save trip to Google Drive), run it after final_agent."""
    if "user_mcp_agent" in _selected_agents(state):
        return "user_mcp_agent"
    return END


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
    graph_builder.add_node("user_mcp_agent",    user_mcp_agent,         retry=LLM_RETRY_POLICY)

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

    # Itinerary → HITL → Final → (optional user_mcp_agent) → END
    graph_builder.add_edge("itinerary_agent", "human_approval")
    graph_builder.add_edge("human_approval",  "final_agent")
    graph_builder.add_conditional_edges(
        "final_agent",
        route_after_final,
        {"user_mcp_agent": "user_mcp_agent", END: END},
    )
    graph_builder.add_edge("user_mcp_agent", END)

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
        "resolved_query":       result.get("resolved_query", result.get("user_query", "")),
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
        "user_mcp_results":     result.get("user_mcp_results", ""),
        "mcp_trace":            result.get("mcp_trace", {}),
        "needs_past_trips":     bool(result.get("needs_past_trips", False)),
    }


# =========================
# Execution Runners
# =========================

def _prepare_turn(
    user_input: str,
    thread_id: str | None = None,
    user_id: str = "anonymous",
) -> tuple[str, RunnableConfig, TravelState, dict[str, Any]]:
    """Shared turn setup for both sync and streaming execution."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # ── Tier 1 + 2 + 4: Load thread memory strategies from unified module ──
    existing_constraints, existing_summary, memory_trace = load_thread_memory(
        travel_graph, config, thread_id
    )

    # ── Tier 2 (Mem0) + Profile + Connected MCPs: Load in parallel ──
    # (Tier 3 pgvector past-trip search is now lazy/on-demand in chat/itinerary!)
    user_profile: dict[str, Any] = {}
    connected_mcps: list[dict[str, Any]] = []
    if user_id and user_id != "anonymous":
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_profile = executor.submit(load_user_profile, user_id)
            f_mem0 = executor.submit(mem0_get_user_context, user_input, user_id, 5)
            f_mcps = executor.submit(get_enabled_user_mcps, user_id, False)

            try:
                user_profile = f_profile.result() or {}
            except Exception as exc:
                logger.warning("[PROFILE] Failed to load profile for user=%s: %s", user_id, exc)

            try:
                memories = f_mem0.result() or []
                memory_trace["mem0_hits"] = len(memories)
                if memories:
                    user_profile["mem0_context"] = mem0_format_for_context(memories)
            except Exception as exc:
                logger.warning("[MEM0] Context load failed for user=%s: %s", user_id, exc)

            try:
                connected_mcps = f_mcps.result() or []
            except Exception as exc:
                logger.warning("[MCP_STORE] Failed to load connected MCPs for user=%s: %s", user_id, exc)

    state_input = cast(TravelState, initial_state(user_input, user_id=user_id, thread_id=thread_id))
    state_input["user_profile"]       = user_profile
    state_input["similar_past_trips"] = []
    state_input["trip_constraints"]   = existing_constraints
    state_input["context_summary"]    = existing_summary
    state_input["connected_mcp_tools"] = connected_mcps
    state_input["user_mcp_results"]   = ""
    state_input["mcp_trace"]          = {}

    return thread_id, config, state_input, memory_trace


def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    """Start a new conversation turn with 3-tier memory + Mem0 support."""
    logger.info("=" * 60)
    logger.info("[START] Turn | Thread: %s | User: %s", thread_id or "(new)", user_id)
    logger.info("[QUERY] \"%s\"", user_input)
    logger.info("-" * 60)
    start_time = time.time()

    thread_id, config, state_input, _memory_trace = _prepare_turn(user_input, thread_id, user_id)

    try:
        result = travel_graph.invoke(state_input, config=config)
        elapsed = time.time() - start_time
        serialized = finalize_memory_trace(_memory_trace, serialize_result(result, thread_id), user_input)

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


def stream_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    user_id: str = "anonymous",
):
    """Yield SSE event tuples (event_name, data_dict) for live streaming."""
    logger.info("=" * 60)
    logger.info("[STREAM START] Turn | Thread: %s | User: %s", thread_id or "(new)", user_id)
    logger.info("[QUERY] \"%s\"", user_input)
    logger.info("-" * 60)
    start_time = time.time()

    thread_id, config, state_input, _memory_trace = _prepare_turn(user_input, thread_id, user_id)
    selected_agents: list[str] = []
    streamed_nodes = {"chat_agent", "final_agent", "itinerary_agent", "guardrail_blocked"}

    yield ("agent_started", {"agent": "supervisor"})

    try:
        for stream_mode, chunk in travel_graph.stream(
            state_input,
            config=config,
            stream_mode=["updates", "messages", "custom"],
        ):
            if stream_mode == "custom" and isinstance(chunk, dict):
                yield ("live_step", chunk)
                continue
            if stream_mode == "updates" and isinstance(chunk, dict):
                for node, update in chunk.items():
                    if node == "__interrupt__" or not isinstance(update, dict):
                        continue
                    yield ("agent_done", {"agent": node})
                    if node == "supervisor":
                        allowed = update.get("guardrail_allowed", True)
                        intent = update.get("query_intent", "travel")
                        selected_agents = update.get("selected_agents", [])
                        yield ("supervisor_plan", {
                            "allowed": allowed,
                            "intent": intent,
                            "selected_agents": selected_agents,
                            "reasoning": update.get("supervisor_reasoning", ""),
                        })
                        if not allowed:
                            yield ("agent_started", {"agent": "guardrail_blocked"})
                        elif intent == "chat":
                            yield ("agent_started", {"agent": "chat_agent"})
                        elif selected_agents:
                            yield ("agent_started", {"agent": selected_agents[0]})
                    elif node == "itinerary_agent":
                        yield ("agent_started", {"agent": "human_approval"})
                    elif node == "human_approval":
                        yield ("agent_started", {"agent": "final_agent"})
                    elif selected_agents and node in selected_agents:
                        curr_idx = selected_agents.index(node)
                        if curr_idx + 1 < len(selected_agents):
                            yield ("agent_started", {"agent": selected_agents[curr_idx + 1]})
                        else:
                            yield ("agent_started", {"agent": "final_agent"})

            elif stream_mode == "messages" and isinstance(chunk, tuple) and len(chunk) == 2:
                msg_chunk, metadata = chunk
                node_name = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                if (
                    node_name in streamed_nodes
                    and isinstance(msg_chunk, AIMessageChunk)
                    and msg_chunk.content
                ):
                    yield ("token", {"text": msg_chunk.content, "node": node_name})

        # Read final persisted state snapshot (handles both normal completion & HITL __interrupt__)
        snapshot = travel_graph.get_state(config)
        final_values = dict(getattr(snapshot, "values", {}) or {})
        if getattr(snapshot, "tasks", None):
            interrupts = []
            for task in snapshot.tasks:
                for intr in getattr(task, "interrupts", ()) or ():
                    interrupts.append(intr)
            if interrupts:
                final_values["__interrupt__"] = interrupts

        serialized = finalize_memory_trace(_memory_trace, serialize_result(final_values, thread_id), user_input)
        elapsed = time.time() - start_time
        logger.info("[STREAM FINISHED] (%.1fs) | Thread: %s", elapsed, thread_id)
        logger.info("=" * 60)
        yield ("result", serialized)

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[STREAM FAILED] Thread %s after %.1fs: %s", thread_id, elapsed, exc)
        logger.info("=" * 60)
        yield ("error", {"message": str(exc)})


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

            # Build a clean summary of non-empty trip details (skip N/A/empty fields
            # so Mem0 doesn't store "User has no specified preferences" noise)
            if constraints.get("destination"):
                parts = [f"I just planned a trip to {constraints['destination']}"]
                if constraints.get("duration"):
                    parts.append(f"Duration: {constraints['duration']}")
                if constraints.get("budget"):
                    parts.append(f"Budget: {constraints['budget']}")
                if constraints.get("travel_style"):
                    parts.append(f"Style: {constraints['travel_style']}")
                if constraints.get("special_preferences"):
                    parts.append(f"Preferences: {constraints['special_preferences']}")
                memory_messages.append({"role": "user", "content": ". ".join(parts) + "."})
            if feedback:
                memory_messages.append({
                    "role": "user",
                    "content": f"Trip feedback: {feedback}",
                })

            if memory_messages:
                mem0_add(memory_messages, user_id=user_id, run_id=thread_id)

        # ── Tier 3: Store finalized itinerary in pgvector ───────────────────
        final_text = serialized.get("answer") or result.get("final_response", "")
        if final_text and not serialized.get("requires_approval") and user_id and user_id != "anonymous":
            destination = (
                result.get("trip_constraints", {}).get("destination")
                or "Unknown Destination"
            )
            store_trip_embedding(
                user_id=user_id,
                thread_id=thread_id,
                destination=destination,
                final_itinerary=final_text,
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
