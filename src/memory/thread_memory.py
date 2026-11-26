"""
src/memory/thread_memory.py
---------------------------
Unified, production-grade **Thread-Level Conversational Memory** strategies.

All 4 strategies live in this single file so the full pipeline is easy to read,
reason about, and tune:

  Strategy 1 — Sliding Window (0 LLM calls)
      Keeps the last N real conversation messages (filtering out internal
      node handoff statuses like "Hotel information processed.") so the LLM
      has immediate thread recall without overflowing the context window.

  Strategy 2 — Rolling Summary (1 LLM call, ONLY when thread >= 10 messages)
      When a thread grows long, older turns beyond the most recent 4 are
      compressed into 2-3 concise bullet points (`context_summary`).

  Strategy 3 — 2-in-1 Supervisor Coreference Resolution (0 EXTRA LLM calls)
      Formatted recent history + active constraints are passed into the
      Supervisor prompt so pronouns ("there", "that place") are resolved
      in the same LLM call that selects specialist agents.

  Strategy 4 — Constraint Inheritance (0 LLM calls)
      Active trip constraints (destination, origin, budget, duration, etc.)
      persist across turns in PostgreSQL checkpoints and are merged with
      any new constraints extracted on the current turn.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.state import empty_constraints
from src.core.logging_config import get_logger
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.memory.thread")

# =========================
# Tunable Thresholds
# =========================

SLIDING_WINDOW_SIZE = 6
ROLLING_SUMMARY_THRESHOLD = 10
ROLLING_SUMMARY_KEEP_RECENT = 4

# Internal LangGraph node handoff messages that should NOT clutter the
# conversational sliding window seen by chat_agent or supervisor_agent.
INTERNAL_HANDOFF_MESSAGES = frozenset({
    "Routing to travel specialists.",
    "Routing to chat agent.",
    "Flight information processed.",
    "Hotel information processed.",
    "Weather information processed.",
    "Weather information requires a destination.",
    "Budget analysis processed.",
    "Draft itinerary created.",
})


# =========================
# Strategy 1: Sliding Window
# =========================

def filter_real_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Strip internal node status messages, keeping only real user/assistant turns."""
    return [
        m for m in messages
        if not (isinstance(m, AIMessage) and str(m.content).strip() in INTERNAL_HANDOFF_MESSAGES)
    ]


def get_sliding_window(
    messages: list[AnyMessage],
    current_query: str,
    window_size: int = SLIDING_WINDOW_SIZE,
) -> list[AnyMessage]:
    """
    Return the last `window_size` real messages for chat_agent, ensuring
    `current_query` is present at the end. (0 LLM calls)
    """
    real = filter_real_messages(messages)
    if real:
        window = list(real[-window_size:])
        if not window or str(window[-1].content).strip() != current_query.strip():
            window.append(HumanMessage(content=current_query))
        return window
    return [HumanMessage(content=current_query)]


def format_recent_history(
    messages: list[AnyMessage],
    max_messages: int = SLIDING_WINDOW_SIZE,
    max_chars: int = 180,
) -> str:
    """
    Format recent real dialogue turns (excluding the current turn) into a compact
    string for the Supervisor's 2-in-1 coreference resolution prompt.
    """
    if not messages:
        return "None (first turn)"

    prior_real = filter_real_messages(messages[:-1])
    if not prior_real:
        return "None (first turn)"

    lines = []
    for msg in prior_real[-max_messages:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        snippet = str(msg.content).strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        lines.append(f"{role}: {snippet}")

    return "\n".join(lines)


# =========================
# Strategy 2: Rolling Summary
# =========================

def generate_rolling_summary(older_messages: list[AnyMessage], current_summary: str) -> str:
    """
    Compress older conversation messages into 2-3 concise bullet points.
    Called ONLY when thread length >= ROLLING_SUMMARY_THRESHOLD. (1 LLM call)
    """
    real_older = filter_real_messages(older_messages)
    if not real_older:
        return current_summary

    convo_text = []
    for m in real_older:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        snippet = str(m.content).strip()
        if len(snippet) > 250:
            snippet = snippet[:250] + "..."
        convo_text.append(f"{role}: {snippet}")

    prompt = (
        "Summarize the key facts, destination, hotel choices, constraints, and "
        "decisions made in this previous conversation into 2-3 concise bullet points.\n"
        f"Previous summary: {current_summary or 'None'}\n\n"
        f"Conversation:\n{chr(10).join(convo_text)}\n\n"
        "Return only the updated 2-3 bullet points."
    )
    try:
        res = resilient_llm.invoke([
            SystemMessage(content="You are a concise conversation summarizer."),
            HumanMessage(content=prompt),
        ])
        return str(res.content).strip()
    except Exception as exc:
        logger.warning("[SUMMARY] Rolling summary failed: %s", exc)
        return current_summary


# =========================
# Strategy 4: Constraint Inheritance
# =========================

def get_inherited_constraint_keys(constraints: dict[str, Any]) -> list[str]:
    """Return non-empty constraint keys inherited from prior turns."""
    return [
        k for k, v in (constraints or {}).items()
        if v and k != "special_preferences"
    ]


def merge_constraints(
    existing: dict[str, Any] | None,
    newly_parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Merge existing thread constraints with newly extracted constraints.
    Existing values are preserved unless explicitly overridden. (0 LLM calls)
    """
    merged = empty_constraints()
    if existing:
        merged.update(existing)
    if isinstance(newly_parsed, dict):
        for k, v in newly_parsed.items():
            if v:
                merged[k] = v
    return merged


# =========================
# Orchestrator Helpers (used by graph.py)
# =========================

def load_thread_memory(
    graph: Any,
    config: RunnableConfig,
    thread_id: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """
    Load Tier 1 (Sliding Window), Tier 2 (Rolling Summary), and Tier 4
    (Constraint Inheritance) from the PostgreSQL checkpoint for a turn.

    Returns:
        (existing_constraints, existing_summary, initial_memory_trace)
    """
    existing_constraints = empty_constraints()
    existing_summary = ""
    thread_msg_count = 0
    rolling_summary_triggered = False
    sliding_window_size = 0
    constraint_keys_inherited: list[str] = []

    try:
        checkpoint_state = graph.get_state(config)
        if checkpoint_state and checkpoint_state.values:
            existing_constraints = checkpoint_state.values.get("trip_constraints") or empty_constraints()
            existing_summary = checkpoint_state.values.get("context_summary") or ""
            existing_messages = checkpoint_state.values.get("messages") or []

            thread_msg_count = len(existing_messages)
            sliding_window_size = min(SLIDING_WINDOW_SIZE, thread_msg_count)
            constraint_keys_inherited = get_inherited_constraint_keys(existing_constraints)

            if thread_msg_count >= ROLLING_SUMMARY_THRESHOLD:
                rolling_summary_triggered = True
                logger.info(
                    "[SUMMARY] Thread %s has %d messages. Generating rolling summary...",
                    thread_id,
                    thread_msg_count,
                )
                existing_summary = generate_rolling_summary(
                    existing_messages[:-ROLLING_SUMMARY_KEEP_RECENT],
                    existing_summary,
                )
    except Exception as cp_exc:
        logger.warning("[CHECKPOINT] Could not read prior thread state: %s", cp_exc)

    tier_llm_calls: dict[str, Any] = {
        "sliding_window":      0,
        "rolling_summary":     1 if rolling_summary_triggered else 0,
        "supervisor_rewriter": 1,
        "constraint_inherit":  0,
    }
    memory_trace: dict[str, Any] = {
        "thread_msg_count":          thread_msg_count,
        "sliding_window_size":       sliding_window_size,
        "rolling_summary_triggered": rolling_summary_triggered,
        "rolling_summary_preview":   existing_summary[:200] if existing_summary else "",
        "constraint_keys_inherited": constraint_keys_inherited,
        "mem0_hits":                 0,
        "vector_hits":               0,
        "has_prior_constraints":     bool(constraint_keys_inherited),
        "tier_llm_calls":            tier_llm_calls,
        "total_memory_llm_calls":    sum(tier_llm_calls.values()),
    }
    return existing_constraints, existing_summary, memory_trace


def finalize_memory_trace(
    memory_trace: dict[str, Any],
    serialized: dict[str, Any],
    user_input: str,
) -> dict[str, Any]:
    """
    Finalize strategy classification and query-rewrite diff on `serialized`
    so the frontend Chat Memory tab can display exact decisions & LLM costs.
    """
    query_intent = serialized.get("query_intent", "travel")
    resolved = serialized.get("resolved_query", user_input)
    query_was_rewritten = bool(resolved and resolved.strip().lower() != user_input.strip().lower())

    memory_trace["query_was_rewritten"] = query_was_rewritten
    memory_trace["resolved_query"] = resolved
    memory_trace["original_query"] = user_input
    memory_trace["strategy_chosen"] = (
        "chat_recall"      if query_intent == "chat" else
        "constraint_reuse" if memory_trace.get("has_prior_constraints") and not query_was_rewritten else
        "coreference"      if query_was_rewritten else
        "fresh"
    )
    if query_was_rewritten:
        memory_trace["tier_llm_calls"]["supervisor_rewriter_note"] = (
            "1 call — rewriting + routing combined (0 extra)"
        )
    serialized["memory_trace"] = memory_trace
    return serialized
