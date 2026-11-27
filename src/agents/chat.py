"""
src/agents/chat.py
------------------
Conversational Chat Agent.

Handles all non-travel messages detected by the intent router:
  - Greetings ("Hi", "Hello", "Good morning")
  - Introductions ("My name is Viswanadh", "I'm from Hyderabad")
  - Stated preferences ("I love vegetarian food", "I prefer budget travel")
  - General questions about TripMate ("What can you do?", "How does this work?")

Behaviour:
  1. Generates a warm, context-aware conversational reply using the LLM.
  2. Retrieves relevant past Mem0 memories to personalize the reply.
  3. Automatically adds the conversation turn to Mem0, which:
       - Extracts facts (name, food preference, city, travel style)
       - Self-evolves conflicting facts (e.g., changing dietary preference)
       - Scopes memories by user_id for cross-session recall
  4. NEVER calls any MCP tool, flight API, or specialist travel agent.
"""
from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.memory.mem0_service import mem0_add_async, mem0_get_user_context, mem0_format_for_context
from src.memory.thread_memory import get_sliding_window
from src.memory.vector_store import find_similar_trips, get_recent_trips
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.agents.chat")

_CHAT_SYSTEM = """\
You are TripMate AI, a friendly and helpful travel planning assistant.
Right now the user is having a casual conversation (not requesting a full trip plan yet).

Guidelines:
- Be warm, friendly, and concise.
- If the user introduces themselves, greet them by name.
- If the user mentions travel preferences (food, style, budget, home city),
  acknowledge them positively and let them know you'll remember it for when
  they're ready to plan a trip.
- If the user asks what you can do, explain that you can plan trips, find
  flights, hotels, weather, and budgets — just describe the trip!
- Never invent or recommend specific flights/hotels unless the user explicitly asks.
- If the user asks about their past trips or saved profile/memory (e.g. "what's my last trip plan?"), summarize the details from Known User Context / Saved Past Trip Plans accurately.
- Keep responses SHORT (2-4 sentences for greetings, 4-6 for capability questions).
"""


def chat_agent(state: TravelState) -> dict:
    start_time = time.time()
    query       = state["user_query"]
    user_id     = state.get("user_id", "anonymous")
    thread_id   = state.get("thread_id", "")

    logger.info("[CHAT_AGENT] -> Handling conversation for user=%s", user_id)

    # ── 1. Build account + Mem0 context (reuse pre-loaded state; zero duplicate search) ──
    user_profile = state.get("user_profile") or {}
    context_parts: list[str] = []

    if user_profile.get("name"):
        context_parts.append(f"Account user name: {user_profile['name']}")
    if user_profile.get("home_airport"):
        context_parts.append(f"Home airport: {user_profile['home_airport']}")
    if user_profile.get("preferred_travel_style"):
        context_parts.append(f"Preferred travel style: {user_profile['preferred_travel_style']}")

    mem0_context = user_profile.get("mem0_context", "")
    if not mem0_context and user_id and user_id != "anonymous" and not user_profile:
        # Fallback only if chat_agent was invoked directly without run_travel_agent
        memories = mem0_get_user_context(query, user_id=user_id, top_k=5)
        mem0_context = mem0_format_for_context(memories)

    if mem0_context:
        context_parts.append(mem0_context)

    # On-demand pgvector past-trip retrieval: only query & inject when the router
    # detected the user actually asked about past/previous/last trips (0 wasted tokens on "Hi").
    if state.get("needs_past_trips") and user_id and user_id != "anonymous":
        try:
            from langgraph.config import get_stream_writer
            w = get_stream_writer()
            if w:
                w({"stage": "memory_pgvector", "text": "Recalling past trips from pgvector…"})
        except Exception:
            pass
        recent_trips = get_recent_trips(user_id, limit=3)
        similar_trips = state.get("similar_past_trips") or find_similar_trips(user_id, query, top_k=2)
        seen_summaries: set[str] = set()
        trip_lines: list[str] = []
        for label, bucket in (("Recent", recent_trips), ("Related", similar_trips)):
            for t in bucket:
                dest = t.get("destination", "Unknown")
                summ = str(t.get("summary", "")).strip()
                key = f"{dest}:{summ[:80]}"
                if key in seen_summaries or not summ:
                    continue
                seen_summaries.add(key)
                created = str(t.get("created_at", ""))[:10]
                date_tag = f" ({created})" if created else ""
                trip_lines.append(f"- [{label}] Trip to {dest}{date_tag}: {summ[:450]}")
                if len(trip_lines) >= 4:
                    break
            if len(trip_lines) >= 4:
                break

        if trip_lines:
            context_parts.append("Saved Past Trip Plans (newest first):\n" + "\n".join(trip_lines))

    memory_context = "\n".join(context_parts)

    # ── 2. Build system prompt (inject account + memory context if available) ──
    system_content = _CHAT_SYSTEM
    if state.get("context_summary"):
        system_content += f"\n\nPrevious Conversation Summary:\n{state['context_summary']}"
    if memory_context:
        system_content += f"\n\nKnown User Context:\n{memory_context}"

    # ── 3. Generate conversational reply (using sliding window of thread messages) ──
    history_window = get_sliding_window(state.get("messages", []), query)

    try:
        response = resilient_llm.invoke(
            [
                SystemMessage(content=system_content),
                *history_window,
            ]
        )
        reply = str(response.content).strip()
    except Exception as exc:
        logger.error("[CHAT_AGENT] LLM error: %s", exc)
        reply = (
            "Hi there! I'm TripMate AI, your travel planning assistant. "
            "How can I help you plan your next trip?"
        )

    # ── 4. Store user facts at ACCOUNT level in background (non-blocking) ────
    # Skip pure questions (e.g. "where am I from?") so we don't burn Groq's
    # 8,000 TPM budget on turns that have no new facts to extract.
    if user_id and user_id != "anonymous":
        q_stripped = query.strip()
        q_lower = q_stripped.lower()
        is_pure_question = (
            q_stripped.endswith("?")
            or q_lower.startswith(("what", "where", "who", "when", "how", "do you", "can you", "tell me", "hey do you"))
        ) and not any(
            kw in q_lower
            for kw in ("i live", "i am from", "i'm from", "my name", "i prefer", "i like", "save ", "remember ")
        )

        if not is_pure_question:
            # Include previous user message only if current turn references "it"/"that"
            needs_prev = any(w in q_lower for w in ("save it", "remember it", "that in your memory"))
            if needs_prev and len(history_window) >= 2:
                msgs_to_store = [
                    {"role": "user", "content": str(m.content).strip()[:250]}
                    for m in history_window[-3:]
                    if isinstance(m, HumanMessage) and str(m.content).strip()
                ]
            else:
                msgs_to_store = [{"role": "user", "content": q_stripped[:300]}]

            mem0_add_async(messages=msgs_to_store, user_id=user_id)

    elapsed = time.time() - start_time
    logger.info("[CHAT_AGENT] OK Reply sent (%.1fs) | user=%s", elapsed, user_id)

    return {
        "final_response": reply,
        "selected_agents": [],
        "messages": [AIMessage(content=reply)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
