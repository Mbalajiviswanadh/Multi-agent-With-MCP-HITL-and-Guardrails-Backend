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
from src.memory.mem0_service import mem0_add, mem0_search, mem0_format_for_context
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
- Keep responses SHORT (2-4 sentences for greetings, 4-6 for capability questions).
"""


def chat_agent(state: TravelState) -> dict:
    start_time = time.time()
    query       = state["user_query"]
    user_id     = state.get("user_id", "anonymous")
    thread_id   = state.get("thread_id", "")

    logger.info("[CHAT_AGENT] -> Handling conversation for user=%s", user_id)

    # ── 1. Retrieve relevant Mem0 memories to personalise the reply ─────────
    memory_context = ""
    if user_id and user_id != "anonymous":
        memories = mem0_search(query, user_id=user_id, top_k=5)
        memory_context = mem0_format_for_context(memories)

    # ── 2. Build system prompt (inject memory context if available) ─────────
    system_content = _CHAT_SYSTEM
    if memory_context:
        system_content += f"\n\n{memory_context}"

    # ── 3. Generate conversational reply ────────────────────────────────────
    try:
        response = resilient_llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=query),
            ]
        )
        reply = str(response.content).strip()
    except Exception as exc:
        logger.error("[CHAT_AGENT] LLM error: %s", exc)
        reply = (
            "Hi there! I'm TripMate AI, your travel planning assistant. "
            "How can I help you plan your next trip?"
        )

    # ── 4. Store this conversation turn in Mem0 (async-safe, non-blocking) ──
    # Mem0 auto-extracts facts (name, preferences, city) from the conversation
    # and self-evolves conflicting memories — no manual JSON parsing needed.
    if user_id and user_id != "anonymous":
        mem0_add(
            messages=[
                {"role": "user",      "content": query},
                {"role": "assistant", "content": reply},
            ],
            user_id=user_id,
            run_id=thread_id or None,
        )

    elapsed = time.time() - start_time
    logger.info("[CHAT_AGENT] OK Reply sent (%.1fs) | user=%s", elapsed, user_id)

    return {
        "final_response": reply,
        "selected_agents": [],
        "messages": [AIMessage(content=reply)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
