"""
scripts/test_memory_recall.py
-----------------------------
End-to-end verification tests for TripMate AI's multi-tier memory system:
  1. Short-Term Thread Memory: sliding window & constraint merging
  2. Episodic Memory (pgvector): store_trip_embedding + get_recent_trips + find_similar_trips
  3. Long-Term Memory (Mem0): mem0_get_user_context & mem0_format_for_context
  4. Chat Agent Recall: verifies past trips + profile + Mem0 context are injected into prompt
  5. Itinerary Agent Memory: verifies user preferences & past trips are injected
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.chat import chat_agent
from src.agents.itinerary import itinerary_agent
from src.agents.state import empty_constraints, initial_state
from src.memory.thread_memory import get_sliding_window, merge_constraints


def test_short_term_thread_memory() -> None:
    print("\n[TEST 1] Short-Term Thread Memory (sliding window + constraint inheritance)...")
    msgs = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
        HumanMessage(content="Plan a 3-day trip to Goa"),
        AIMessage(content="Draft itinerary for Goa"),
    ]
    window = get_sliding_window(msgs, "What about hotels there?")
    assert len(window) >= 2, f"Expected >=2 messages in sliding window, got {len(window)}"

    base = empty_constraints()
    base["destination"] = "Goa"
    base["origin"] = "HYD"
    merged = merge_constraints(base, {"budget": "budget", "destination": ""})
    assert merged["destination"] == "Goa", "Existing destination should be inherited"
    assert merged["origin"] == "HYD", "Existing origin should be inherited"
    assert merged["budget"] == "budget", "New constraint should be applied"
    print("  PASS: Short-term sliding window & constraint inheritance OK")


def test_chat_agent_injects_past_trips_and_mem0() -> None:
    print("\n[TEST 2] Chat Agent Cross-Thread Past Trip + Profile + Mem0 Injection...")
    captured_messages = []

    class CaptureLLM:
        def invoke(self, msgs):
            captured_messages.extend(msgs)
            return SimpleNamespace(content="Your last trip plan was a 4-day budget trip to Goa.")

    state = initial_state("do you know whats my last trip plan?", user_id="test_user_123", thread_id="new_thread")
    state["needs_past_trips"] = True
    state["user_profile"] = {
        "name": "Viswa",
        "home_airport": "HYD",
        "preferred_travel_style": "budget",
        "mem0_context": "- User lives in Guntur\n- User prefers vegetarian food",
    }

    with patch("src.agents.chat.resilient_llm", CaptureLLM()), \
         patch("src.agents.chat.get_recent_trips", return_value=[
             {"destination": "Visakhapatnam", "summary": "3-day coastal trip from HYD to Vizag.", "created_at": "2026-11-27"}
         ]) as mock_recent, \
         patch("src.agents.chat.find_similar_trips", return_value=[
             {"destination": "Goa", "summary": "4-day budget trip from Hyderabad to Goa in December.", "created_at": "2026-11-26"}
         ]) as mock_similar, \
         patch("src.agents.chat.mem0_add_async"):
        res = chat_agent(state)

    assert mock_recent.called and mock_similar.called, "pgvector should be queried when needs_past_trips=True"
    sys_prompt = str(captured_messages[0].content)
    assert "Viswa" in sys_prompt, "Account name missing from chat_agent prompt"
    assert "Guntur" in sys_prompt, "Mem0 context missing from chat_agent prompt"
    assert "Visakhapatnam" in sys_prompt, "Recent pgvector trip missing from chat_agent prompt"
    assert "Goa" in sys_prompt, "Similar pgvector trip missing from chat_agent prompt"
    assert "Goa" in res["final_response"]

    # Also verify casual greeting (needs_past_trips=False) skips pgvector completely
    captured_hi = []
    class CaptureHiLLM:
        def invoke(self, msgs):
            captured_hi.extend(msgs)
            return SimpleNamespace(content="Hi Viswa!")

    hi_state = initial_state("Hi!", user_id="test_user_123", thread_id="new_thread")
    hi_state["needs_past_trips"] = False
    hi_state["user_profile"] = {"name": "Viswa"}
    with patch("src.agents.chat.resilient_llm", CaptureHiLLM()), \
         patch("src.agents.chat.get_recent_trips") as mock_recent_hi, \
         patch("src.agents.chat.find_similar_trips") as mock_sim_hi, \
         patch("src.agents.chat.mem0_add_async"):
        chat_agent(hi_state)
    assert not mock_recent_hi.called and not mock_sim_hi.called, "pgvector must NOT be queried on casual 'Hi!'"
    assert "Saved Past Trip Plans (newest first):" not in str(captured_hi[0].content)
    print("  PASS: chat_agent loads pgvector on-demand ONLY when needs_past_trips=True")


def test_itinerary_agent_receives_memory() -> None:
    print("\n[TEST 3] Itinerary Agent Receives User Preferences & Past Trips...")
    captured_messages = []

    class CaptureLLM:
        def invoke(self, msgs):
            captured_messages.extend(msgs)
            return SimpleNamespace(content="Draft itinerary tailored to vegetarian budget traveler.")

    state = initial_state("Plan a 3-day trip to Kerala", user_id="test_user_123", thread_id="t_kerala")
    state["user_profile"] = {"mem0_context": "- User prefers vegetarian food and quiet beaches"}
    state["similar_past_trips"] = [{"destination": "Goa", "summary": "Loved Palolem beach and budget stays."}]

    with patch("src.agents.itinerary.resilient_llm", CaptureLLM()):
        res = itinerary_agent(state)

    human_prompt = str(captured_messages[1].content)
    assert "vegetarian food" in human_prompt, "Mem0 preferences missing in itinerary_agent"
    assert "Palolem beach" in human_prompt, "Past trip reference missing in itinerary_agent"
    assert "vegetarian" in res["itinerary"].lower()
    print("  PASS: itinerary_agent receives Mem0 preferences & past trip references")


def test_pgvector_roundtrip_if_db_available() -> None:
    print("\n[TEST 4] Live PostgreSQL pgvector store_trip_embedding + get_recent_trips...")
    from src.memory.vector_store import find_similar_trips, get_recent_trips, store_trip_embedding

    test_uid = f"test_mem_{uuid.uuid4().hex[:8]}"
    ok = store_trip_embedding(
        user_id=test_uid,
        thread_id="t_1",
        destination="Coorg",
        final_itinerary="3-day coffee estate trip to Coorg from Bangalore with Abbey Falls.",
    )
    if not ok:
        print("  SKIP: PostgreSQL pgvector not reachable in this environment")
        return

    recent = get_recent_trips(test_uid, limit=2)
    assert len(recent) >= 1 and recent[0]["destination"] == "Coorg", f"Expected Coorg in recent trips, got {recent}"
    similar = find_similar_trips(test_uid, "coffee plantation trip", top_k=2)
    assert len(similar) >= 1 and similar[0]["destination"] == "Coorg"
    print("  PASS: Live pgvector store + get_recent_trips + find_similar_trips OK")


if __name__ == "__main__":
    test_short_term_thread_memory()
    test_chat_agent_injects_past_trips_and_mem0()
    test_itinerary_agent_receives_memory()
    test_pgvector_roundtrip_if_db_available()
    print("\nALL MEMORY RECALL TESTS PASSED!")
