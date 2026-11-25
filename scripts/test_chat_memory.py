"""
scripts/test_chat_memory.py
---------------------------
Verifies thread-level memory, coreference resolution, and constraint inheritance:
  Test 1: Casual Chat Recall
    Turn 1: "hey what are the nearst best hotels near guntur railwaystation?"
    Turn 2: "can you tell me what i asked above?"
    -> Checks that Turn 2 correctly recalls Turn 1 question and hotel recommendations.

  Test 2: Ambiguous Follow-Up (Query Disambiguation + Constraint Inheritance)
    Turn 3: "what are the flight options there from Hyderabad?"
    -> Checks that Supervisor resolves "there" -> "Guntur"
    -> Inherits destination="Guntur"
    -> Selects flight_agent
    -> Returns flight guidance for Hyderabad to Guntur / Vijayawada.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import run_travel_agent


def test_thread_memory_and_coreference():
    thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"
    user_id = f"test_user_{uuid.uuid4().hex[:6]}"

    print("\n" + "=" * 65)
    print(f"THREAD MEMORY & QUERY DISAMBIGUATION TEST SUITE")
    print(f"Thread ID : {thread_id}")
    print(f"User ID   : {user_id}")
    print("=" * 65)

    # -------------------------------------------------------------
    # TURN 1: Specific Hotel Inquiry
    # -------------------------------------------------------------
    print("\n[TURN 1] User asks for hotels near Guntur railway station...")
    res1 = run_travel_agent(
        user_input="hey what are the nearst best hotels near guntur railwaystation?",
        thread_id=thread_id,
        user_id=user_id,
    )

    answer1 = res1.get("answer", "")
    agents1 = res1.get("selected_agents", [])
    constraints1 = res1.get("trip_constraints", {})

    print(f"Turn 1 Intent        : {res1.get('query_intent')}")
    print(f"Turn 1 Agents        : {agents1}")
    print(f"Turn 1 Destination   : {constraints1.get('destination')}")
    print(f"Turn 1 Answer Snippet:\n{answer1[:300]}...\n")

    assert "hotel_agent" in agents1, "Turn 1 must call hotel_agent"
    assert "itinerary_agent" not in agents1, "Turn 1 should not call itinerary_agent"
    print("✓ Turn 1 succeeded: Hotels retrieved.")

    # -------------------------------------------------------------
    # TURN 2: Chat Recall ("can you tell me what i asked above?")
    # -------------------------------------------------------------
    print("\n[TURN 2] User asks: 'can you tell me what i asked above?'")
    res2 = run_travel_agent(
        user_input="can you tell me what i asked above?",
        thread_id=thread_id,
        user_id=user_id,
    )

    answer2 = res2.get("answer", "")
    intent2 = res2.get("query_intent")
    print(f"Turn 2 Intent        : {intent2}")
    print(f"Turn 2 Answer:\n{answer2}\n")

    # Check that answer mentions hotels and Guntur or railway station
    answer2_lower = answer2.lower()
    has_guntur = "guntur" in answer2_lower or "hotel" in answer2_lower or "railway" in answer2_lower
    not_blank = "haven't asked" not in answer2_lower and "no question" not in answer2_lower

    print(f"Mentioned prior topic? : {has_guntur}")
    print(f"Avoided 'haven't asked'?: {not_blank}")

    assert has_guntur, f"Turn 2 must recall that Turn 1 asked about hotels / Guntur! Got: {answer2}"
    assert not_blank, "Turn 2 must NOT claim the user hasn't asked a question yet!"
    print("✓ Turn 2 succeeded: Chat agent correctly recalled previous turn from thread history!")

    # -------------------------------------------------------------
    # TURN 3: Ambiguous Follow-Up ("what are the flight options there from Hyderabad?")
    # -------------------------------------------------------------
    print("\n[TURN 3] User asks: 'what are the flight options there from Hyderabad?'")
    res3 = run_travel_agent(
        user_input="what are the flight options there from Hyderabad?",
        thread_id=thread_id,
        user_id=user_id,
    )

    answer3 = res3.get("answer", "")
    agents3 = res3.get("selected_agents", [])
    resolved3 = res3.get("resolved_query", "")
    constraints3 = res3.get("trip_constraints", {})

    print(f"Turn 3 Intent        : {res3.get('query_intent')}")
    print(f"Turn 3 Resolved Query: {resolved3}")
    print(f"Turn 3 Agents        : {agents3}")
    print(f"Turn 3 Origin        : {constraints3.get('origin')}")
    print(f"Turn 3 Destination   : {constraints3.get('destination')}")
    print(f"Turn 3 Answer Snippet:\n{answer3[:350]}...\n")

    assert "flight_agent" in agents3, "Turn 3 must call flight_agent"
    assert "hyderabad" in str(constraints3.get("origin", "")).lower() or "hyd" in str(constraints3.get("origin", "")).lower(), "Origin must be Hyderabad"
    assert "guntur" in str(constraints3.get("destination", "")).lower() or "guntur" in resolved3.lower() or "vijayawada" in resolved3.lower(), f"Destination 'there' must be resolved to Guntur! Got constraints: {constraints3}"
    print("✓ Turn 3 succeeded: Ambiguity resolved ('there' -> Guntur), constraint inherited, and flight search performed!")

    print("\n" + "=" * 65)
    print("ALL THREAD MEMORY & CONVERSATIONAL TESTS PASSED 100%!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    test_thread_memory_and_coreference()
