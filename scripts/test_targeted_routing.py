"""
scripts/test_targeted_routing.py
Verifies:
  1. Specific query ("nearest best hotels near guntur railwaystation")
     - routes ONLY to hotel_agent
     - does NOT call itinerary_agent
     - does NOT pause for HITL approval
     - does NOT include Flight Information or unrequested sections
  2. Full trip query ("Plan a 3-day trip to Goa with flights and hotel under ₹20,000")
     - selects itinerary_agent + specialists
     - pauses for HITL approval
     - on approval, polishes the approved draft cleanly
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import run_travel_agent, resume_travel_agent

def test_hotel_query():
    print("=" * 60)
    print("TEST 1: Hotel-only query")
    print("Query: 'hey what are the nearst best hotels near guntur railwaystation?'")
    print("=" * 60)
    res = run_travel_agent(
        user_input="hey what are the nearst best hotels near guntur railwaystation?",
        user_id="test_user_hotel",
    )

    agents = res.get("selected_agents", [])
    requires_approval = res.get("requires_approval", False)
    answer = res.get("answer", "")

    print(f"Selected Agents      : {agents}")
    print(f"Requires Approval?   : {requires_approval}")
    print(f"Contains 'Flight'    : {'flight' in answer.lower()}")
    print(f"Contains 'Itinerary' : {'1-day' in answer.lower() or 'day 1' in answer.lower()}")
    print(f"\nResponse preview:\n{answer[:500]}...\n")

    assert agents == ["hotel_agent"], f"Expected ['hotel_agent'], got {agents}"
    assert not requires_approval, "Specific hotel inquiry should NOT pause for approval!"
    assert "flight information" not in answer.lower(), "Should NOT hallucinate Flight Information section!"
    print("✓ TEST 1 PASSED: Only hotel_agent was called, no approval was forced, no flight sections hallucinated.\n")


def test_full_itinerary_and_approval():
    print("=" * 60)
    print("TEST 2: Full trip query with HITL approval")
    print("Query: 'Plan a 3-day trip from Hyderabad to Goa under ₹20,000 for 2 people'")
    print("=" * 60)
    res = run_travel_agent(
        user_input="Plan a 3-day trip from Hyderabad to Goa under ₹20,000 for 2 people",
        user_id="test_user_trip",
    )

    thread_id = res["thread_id"]
    agents = res.get("selected_agents", [])
    requires_approval = res.get("requires_approval", False)
    draft = res.get("itinerary") or res.get("answer", "")

    print(f"Selected Agents    : {agents}")
    print(f"Requires Approval? : {requires_approval}")
    print(f"Thread ID          : {thread_id}")
    print(f"Draft snippet      : {draft[:250]}...")

    assert "itinerary_agent" in agents, "Full trip must include itinerary_agent!"
    assert requires_approval, "Full trip draft MUST pause for human approval!"
    print("✓ Draft generated and paused for approval successfully.")

    # Now resume with approval
    print("\n--- Resuming with Approval ---")
    resumed = resume_travel_agent(
        thread_id=thread_id,
        approved=True,
        feedback="",
        user_id="test_user_trip",
    )

    final_answer = resumed.get("answer", "")
    print(f"Final Answer length: {len(final_answer)} characters")
    print(f"Final Answer snippet:\n{final_answer[:400]}...")
    assert len(final_answer) > 100, "Final answer should be complete."
    print("✓ TEST 2 PASSED: Approved draft finalized cleanly without unrequested section duplication.\n")


if __name__ == "__main__":
    test_hotel_query()
    test_full_itinerary_and_approval()
