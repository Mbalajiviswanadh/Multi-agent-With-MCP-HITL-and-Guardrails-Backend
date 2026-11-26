"""
scripts/test_vector_and_mem0.py
------------------------------
Standalone test for:
  1. pgvector (`trip_memories_vector`) — storing finalized trip embeddings
     and retrieving semantically similar past trips via cosine similarity.
  2. Mem0 + Qdrant Vector DB — extracting user facts/preferences,
     retrieving them across queries, and cleaning up test data.

Run:
  uv run python scripts/test_vector_and_mem0.py
"""
from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import get_pool, close_pool
from src.memory.vector_store import (
    run_vector_schema_migrations,
    store_trip_embedding,
    find_similar_trips,
)
from src.memory.mem0_service import (
    mem0_add,
    mem0_get_user_context,
    mem0_delete_user,
)


def test_pgvector() -> None:
    print("=" * 65)
    print("TEST 1: pgvector (trip_memories_vector)")
    print("=" * 65)
    run_vector_schema_migrations()

    test_user = f"vec_test_{uuid.uuid4().hex[:8]}"
    t1 = f"thread_{uuid.uuid4().hex[:8]}"
    t2 = f"thread_{uuid.uuid4().hex[:8]}"

    print(f"Test User ID: {test_user}\n")

    ok1 = store_trip_embedding(
        user_id=test_user,
        thread_id=t1,
        destination="Goa",
        final_itinerary=(
            "3-day beach vacation in North Goa staying near Baga Beach, "
            "water sports, seafood shacks, sunset cruise, and vibrant nightlife."
        ),
    )
    ok2 = store_trip_embedding(
        user_id=test_user,
        thread_id=t2,
        destination="Shimla",
        final_itinerary=(
            "4-day Himalayan mountain retreat in Shimla and Kufri with snowfall, "
            "Mall Road cafes, toy train ride, and cozy mountain resort."
        ),
    )
    print(f"Stored Goa embedding   : {ok1}")
    print(f"Stored Shimla embedding: {ok2}")
    assert ok1 and ok2, "Failed to store trip embeddings in pgvector"

    query = "Plan something similar to my relaxing beach trip"
    print(f"\nSemantic Query: \"{query}\"")
    matches = find_similar_trips(user_id=test_user, query=query, top_k=2)

    print(f"Matches found: {len(matches)}")
    for i, m in enumerate(matches, 1):
        sim = float(m.get("similarity", 0))
        print(f"  {i}. Destination: {m['destination']:10s} | Similarity: {sim:.4f}")
        print(f"     Summary    : {m['summary'][:100]}...")

    assert len(matches) == 2, "Expected 2 trips back"
    assert matches[0]["destination"] == "Goa", f"Expected Goa first, got {matches[0]['destination']}"
    print("\n✓ pgvector ranked Goa #1 for the beach query!")

    # Cleanup test rows
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM trip_memories_vector WHERE user_id = %s", (test_user,))
    print("✓ Cleaned up test rows from trip_memories_vector.\n")


def test_mem0_qdrant() -> None:
    print("=" * 65)
    print("TEST 2: Mem0 + Qdrant Vector DB")
    print("=" * 65)

    test_user = f"mem0_test_{uuid.uuid4().hex[:8]}"
    print(f"Test User ID: {test_user}\n")

    fact_msg = "Hi, my name is Viswa. My home city is Hyderabad and I prefer vegetarian food and budget hotels."
    print(f"Adding fact to Mem0: \"{fact_msg}\"")
    added = mem0_add(messages=[{"role": "user", "content": fact_msg}], user_id=test_user)
    print(f"Stored in Qdrant: {added}")
    assert added, "Failed to add memory to Mem0/Qdrant"

    search_q = "What food and hotel style does the user like?"
    print(f"\nSearching Qdrant: \"{search_q}\"")
    memories = mem0_get_user_context(query=search_q, user_id=test_user, top_k=5)
    print(f"Memories retrieved: {len(memories)}")
    for i, mem in enumerate(memories, 1):
        print(f"  {i}. {mem.get('memory', mem)}")

    assert len(memories) > 0, "Expected at least 1 memory from Qdrant"
    print("\n✓ Mem0 successfully extracted and retrieved facts from Qdrant!")

    mem0_delete_user(test_user)
    print("✓ Cleaned up test user memories from Qdrant.\n")


if __name__ == "__main__":
    try:
        test_pgvector()
        test_mem0_qdrant()
        print("=" * 65)
        print("ALL PGVECTOR & MEM0/QDRANT TESTS PASSED!")
        print("=" * 65)
    finally:
        close_pool()
