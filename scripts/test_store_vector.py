"""
scripts/test_store_vector.py
----------------------------
Test script to verify direct storing and semantic retrieval of trips
in the Render PostgreSQL `trip_memories_vector` table.

Usage:
    uv run python -m scripts.test_store_vector
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.memory.vector_store import (
    run_vector_schema_migrations,
    store_trip_embedding,
    find_similar_trips,
)
from src.core.database import get_pool, close_pool


def main():
    print("=" * 65)
    print("TripMate AI — Test Storing & Searching in `trip_memories_vector`")
    print("=" * 65)

    test_user_id = "viswanadh_test"
    test_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    test_destination = "Manali, Himachal Pradesh"
    test_itinerary = (
        "Day 1: Arrive in Manali, check into snow-view resort, evening walk at Mall Road. "
        "Day 2: Solang Valley adventure sports, paragliding, zorbing, and ropeway ride. "
        "Day 3: Rohtang Pass snow point excursion and mountain photography. "
        "Day 4: Visit Hadimba Temple, local cafes in Old Manali, and depart."
    )

    print(f"\n[1/3] Ensuring `trip_memories_vector` schema exists...")
    run_vector_schema_migrations()

    print(f"\n[2/3] Storing sample trip into `trip_memories_vector`:")
    print(f"      User ID:     {test_user_id}")
    print(f"      Thread ID:   {test_thread_id}")
    print(f"      Destination: {test_destination}")
    print(f"      Summary:     {test_itinerary[:80]}...")

    success = store_trip_embedding(
        user_id=test_user_id,
        thread_id=test_thread_id,
        destination=test_destination,
        final_itinerary=test_itinerary,
        metadata={
            "trip_type": "mountain_adventure",
            "budget": "₹25,000",
            "duration_days": 4,
        },
    )

    if not success:
        print("\n❌ Failed to store trip embedding in database.")
        sys.exit(1)

    print("      ✅ Successfully inserted row into `trip_memories_vector`!")

    # 3. Retrieve row using Semantic Search
    print(f"\n[3/3] Performing Semantic Vector Search on PostgreSQL:")
    search_query = "Snowy mountain vacation with skiing and paragliding"
    print(f"      Search Query: \"{search_query}\"")

    similar_trips = find_similar_trips(
        user_id=test_user_id,
        query=search_query,
        top_k=3,
    )

    if similar_trips:
        print(f"      ✅ Retrieved {len(similar_trips)} matching trip(s) from pgvector:")
        for idx, trip in enumerate(similar_trips, 1):
            print(f"      [{idx}] Destination: {trip.get('destination')}")
            print(f"          Similarity:  {round(trip.get('similarity', 0.0) * 100, 2)}%")
            print(f"          Summary:     {trip.get('summary', '')[:90]}...")
    else:
        print("      ⚠️ No similar trips returned.")

    # 4. Check total rows in the table
    pool = get_pool()
    with pool.connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM trip_memories_vector;").fetchone()["c"]
        rows = conn.execute(
            """
            SELECT id, user_id, destination, created_at
            FROM trip_memories_vector
            ORDER BY created_at DESC
            LIMIT 3;
            """
        ).fetchall()

    print("\n" + "=" * 65)
    print(f"✅ VERIFICATION COMPLETE! Total rows in `trip_memories_vector`: {count}")
    print("Latest 3 entries in database:")
    for r in rows:
        print(f"  - [{r['created_at'].strftime('%Y-%m-%d %H:%M:%S')}] User: {r['user_id']} | Dest: {r['destination']}")
    print("=" * 65)
    print("\nRun this query in pgAdmin to see it:")
    print("SELECT * FROM public.trip_memories_vector ORDER BY created_at DESC LIMIT 5;")
    close_pool()


if __name__ == "__main__":
    main()
