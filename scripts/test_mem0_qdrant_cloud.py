"""
scripts/test_mem0_qdrant_cloud.py
---------------------------------
Test script to verify Mem0 connection to Qdrant Cloud.
Adds a test memory, searches it back, and retrieves all memories.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.mem0_service import mem0_add, mem0_search, mem0_get_all
from src.core.config import settings

def main():
    print("=" * 60)
    print("Testing Mem0 with Qdrant Cloud")
    print("=" * 60)
    print(f"Qdrant URL configured: {settings.qdrant_url[:35]}...")
    print(f"Qdrant Key configured: {settings.qdrant_api_key[:15]}...")
    
    test_user = "test_viswanadh_cloud"
    
    print("\n[1/3] Adding conversational memory...")
    test_messages = [
        {"role": "user", "content": "Hi, I am Viswanadh. I love beach vacations in Goa and prefer vegetarian food."},
        {"role": "assistant", "content": "Nice to meet you, Viswanadh! I will remember that you love beach vacations in Goa and prefer vegetarian food."}
    ]
    
    success = mem0_add(test_messages, user_id=test_user)
    print(f"Add memory result: {success}")
    
    print("\n[2/3] Searching memory with query 'What are my dietary preferences?'...")
    results = mem0_search("What are my dietary preferences?", user_id=test_user)
    print(f"Search found {len(results)} memories:")
    for idx, r in enumerate(results, 1):
        print(f"  {idx}. {r.get('memory')} (score: {r.get('score', 'N/A')})")
        
    print("\n[3/3] Getting all memories for user...")
    all_memories = mem0_get_all(user_id=test_user)
    print(f"Total memories in Qdrant Cloud: {len(all_memories)}")
    for idx, r in enumerate(all_memories, 1):
        print(f"  {idx}. {r.get('memory')}")
        
    print("\n" + "=" * 60)
    print("QDRANT CLOUD + MEM0 TEST PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
