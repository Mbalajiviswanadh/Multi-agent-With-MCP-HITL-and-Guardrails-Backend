"""
src/memory/mem0_service.py
--------------------------
Mem0 Integration Service for TripMate AI.

Provides intelligent, self-evolving memory management using:
  - LLM:         Groq (llama-3.1-8b-instant) — fact extraction from conversations
  - Embedder:    FastEmbed (sentence-transformers/all-MiniLM-L6-v2) — 384-dim local ONNX
  - Vector Store: Qdrant (local on-disk at ~/.tripmate_mem0/) — zero extra infrastructure

Why Mem0 over raw SQL upserts:
  ┌──────────────────────┬──────────────────────────┬─────────────────────────────────┐
  │ Capability           │ Our Old SQL Approach     │ Mem0                            │
  ├──────────────────────┼──────────────────────────┼─────────────────────────────────┤
  │ Fact extraction      │ Manual JSON prompts      │ Automatic via LLM               │
  │ Conflict resolution  │ SQL COALESCE (additive)  │ Self-evolving (overwrites old)  │
  │ Semantic retrieval   │ pgvector cosine          │ Built-in semantic search        │
  │ Memory scope         │ Fixed SQL columns        │ user_id / agent_id / run_id     │
  │ What it stores       │ Fixed typed fields       │ Any natural-language fact       │
  └──────────────────────┴──────────────────────────┴─────────────────────────────────┘

Memory Scopes:
  - user_id  → Cross-session user preferences ("I prefer vegetarian food")
  - agent_id → "tripmate_ai" (agent identity tag)
  - run_id   → Per-trip thread_id (trip-specific context)

Public API (used by chat.py and graph.py):
  - mem0_add(messages, user_id, run_id)         → Store facts from conversation
  - mem0_search(query, user_id)                 → Retrieve relevant memories
  - mem0_get_all(user_id)                       → List all memories for a user
  - mem0_delete_user(user_id)                   → Remove all memories for a user
  - mem0_format_for_context(memories)           → Format memories as LLM context string
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.memory.mem0")

# Local Qdrant fallback path (used only if QDRANT_URL is not set)
_QDRANT_LOCAL_PATH = str(Path.home() / ".tripmate_mem0" / "qdrant")


def _build_mem0_config() -> dict:
    """
    Build the Mem0 configuration using:
    - Groq LLM (llama-3.1-8b-instant) — lightweight model for fact extraction
    - FastEmbed (sentence-transformers/all-MiniLM-L6-v2) — 384-dim local embeddings
    - Qdrant Cloud (Managed Cloud Vector DB) — if QDRANT_URL & QDRANT_API_KEY exist, else fallback to local disk
    """
    vector_store_config: dict[str, Any] = {
        "collection_name": "tripmate_memories",
        "embedding_model_dims": 384,
    }

    if settings.qdrant_url and settings.qdrant_api_key:
        vector_store_config["url"] = settings.qdrant_url
        vector_store_config["api_key"] = settings.qdrant_api_key
    else:
        vector_store_config["path"] = _QDRANT_LOCAL_PATH

    return {
        "llm": {
            "provider": "groq",
            "config": {
                "model": "openai/gpt-oss-20b",      # Fast, high TPM model available on your Groq key
                "api_key": settings.groq_api_key,
                "temperature": 0.1,                  # Low temp for factual extraction
                "max_tokens": 256,                   # Keep under Groq 8k TPM reservation limit
            },
        },
        "embedder": {
            "provider": "fastembed",                 # Local ONNX via fastembed — no API cost
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_store_config,
        },
    }


# ─── Singleton Initialization ──────────────────────────────────────────────────

_mem0_instance = None
_mem0_failed = False


def _get_mem0():
    """
    Lazy singleton: initializes Mem0 on first use.
    Returns the Memory instance, or None if initialization failed.
    """
    global _mem0_instance, _mem0_failed

    if _mem0_failed:
        return None

    if _mem0_instance is not None:
        return _mem0_instance

    try:
        from mem0 import Memory

        config = _build_mem0_config()
        _mem0_instance = Memory.from_config(config)
        dest = settings.qdrant_url if settings.qdrant_url else _QDRANT_LOCAL_PATH
        logger.info(
            "[MEM0] Initialized | LLM: Groq/openai/gpt-oss-20b | "
            "Embedder: FastEmbed/all-MiniLM-L6-v2 (384-dim) | "
            "Store: Qdrant Cloud @ %s",
            dest,
        )
        return _mem0_instance

    except Exception as exc:
        _mem0_failed = True
        logger.warning(
            "[MEM0] Initialization failed — memory features degraded: %s", exc
        )
        return None


# ─── Public API ────────────────────────────────────────────────────────────────

def mem0_add(
    messages: list[dict[str, str]] | str,
    user_id: str,
    run_id: str | None = None,
) -> bool:
    """
    Add memories from a conversation turn.

    Mem0 automatically extracts facts from the messages — you don't need
    to write extraction prompts. It also auto-resolves conflicts with
    existing memories (e.g., if the user changes their dietary preference).

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
                  OR a plain string (auto-wrapped as user message).
        user_id:  User identifier for memory scoping.
        run_id:   Optional thread/trip ID for run-level scoping.

    Returns:
        True if memories were stored successfully, False on failure.
    """
    mem0 = _get_mem0()
    if mem0 is None:
        return False

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    kwargs: dict[str, Any] = {
        "user_id": user_id,
        "agent_id": "tripmate_ai",
        "infer": True,                  # Let Mem0 auto-extract facts from text
    }
    if run_id:
        kwargs["run_id"] = run_id

    try:
        result = mem0.add(messages, **kwargs)
        # Mem0 returns a dict with "results" list of added/updated memories
        added = result.get("results", []) if isinstance(result, dict) else []
        if added:
            logger.info(
                "[MEM0] Stored %d memory fact(s) for user=%s",
                len(added), user_id,
            )
            for m in added:
                action = m.get("event", "ADD")
                content = m.get("memory", "")[:80]
                logger.debug("[MEM0]   [%s] %s", action, content)
        return True

    except Exception as exc:
        logger.error("[MEM0] Failed to add memories for user=%s: %s", user_id, exc)
        return False


def mem0_add_async(
    messages: list[dict[str, str]] | str,
    user_id: str,
    run_id: str | None = None,
) -> None:
    """
    Non-blocking wrapper around mem0_add() for low-latency chat responses.
    Runs fact extraction and Qdrant upsert in a background daemon thread.
    """
    thread = threading.Thread(
        target=mem0_add,
        kwargs={"messages": messages, "user_id": user_id, "run_id": run_id},
        daemon=True,
    )
    thread.start()


def mem0_get_user_context(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Fast account-level memory retrieval:
      1. Runs semantic search for `query` scoped to `user_id` + `agent_id`.
      2. If semantic search returns nothing (e.g. short/typo/general query like
         'hi' or 'do you know myname?'), falls back to mem0_get_all(user_id)
         so core account facts (name, preferences) are always available.
    """
    memories = mem0_search(query, user_id=user_id, top_k=top_k)
    if memories:
        return memories
    all_memories = mem0_get_all(user_id)
    return all_memories[:top_k] if all_memories else []


def mem0_search(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Semantic search over stored memories for a user.

    Returns the top_k most relevant memories as a list of dicts:
    [{"id": ..., "memory": "...", "score": 0.85, "created_at": ...}, ...]

    Returns empty list on failure (non-blocking).
    """
    mem0 = _get_mem0()
    if mem0 is None:
        return []

    try:
        results = mem0.search(
            query,
            filters={"user_id": user_id, "agent_id": "tripmate_ai"},
            limit=top_k,
        )
        memories = results.get("results", []) if isinstance(results, dict) else (results or [])
        logger.info(
            "[MEM0] Found %d relevant memories for user=%s query='%s'",
            len(memories), user_id, query[:60],
        )
        return memories

    except Exception as exc:
        logger.error("[MEM0] Search failed for user=%s: %s", user_id, exc)
        return []


def mem0_get_all(user_id: str) -> list[dict[str, Any]]:
    """
    Retrieve all stored memories for a user (for the /api/memory/{user_id} endpoint).

    Returns list of memory dicts. Returns empty list on failure.
    """
    mem0 = _get_mem0()
    if mem0 is None:
        return []

    try:
        results = mem0.get_all(
            filters={"user_id": user_id, "agent_id": "tripmate_ai"},
        )
        memories = results.get("results", []) if isinstance(results, dict) else (results or [])
        logger.info("[MEM0] Retrieved %d total memories for user=%s", len(memories), user_id)
        return memories

    except Exception as exc:
        logger.error("[MEM0] get_all failed for user=%s: %s", user_id, exc)
        return []


def mem0_delete_user(user_id: str) -> bool:
    """
    Delete ALL memories for a specific user.
    Used for GDPR-style right-to-be-forgotten requests.
    """
    mem0 = _get_mem0()
    if mem0 is None:
        return False

    try:
        # Mem0 delete_all accepts user_id as a direct kwarg (not via filters)
        mem0.delete_all(user_id=user_id)
        logger.info("[MEM0] Deleted all memories for user=%s", user_id)
        return True

    except Exception as exc:
        logger.error("[MEM0] delete_all failed for user=%s: %s", user_id, exc)
        return False


def mem0_format_for_context(memories: list[dict[str, Any]]) -> str:
    """
    Format a list of Mem0 memory dicts into a clean LLM context string.

    Example output:
        User Memory Context:
        - Prefers vegetarian food
        - Likes budget travel
        - Home city: Hyderabad
        - Previously visited Goa (loved beach activities)
    """
    if not memories:
        return ""

    lines = ["User Memory Context (learned from past conversations):"]
    for m in memories:
        content = m.get("memory", "").strip()
        if content:
            lines.append(f"  - {content}")

    return "\n".join(lines) if len(lines) > 1 else ""
