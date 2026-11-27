"""
src/memory/vector_store.py
--------------------------
Semantic / Episodic Memory Layer — pgvector-backed storage and retrieval
of past finalized itineraries.

Flow:
  1. After a trip is approved and finalized, `store_trip_embedding()` is
     called to convert the itinerary text to a vector and store it.
  2. When a user starts a new trip, `find_similar_trips()` is called in
     the supervisor to fetch the top-K most semantically similar past trips,
     which are injected into the planning context.

This enables queries like: "Plan something similar to my Bali trip"

Embedding Model:
  Uses FastEmbed (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions).
  - 100% Free, runs locally on CPU via ONNX.
  - Zero API cost, no rate limits, zero network latency.
  - Exactly 384 dimensions matching PostgreSQL `vector(384)`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from typing import Any

from src.core.database import get_pool
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.memory.vector")

# SQL to run once at startup
_VECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trip_memories_vector (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(64),
    thread_id   VARCHAR(64),
    destination VARCHAR(128),
    summary     TEXT,
    embedding   vector(384),
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS trip_memories_hnsw
    ON trip_memories_vector USING hnsw (embedding vector_cosine_ops);
"""


def run_vector_schema_migrations() -> None:
    """Create pgvector table and index if they don't exist."""
    pool = get_pool()
    try:
        with pool.connection() as conn:
            conn.execute(_VECTOR_SCHEMA_SQL)
        logger.info("[VECTOR] pgvector schema ready (trip_memories_vector)")
    except Exception as exc:
        logger.warning(
            "[VECTOR] Could not set up pgvector schema (is 'vector' extension available?): %s", exc
        )


def _get_embedding_model():
    """
    Load the embedding model:
    1. Primary: FastEmbed with 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions, local, 100% free).
    2. Fallback: PurePythonEmbeddings (built-in 384-dimension unit vector).
    """
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        logger.warning("[VECTOR] FastEmbed init notice: %s, falling back to pure Python", exc)
        return _PurePythonEmbeddings(dim=384)


class _PurePythonEmbeddings:
    """
    Built-in, zero-dependency 384-dimension text embedder.
    Uses MD5 hashed token n-grams with L2 normalization to produce
    valid 384-dimension unit vectors for pgvector cosine distance (<=>).
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [round(x / norm, 6) for x in vec]
        return vec


def store_trip_embedding(
    user_id: str,
    thread_id: str,
    destination: str,
    final_itinerary: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Convert a finalized itinerary to a vector embedding and store it in pgvector.
    Returns True on success, False on failure (non-blocking).
    """
    embed_model = _get_embedding_model()
    if embed_model is None:
        return False

    try:
        # Embed a concise summary (destination + first 500 chars of itinerary)
        text_to_embed = f"Trip to {destination}. {final_itinerary[:500]}"
        vector = embed_model.embed_query(text_to_embed)
        # Ensure list of standard floats
        vector = [float(x) for x in vector]

        pool = get_pool()
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO trip_memories_vector
                    (user_id, thread_id, destination, summary, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                """,
                (
                    user_id,
                    thread_id,
                    destination,
                    final_itinerary[:2000],      # Store a 2000-char excerpt as summary
                    str(vector),                  # psycopg vector cast
                    json.dumps(metadata or {}),
                ),
            )
        logger.info(
            "[VECTOR] Stored trip embedding: user=%s destination=%s", user_id, destination
        )
        return True

    except Exception as exc:
        logger.error("[VECTOR] Failed to store trip embedding: %s", exc)
        return False


def find_similar_trips(
    user_id: str,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve the top_k most semantically similar past trips for this user.
    Returns an empty list if the vector table or embed model is unavailable.
    """
    embed_model = _get_embedding_model()
    if embed_model is None:
        return []

    try:
        query_vector = embed_model.embed_query(query)
        query_vector = [float(x) for x in query_vector]

        pool = get_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT destination, summary, metadata, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM trip_memories_vector
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(query_vector), user_id, str(query_vector), top_k),
            ).fetchall()

        results = [dict(row) for row in rows]
        logger.info(
            "[VECTOR] Found %d similar trips for user=%s query='%s'",
            len(results), user_id, query[:60],
        )
        return results

    except Exception as exc:
        logger.error("[VECTOR] Semantic search failed: %s", exc)
        return []


def get_recent_trips(user_id: str, limit: int = 3) -> list[dict[str, Any]]:
    """
    Fetch the user's most recent finalized trips ordered by `created_at DESC`
    so queries like "what is my last trip plan?" always get the latest trip.
    """
    if not user_id or user_id == "anonymous":
        return []
    try:
        pool = get_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT destination, summary, metadata, created_at
                FROM trip_memories_vector
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        results = [dict(row) for row in rows]
        logger.info("[VECTOR] Loaded %d recent trips for user=%s", len(results), user_id)
        return results
    except Exception as exc:
        logger.warning("[VECTOR] Recent trips fetch failed: %s", exc)
        return []
