"""
src/api/routers/threads.py
--------------------------
Thread management endpoints — powers the chatbot sidebar.

Endpoints:
  POST   /api/threads              — Create a new chat thread for a user
  GET    /api/threads?user_id=x   — List all threads for a user
  DELETE /api/threads/{thread_id} — Delete a thread and its checkpoints

These endpoints let the frontend display a sidebar of past conversations,
just like ChatGPT / Claude.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.database import get_pool
from src.core.logging_config import get_logger
from src.memory.profile_store import ensure_user_exists

logger = get_logger("TripMateAI.api.threads")
router = APIRouter(prefix="/api/threads", tags=["threads"])


class CreateThreadRequest(BaseModel):
    user_id: str
    title: str = "New Trip"


@router.post("")
def create_thread(body: CreateThreadRequest):
    """Create a new chat session thread for a user."""
    thread_id = f"thread_{uuid.uuid4().hex}"
    ensure_user_exists(body.user_id)
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO threads (id, user_id, title) VALUES (%s, %s, %s)",
            (thread_id, body.user_id, body.title),
        )
    logger.info("[THREADS] Created thread_id=%s for user_id=%s", thread_id, body.user_id)
    return JSONResponse({"thread_id": thread_id, "title": body.title})


@router.get("")
def list_threads(user_id: str = Query(..., description="User ID to fetch threads for")):
    """Return all threads for a user, newest first (for chatbot sidebar)."""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM threads WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    threads = [dict(row) for row in rows]
    logger.info("[THREADS] Listed %d threads for user_id=%s", len(threads), user_id)
    return JSONResponse({"user_id": user_id, "threads": threads})


@router.delete("/{thread_id}")
def delete_thread(thread_id: str, user_id: str = Query(...)):
    """Delete a thread (removes it from sidebar; LangGraph checkpoints remain for audit)."""
    pool = get_pool()
    with pool.connection() as conn:
        result = conn.execute(
            "DELETE FROM threads WHERE id = %s AND user_id = %s",
            (thread_id, user_id),
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Thread not found or not owned by user")
    logger.info("[THREADS] Deleted thread_id=%s for user_id=%s", thread_id, user_id)
    return JSONResponse({"deleted": True, "thread_id": thread_id})
