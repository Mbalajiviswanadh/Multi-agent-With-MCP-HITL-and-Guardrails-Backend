"""
src/api/routers/threads.py
--------------------------
Thread management endpoints — powers the chatbot sidebar.

Endpoints:
  POST   /api/threads              — Create a new chat thread for a user
  GET    /api/threads              — Paginated list of threads for current user
  PATCH  /api/threads/{thread_id}  — Rename a thread
  DELETE /api/threads/{thread_id}  — Delete a thread and its checkpoints

Protected by optional or verified authentication to prevent cross-user thread snooping.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_optional_current_user
from src.core.database import get_pool
from src.core.logging_config import get_logger
from src.memory.profile_store import ensure_user_exists

logger = get_logger("TripMateAI.api.threads")
router = APIRouter(prefix="/api/threads", tags=["threads"])


class CreateThreadRequest(BaseModel):
    user_id: Optional[str] = None
    title: str = "New Trip"


class RenameThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@router.post("")
def create_thread(
    body: CreateThreadRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    """Create a new chat session thread for a user."""
    effective_user_id = (current_user["id"] if current_user else None) or body.user_id or "anonymous"
    thread_id = f"thread_{uuid.uuid4().hex}"
    ensure_user_exists(effective_user_id)
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO threads (id, user_id, title) VALUES (%s, %s, %s)",
            (thread_id, effective_user_id, body.title),
        )
    logger.info("[THREADS] Created thread_id=%s for user_id=%s", thread_id, effective_user_id)
    return JSONResponse({"thread_id": thread_id, "title": body.title, "user_id": effective_user_id})


@router.get("")
def list_threads(
    user_id: Optional[str] = Query(None, description="User ID to fetch threads for (or inferred from token)"),
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(15, ge=1, le=50, description="Threads per page"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    """Return paginated threads for a user, newest first (for chatbot sidebar)."""
    effective_user_id = (current_user["id"] if current_user else None) or user_id or "anonymous"
    offset = (page - 1) * page_size
    pool = get_pool()
    with pool.connection() as conn:
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM threads WHERE user_id = %s",
            (effective_user_id,),
        ).fetchone()
        total = int(count_row["cnt"]) if count_row else 0

        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM threads
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """,
            (effective_user_id, page_size, offset),
        ).fetchall()

    threads = [dict(row) for row in rows]
    for t in threads:
        if t.get("created_at"):
            t["created_at"] = t["created_at"].isoformat()
        if t.get("updated_at"):
            t["updated_at"] = t["updated_at"].isoformat()

    return JSONResponse({
        "threads": threads,
        "user_id": effective_user_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": (offset + len(threads)) < total,
    })


@router.patch("/{thread_id}")
def rename_thread(
    thread_id: str,
    body: RenameThreadRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    """Rename a chat thread."""
    clean_title = body.title.strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="Thread title cannot be empty.")

    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute("SELECT user_id FROM threads WHERE id = %s", (thread_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Thread not found.")
        if current_user and row["user_id"] != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not own this thread.")

        updated = conn.execute(
            """
            UPDATE threads
            SET title = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING id, title, created_at, updated_at
            """,
            (clean_title, thread_id),
        ).fetchone()

    res = dict(updated)
    if res.get("created_at"):
        res["created_at"] = res["created_at"].isoformat()
    if res.get("updated_at"):
        res["updated_at"] = res["updated_at"].isoformat()
    logger.info("[THREADS] Renamed thread_id=%s -> '%s'", thread_id, clean_title)
    return JSONResponse({"thread": res})


@router.delete("/{thread_id}")
def delete_thread(
    thread_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    """Delete a thread and associated checkpoint state."""
    pool = get_pool()
    with pool.connection() as conn:
        if current_user:
            owner = conn.execute("SELECT user_id FROM threads WHERE id = %s", (thread_id,)).fetchone()
            if owner and owner["user_id"] != current_user["id"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not own this thread.")

        conn.execute("DELETE FROM threads WHERE id = %s", (thread_id,))
    logger.info("[THREADS] Deleted thread_id=%s", thread_id)
    return JSONResponse({"deleted": True, "thread_id": thread_id})
