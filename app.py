from pathlib import Path
import traceback
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.agents.graph import run_travel_agent, travel_graph, resume_travel_agent
from src.api.routers.threads import router as threads_router
from src.api.routers.auth import router as auth_router
from src.api.dependencies import get_current_user, get_optional_current_user
from src.memory.profile_store import run_schema_migrations
from src.memory.vector_store import run_vector_schema_migrations
from src.memory.mem0_service import mem0_get_all, mem0_delete_user
from src.core.database import close_pool
import os
import json
import uuid

from langchain_core.messages import HumanMessage

# This is to allow nested event loops for async calls in FastAPI
import nest_asyncio
nest_asyncio.apply()


TRAVEL_AGENTS = [
    "supervisor",
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
    "human_approval",
    "final_agent",
]


# =========================
# Lifespan Management (DB Migrations & Cleanup)
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize long-term and vector memory schemas at startup
    try:
        run_schema_migrations()
        run_vector_schema_migrations()
    except Exception as exc:
        print(f"[STARTUP] Notice: Database schema check: {exc}")
    yield
    # Cleanup connection pool on shutdown
    close_pool()


# =========================
# FastAPI Application
# =========================

app = FastAPI(
    title="Trip Planner AI API",
    description="LangGraph Multi-Agent Travel Planner API with 3-Tier Memory & Guardrails",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# CORS Configuration
# =========================

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(threads_router)
app.include_router(auth_router)


# =========================
# Request Schema
# =========================

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str = "anonymous"

class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""
    user_id: str = "anonymous"


# =========================
# Travel API Endpoint
# =========================

@app.post("/api/travel")
async def travel_planner(
    request_data: TravelRequest,
    current_user: dict | None = Depends(get_optional_current_user),
):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        # Infer user_id from verified JWT token if authenticated, else request_data.user_id
        effective_user_id = (current_user["id"] if current_user else None) or request_data.user_id or "anonymous"

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
            user_id=effective_user_id,
        )

        return JSONResponse(
            content={
                "success": True,
                "user_id": effective_user_id,
                **result,
            }
        )

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )

@app.post("/api/travel/approve")
async def approve_travel_plan(
    request_data: ApprovalRequest,
    current_user: dict | None = Depends(get_optional_current_user),
):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        effective_user_id = (current_user["id"] if current_user else None) or request_data.user_id or "anonymous"

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
            user_id=effective_user_id,
        )

        return JSONResponse(
            content={
                "success": True,
                "user_id": effective_user_id,
                **result,
            }
        )

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )

# =========================
# Health Check Endpoint
# =========================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "intent_routing",
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
            "mem0_memory",
        ],
    }


# =========================
# Mem0 Memory Endpoints
# =========================

@app.get("/api/memory/{user_id}")
async def get_user_memories(
    user_id: str,
    current_user: dict | None = Depends(get_optional_current_user),
):
    """
    Retrieve all Mem0 memories for a user.
    If authenticated, ensures user can only view their own memories.
    """
    effective_user_id = (current_user["id"] if current_user else None) or user_id
    if current_user and current_user["id"] != user_id and user_id != "anonymous":
        raise HTTPException(status_code=403, detail="Forbidden: Cannot view another user's memories.")

    memories = mem0_get_all(effective_user_id)
    return {
        "user_id": effective_user_id,
        "total": len(memories),
        "memories": memories,
    }


@app.delete("/api/memory/{user_id}")
async def delete_user_memories(
    user_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete all Mem0 memories for a user (GDPR right-to-be-forgotten).
    Requires valid JWT authentication.
    """
    if current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot delete another user's memories.")

    success = mem0_delete_user(user_id)
    return {
        "user_id": user_id,
        "deleted": success,
        "message": "All memories deleted." if success else "Memory deletion failed or no memories found.",
    }

# =========================
# History Endpoint  (paginated trips)
# =========================

@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str, page: int = 1, limit: int = 3):
    """
    Fetch paginated trip history for a given thread_id.
    Messages are grouped into individual trips (each HumanMessage starts a trip).
    Returns newest trips first.
    """
    EMPTY = {
        "thread_id": thread_id,
        "found": False,
        "trips": [],
        "total": 0,
        "page": 1,
        "limit": limit,
        "total_pages": 0,
    }

    try:
        config = {"configurable": {"thread_id": thread_id}}

        # get_state may return None or a snapshot with empty values
        try:
            state = travel_graph.get_state(config)
        except Exception:
            return EMPTY

        if state is None:
            return EMPTY

        values = getattr(state, "values", None) or {}
        raw_messages = values.get("messages", [])

        if not raw_messages:
            return EMPTY

        # Pull thread-level metadata stored in checkpoint state
        # (These reflect the most recent turn's values — good enough for history display)
        state_selected_agents  = values.get("selected_agents", [])
        state_trip_constraints = values.get("trip_constraints", {})
        state_reasoning        = values.get("supervisor_reasoning", "")
        state_summary          = values.get("context_summary", "")
        state_resolved_query   = values.get("resolved_query", "")
        state_intent           = values.get("query_intent", "travel")
        state_llm_calls        = values.get("llm_calls", 0)
        constraint_keys        = [
            k for k, v in state_trip_constraints.items()
            if v and k != "special_preferences"
        ]

        # Group messages into trips — each HumanMessage starts a new trip
        trips = []
        current = None

        for msg in raw_messages:
            cls = msg.__class__.__name__
            if cls == "HumanMessage":
                if current:
                    trips.append(current)
                current = {
                    "query": msg.content,
                    "answer": "",
                    "answer_preview": "",
                    # run metadata — populated below from state
                    "selected_agents":      state_selected_agents,
                    "trip_constraints":     state_trip_constraints,
                    "supervisor_reasoning": state_reasoning,
                    "query_intent":         "travel",
                    "llm_calls":            0,
                    "memory_trace":         None,
                }
            elif cls == "AIMessage" and current:
                # Only the long final answer (skip short agent hand-off messages),
                # or capture short chat replies if no answer has been recorded yet.
                if len(msg.content) > 200 or not current["answer"]:
                    current["answer"] = msg.content
                    current["answer_preview"] = (
                        msg.content[:300].rstrip()
                        + ("…" if len(msg.content) > 300 else "")
                    )

        if current:
            trips.append(current)

        # Reconstruct memory_trace for each trip so Chat Memory tab works across thread switches
        total_trips = len(trips)
        for idx, trip in enumerate(trips):
            is_last = (idx == total_trips - 1)
            prior_msg_count = idx * 2
            window_size = min(6, prior_msg_count)
            rolling_triggered = bool(state_summary) and prior_msg_count >= 10
            has_prior_constraints = idx > 0 and bool(constraint_keys)

            resolved_q = state_resolved_query if (is_last and state_resolved_query) else trip["query"]
            rewritten = bool(resolved_q and resolved_q.strip().lower() != trip["query"].strip().lower())
            turn_intent = state_intent if is_last else "travel"

            if is_last:
                trip["query_intent"] = turn_intent
                trip["llm_calls"] = state_llm_calls

            summary_calls = 1 if rolling_triggered else 0
            tier_calls = {
                "sliding_window":      0,
                "rolling_summary":     summary_calls,
                "supervisor_rewriter": 1,
                "constraint_inherit":  0,
            }
            if rewritten:
                tier_calls["supervisor_rewriter_note"] = "1 call — rewriting + routing combined (0 extra)"

            trip["memory_trace"] = {
                "thread_msg_count":          prior_msg_count,
                "sliding_window_size":       window_size,
                "rolling_summary_triggered": rolling_triggered,
                "rolling_summary_preview":   state_summary[:200] if rolling_triggered else "",
                "constraint_keys_inherited": constraint_keys if has_prior_constraints else [],
                "mem0_hits":                 0,
                "vector_hits":               0,
                "has_prior_constraints":     has_prior_constraints,
                "query_was_rewritten":       rewritten,
                "resolved_query":            resolved_q,
                "original_query":            trip["query"],
                "strategy_chosen": (
                    "chat_recall"      if turn_intent == "chat" else
                    "constraint_reuse" if has_prior_constraints and not rewritten else
                    "coreference"      if rewritten else
                    "fresh"
                ),
                "tier_llm_calls":         tier_calls,
                "total_memory_llm_calls": 1 + summary_calls,
            }

        # Newest first
        trips.reverse()

        total       = len(trips)
        limit       = max(1, min(limit, 20))
        page        = max(1, page)
        total_pages = max(1, (total + limit - 1) // limit) if total else 0
        page        = min(page, total_pages) if total_pages else 1

        start     = (page - 1) * limit
        paginated = trips[start : start + limit]

        return {
            "thread_id":   thread_id,
            "found":       total > 0,
            "trips":       paginated,
            "total":       total,
            "page":        page,
            "limit":       limit,
            "total_pages": total_pages,
        }

    except Exception as e:
        print("HISTORY ERROR:", e)
        traceback.print_exc()
        # Return empty instead of crashing with 500
        return EMPTY


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/travel/stream")
def travel_stream(request_data: TravelRequest):
    message = request_data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = request_data.thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    def events():
        try:
            merged = {
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "llm_calls": 0,
            }
            answer = ""
            selected_agents = []

            # Emit initial supervisor start event
            yield _sse("agent_started", {"agent": "supervisor"})

            for stream_mode, chunk in travel_graph.stream(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_query": message,
                    "guardrail_allowed": True,
                    "guardrail_reason": "",
                    "selected_agents": [],
                    "trip_constraints": {},
                    "supervisor_reasoning": "",
                    "flight_results": "",
                    "hotel_results": "",
                    "weather_results": "",
                    "budget_results": "",
                    "itinerary": "",
                    "approval_request": "",
                    "approved": False,
                    "human_feedback": "",
                    "final_response": "",
                    "llm_calls": 0,
                },
                config=config,
                stream_mode=["updates", "messages"],
            ):
                # ── Node-completion events ──
                if stream_mode == "updates":
                    for node, update in chunk.items():
                        # Supervisor finished
                        if node == "supervisor":
                            allowed = update.get("guardrail_allowed", True)
                            selected_agents = update.get("selected_agents", [])
                            yield _sse("agent_done", {"agent": "supervisor"})
                            yield _sse("supervisor_plan", {
                                "allowed": allowed,
                                "selected_agents": selected_agents,
                                "reasoning": update.get("supervisor_reasoning", ""),
                            })
                            if not allowed:
                                answer = update.get("final_response", "")
                                continue
                            if selected_agents:
                                yield _sse("agent_started", {"agent": selected_agents[0]})
                            continue

                        # Guardrail blocked response
                        if node == "guardrail_blocked":
                            answer = update.get("final_response", "")
                            yield _sse("agent_done", {"agent": "guardrail_blocked"})
                            continue

                        # Specialist & planning updates
                        for key in ("flight_results", "hotel_results", "weather_results", "budget_results", "itinerary"):
                            if key in update:
                                merged[key] = update[key]

                        yield _sse("agent_done", {"agent": node})

                        if node == "final_agent":
                            answer = update.get("final_response") or (
                                update.get("messages", [])[-1].content if update.get("messages") else ""
                            )
                        elif node == "itinerary_agent":
                            yield _sse("agent_started", {"agent": "human_approval"})
                        elif node == "human_approval":
                            yield _sse("agent_started", {"agent": "final_agent"})
                        elif selected_agents and node in selected_agents:
                            curr_idx = selected_agents.index(node)
                            if curr_idx + 1 < len(selected_agents):
                                yield _sse("agent_started", {"agent": selected_agents[curr_idx + 1]})

                # ── Token-level streaming from final LLM ──
                elif stream_mode == "messages":
                    msg_chunk, metadata = chunk
                    node_name = metadata.get("langgraph_node", "")
                    if node_name == "final_agent" and hasattr(msg_chunk, "content") and msg_chunk.content:
                        yield _sse("token", {"text": msg_chunk.content})

            yield _sse("result", {
                "thread_id": thread_id,
                "answer": answer,
                "selected_agents": selected_agents,
                **merged,
            })

        except Exception as exc:
            traceback.print_exc()
            yield _sse("error", {"message": f"Error during streaming: {str(exc)}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =========================
# Run Server
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
