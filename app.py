from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import run_travel_agent, travel_graph, _conn, resume_travel_agent
import os
import json
import uuid

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

# This is to allow nested event loops for async calls in FastAPI
import nest_asyncio
nest_asyncio.apply()


TRAVEL_AGENTS = ["flight_agent", "hotel_agent", "itinerary_agent", "final_agent"]



# =========================
# FastAPI Application
# =========================

app = FastAPI(
    title="Trip Planner AI API",
    description="LangGraph Multi-Agent Travel Planner API",
    version="1.0.0"
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


# =========================
# Request Schema
# =========================

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None

class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


# =========================
# Travel API Endpoint
# =========================

@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
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

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
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
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
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
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
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
                }
            elif cls == "AIMessage" and current:
                # Only the long final answer (skip short agent hand-off messages)
                if len(msg.content) > 200:
                    current["answer"] = msg.content
                    current["answer_preview"] = (
                        msg.content[:300].rstrip()
                        + ("…" if len(msg.content) > 300 else "")
                    )

        if current:
            trips.append(current)

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
def travel_stream(request_data: TravelRequest):  # plain `def`: FastAPI runs it in a worker thread
    message = request_data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = request_data.thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    def events():
        try:
            merged = {"flight_results": "", "hotel_results": "", "itinerary": "", "llm_calls": 0}
            answer = ""
            route = "travel"  # default

            # Start with the router
            yield _sse("agent_started", {"agent": "router_agent"})

            for stream_mode, chunk in travel_graph.stream(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_query": message,
                    "route": "",
                    "flight_results": "",
                    "hotel_results": "",
                    "itinerary": "",
                    "llm_calls": 0,
                },
                config=config,
                stream_mode=["updates", "messages"],
            ):
                # ── Node-completion events ──
                if stream_mode == "updates":
                    for node, update in chunk.items():

                        # Router finished — emit route decision
                        if node == "router_agent":
                            route = update.get("route", "travel")
                            yield _sse("agent_done", {"agent": "router_agent"})
                            yield _sse("route", {"route": route})
                            if route == "travel":
                                yield _sse("agent_started", {"agent": TRAVEL_AGENTS[0]})
                            elif route == "chat":
                                yield _sse("agent_started", {"agent": "chat_agent"})
                            continue

                        # Chat agent finished
                        if node == "chat_agent":
                            answer = update["messages"][-1].content
                            yield _sse("agent_done", {"agent": "chat_agent"})
                            continue

                        # Travel pipeline agents
                        if node in TRAVEL_AGENTS:
                            for key in merged:
                                if key in update:
                                    merged[key] = update[key]
                            if node == "final_agent":
                                answer = update["messages"][-1].content

                            yield _sse("agent_done", {"agent": node})
                            nxt = TRAVEL_AGENTS.index(node) + 1
                            if nxt < len(TRAVEL_AGENTS):
                                yield _sse("agent_started", {"agent": TRAVEL_AGENTS[nxt]})

                # ── Token-level streaming from LLM ──
                elif stream_mode == "messages":
                    msg_chunk, metadata = chunk
                    node_name = metadata.get("langgraph_node", "")
                    # Stream tokens from both final_agent and chat_agent
                    if node_name in ("final_agent", "chat_agent") and hasattr(msg_chunk, "content") and msg_chunk.content:
                        yield _sse("token", {"text": msg_chunk.content})

            yield _sse("result", {"thread_id": thread_id, "answer": answer, "route": route, **merged})

        except Exception:
            traceback.print_exc()
            yield _sse("error", {"message": "Something went wrong while processing your travel request."})

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
