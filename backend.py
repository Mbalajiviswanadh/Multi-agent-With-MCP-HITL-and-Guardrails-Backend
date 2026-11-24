import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import logging
import sys
import time
from typing import TypedDict, Annotated, Any
import operator
import uuid
import json

import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from mymcp.mcp_client_main import (
    tavily_mcp_search, 
    aviation_mcp_call, 
    extract_destination, 
    forecast_mcp_search, 
    weather_mcp_search
)
from src.memory.profile_store import ensure_user_exists, upsert_user_profile, load_user_profile
from src.memory.vector_store import store_trip_embedding, find_similar_trips

# =========================
# Structured Logging Setup
# =========================
logger = logging.getLogger("TripMateBackend")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if the file is imported multiple times
if not logger.handlers:
    # Clean, compact format for the console to prevent line-wrapping
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Detailed format for the persistent log file
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_runs.log")
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as _log_exc:
        print(f"Warning: Could not configure file logger: {_log_exc}")


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM with Fallback (openai/gpt-oss-120b -> openai/gpt-oss-20b)
# =========================
try:
    from src.reliability.llm_factory import resilient_llm as llm
except Exception as _llm_init_exc:
    logger.warning("Could not load resilient_llm, falling back to direct ChatGroq: %s", _llm_init_exc)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY
    )


# =========================
# Context Trimming Helper
# =========================

def _trim_context(text: Any, max_chars: int = 1500) -> str:
    """Trim large text or structured data to keep prompt sizes within Groq token limits."""
    if not text:
        return ""
    text_str = str(text).strip()
    if len(text_str) <= max_chars:
        return text_str
    return text_str[:max_chars] + "\n... [trimmed for length]"


# =========================
# State
# =========================

class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]

def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)

def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])

def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    start_time = time.time()
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    logger.info("[SUPERVISOR] -> Validating request with guardrail...")

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        elapsed = time.time() - start_time
        logger.info("[SUPERVISOR] X BLOCKED by guardrail (%.1fs): %s", elapsed, reason)
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing fallback: selected full travel workflow."
        )

    elapsed = time.time() - start_time
    agent_names = ", ".join(name.replace("_agent", "") for name in selected_agents)
    logger.info("[SUPERVISOR] OK Allowed (%.1fs) | Planned agents: [%s]", elapsed, agent_names)

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    logger.info("[GUARDRAIL] Delivered blocked response to user.")
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


# Flight Agent
def flight_agent(state: TravelState):
    start_time = time.time()
    logger.info("[FLIGHT_AGENT] -> Running flight search...")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:1500],
            airline_data=str(airlines)[:1500],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
        elapsed = time.time() - start_time
        logger.info("[FLIGHT_AGENT] OK Finished (%.1fs)", elapsed)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FLIGHT_AGENT] FAIL Failed after %.1fs: %s", elapsed, exc)
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    start_time = time.time()
    logger.info("[HOTEL_AGENT] -> Running hotel search via Tavily MCP...")
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        raw_results = asyncio.run(
            tavily_mcp_search(query)
        )
        hotel_results = _trim_context(raw_results, max_chars=1800)
        elapsed = time.time() - start_time
        logger.info("[HOTEL_AGENT] OK Finished (%.1fs)", elapsed)

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[HOTEL_AGENT] FAIL Tavily search error after %.1fs: %s", elapsed, exc)
        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }

# =========================
# Weather Agent
# =========================

def weather_agent(state: TravelState):
    start_time = time.time()
    city = extract_destination(state["user_query"])
    logger.info("[WEATHER_AGENT] -> Fetching weather for '%s'...", city)

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
        elapsed = time.time() - start_time
        logger.info("[WEATHER_AGENT] OK Finished (%.1fs)", elapsed)

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[WEATHER_AGENT] FAIL Weather tool error after %.1fs: %s", elapsed, exc)

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - specialist
# =========================
def budget_agent(state: TravelState):
    start_time = time.time()
    logger.info("[BUDGET_AGENT] -> Analyzing budget feasibility...")
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{_trim_context(state.get('flight_results', ''), 1200)}

Hotel Results:
{_trim_context(state.get('hotel_results', ''), 1200)}

Weather Results:
{_trim_context(state.get('weather_results', ''), 1000)}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are a practical travel budget analyst."),
                HumanMessage(content=prompt),
            ]
        )
        budget_results = response.content
        elapsed = time.time() - start_time
        logger.info("[BUDGET_AGENT] OK Finished (%.1fs)", elapsed)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[BUDGET_AGENT] FAIL Analysis failed after %.1fs: %s", elapsed, exc)
        budget_results = f"Budget analysis unavailable due to error: {exc}"

    return {
        "budget_results": budget_results,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - integrated plan for review
# =========================
def itinerary_agent(state: TravelState):
    start_time = time.time()
    logger.info("[ITINERARY_AGENT] -> Generating draft itinerary...")
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{_trim_context(state.get('flight_results', ''), 1200)}

Hotel Results:
{_trim_context(state.get('hotel_results', ''), 1200)}

Weather Results:
{_trim_context(state.get('weather_results', ''), 1000)}

Budget Results:
{_trim_context(state.get('budget_results', ''), 1200)}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel planner."),
                HumanMessage(content=prompt),
            ]
        )
        itinerary_content = response.content
        elapsed = time.time() - start_time
        logger.info("[ITINERARY_AGENT] OK Draft itinerary ready (%.1fs)", elapsed)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[ITINERARY_AGENT] FAIL Itinerary generation failed after %.1fs: %s", elapsed, exc)
        itinerary_content = f"Draft itinerary generation failed: {exc}"

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": itinerary_content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    logger.info("[HUMAN_APPROVAL] || Pausing for user review (interrupt)...")
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()
    logger.info("[HUMAN_APPROVAL] OK Resumed | Approved: %s | Feedback: '%s'", approved, human_feedback)

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - trimmed context to prevent token rate limits
# =========================
def final_agent(state: TravelState):
    start_time = time.time()
    logger.info("[FINAL_AGENT] -> Polishing final travel plan with user feedback...")

    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{_trim_context(state.get('flight_results', ''), 1200)}

Hotels:
{_trim_context(state.get('hotel_results', ''), 1200)}

Weather:
{_trim_context(state.get('weather_results', ''), 1000)}

Budget Analysis:
{_trim_context(state.get('budget_results', ''), 1200)}

Draft Itinerary:
{_trim_context(state.get('itinerary', ''), 2000)}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a professional AI travel booking assistant."
                ),
                HumanMessage(content=final_prompt),
            ]
        )
        final_response_content = response.content
        elapsed = time.time() - start_time
        logger.info("[FINAL_AGENT] OK Final travel plan generated (%.1fs)", elapsed)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FINAL_AGENT] FAIL Synthesis failed after %.1fs: %s", elapsed, exc)
        final_response_content = f"Failed to generate final travel plan: {exc}"
        response = AIMessage(content=final_response_content)

    return {
        "final_response": final_response_content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None, user_id: str = "anonymous"):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    logger.info("=" * 60)
    logger.info("[START] Travel Workflow | Thread: %s | User: %s", thread_id, user_id)
    logger.info("[QUERY] \"%s\"", user_input)
    logger.info("-" * 60)
    start_time = time.time()

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = travel_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "trip_constraints": _empty_constraints(),
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
        )
        elapsed = time.time() - start_time
        serialized = _serialize_result(result, thread_id)
        logger.info("-" * 60)
        if serialized.get("requires_approval"):
            logger.info("[PAUSED] Waiting for human approval | Thread: %s (took %.1fs)", thread_id, elapsed)
        else:
            logger.info("[FINISHED] Workflow completed | Thread: %s (took %.1fs)", thread_id, elapsed)
        logger.info("=" * 60)
        return serialized

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FAILED] Workflow error for Thread %s after %.1fs: %s", thread_id, elapsed, exc)
        logger.info("=" * 60)
        raise


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
    user_id: str = "anonymous",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    logger.info("=" * 60)
    logger.info("[RESUME] Resuming Workflow | Thread: %s | User: %s | Approved: %s", thread_id, user_id, approved)
    if feedback:
        logger.info("[FEEDBACK] \"%s\"", feedback)
    logger.info("-" * 60)
    start_time = time.time()

    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = travel_graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback.strip(),
                }
            ),
            config=config,
        )
        elapsed = time.time() - start_time
        serialized = _serialize_result(result, thread_id)

        # Tier 2: Save learned profile from structured trip constraints (generated by LLM)
        if user_id and user_id != "anonymous":
            ensure_user_exists(user_id)
            constraints = result.get("trip_constraints", {})
            profile_update = {}
            if constraints.get("travel_style"):
                profile_update["preferred_travel_style"] = str(constraints["travel_style"])
            if constraints.get("origin"):
                profile_update["home_airport"] = str(constraints["origin"])[:10]
            if feedback:
                profile_update["custom_notes"] = feedback.strip()[:255]
            if profile_update:
                upsert_user_profile(user_id, profile_update)

        # Tier 3: Store finalized trip in pgvector for future semantic recall
        if serialized.get("final_response") and user_id and user_id != "anonymous":
            destination = (
                result.get("trip_constraints", {}).get("destination")
                or "Unknown Destination"
            )
            store_trip_embedding(
                user_id=user_id,
                thread_id=thread_id,
                destination=destination,
                final_itinerary=serialized["final_response"],
            )

        logger.info("-" * 60)
        logger.info("[COMPLETE] Workflow finished successfully! (took %.1fs)", elapsed)
        logger.info("=" * 60)
        return serialized

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FAILED] Resume error for Thread %s after %.1fs: %s", thread_id, elapsed, exc)
        logger.info("=" * 60)
        raise
