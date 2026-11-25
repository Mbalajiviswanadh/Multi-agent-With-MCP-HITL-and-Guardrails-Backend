"""
src/agents/specialists/flight.py
--------------------------------
Flight specialist agent.
Queries AviationStack MCP tools (airports, airlines) and uses LLM to generate
flight guidance.
"""
from __future__ import annotations

import asyncio
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.reliability.llm_factory import resilient_llm
from mymcp.mcp_client_main import aviation_mcp_call

logger = get_logger("TripMateAI.agents.flight")

FLIGHT_AGENT_PROMPT = """\
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


def flight_agent(state: TravelState) -> dict:
    start_time = time.time()
    logger.info("[FLIGHT_AGENT] -> Running flight search...")
    query = state.get("resolved_query") or state.get("user_query", "")

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:1500],
            airline_data=str(airlines)[:1500],
        )

        response = resilient_llm.invoke(
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
