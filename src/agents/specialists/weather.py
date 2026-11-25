"""
src/agents/specialists/weather.py
---------------------------------
Weather specialist agent.
Extracts destination city and calls Weather & Forecast MCP tools.
"""
from __future__ import annotations

import asyncio
import time
from langchain_core.messages import AIMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from mymcp.mcp_client_main import (
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)

logger = get_logger("TripMateAI.agents.weather")


def weather_agent(state: TravelState) -> dict:
    start_time = time.time()
    city = state.get("trip_constraints", {}).get("destination", "").strip()
    if not city:
        try:
            city = extract_destination(state.get("resolved_query") or state["user_query"])
        except ValueError:
            city = ""

    if not city:
        logger.warning("[WEATHER_AGENT] No destination found in constraints or query.")
        return {
            "weather_results": (
                "No destination city was specified or found in the current conversation. "
                "Please ask the user which city or place they would like the weather forecast for."
            ),
            "messages": [AIMessage(content="Weather information requires a destination.")],
        }

    logger.info("[WEATHER_AGENT] -> Fetching weather for '%s'...", city)

    try:
        weather_data = asyncio.run(weather_mcp_search(city))
        forecast_data = asyncio.run(forecast_mcp_search(city))

        weather_results = f"""\
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
            f"Live weather information for {city} is temporarily unavailable. "
            "Give general seasonal guidance."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
    }
