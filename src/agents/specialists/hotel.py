"""
src/agents/specialists/hotel.py
-------------------------------
Hotel specialist agent.
Queries Tavily MCP search for top hotels and accommodation guidance.
"""
from __future__ import annotations

import asyncio
import time
from langchain_core.messages import AIMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from mymcp.mcp_client_main import tavily_mcp_search

logger = get_logger("TripMateAI.agents.hotel")


def _trim_context(text, max_chars: int = 1800) -> str:
    if not text:
        return ""
    text_str = str(text).strip()
    return text_str[:max_chars] if len(text_str) <= max_chars else text_str[:max_chars] + "\n... [trimmed]"


def hotel_agent(state: TravelState) -> dict:
    start_time = time.time()
    logger.info("[HOTEL_AGENT] -> Running hotel search via Tavily MCP...")
    effective_query = state.get("resolved_query") or state.get("user_query", "")
    query = f"Best hotels for {effective_query}"

    try:
        raw_results = asyncio.run(tavily_mcp_search(query))
        hotel_results = _trim_context(raw_results, max_chars=1800)
        elapsed = time.time() - start_time
        logger.info("[HOTEL_AGENT] OK Finished (%.1fs)", elapsed)

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[HOTEL_AGENT] FAIL Tavily search error after %.1fs: %s", elapsed, exc)
        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood guidance based on the destination."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
