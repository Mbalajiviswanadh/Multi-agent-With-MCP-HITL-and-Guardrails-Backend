"""
src/agents/itinerary.py
-----------------------
Itinerary integrator agent.
Consolidates specialist findings into a draft travel plan ready for
Human-in-the-Loop review.
"""
from __future__ import annotations

import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.agents.itinerary")


def _trim_context(text, max_chars: int = 1200) -> str:
    if not text:
        return ""
    text_str = str(text).strip()
    return text_str[:max_chars] if len(text_str) <= max_chars else text_str[:max_chars] + "\n... [trimmed]"


def itinerary_agent(state: TravelState) -> dict:
    start_time = time.time()
    logger.info("[ITINERARY_AGENT] -> Generating draft itinerary...")
    prompt = f"""\
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
        response = resilient_llm.invoke(
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
