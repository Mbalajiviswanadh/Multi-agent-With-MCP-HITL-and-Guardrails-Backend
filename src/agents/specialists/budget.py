"""
src/agents/specialists/budget.py
--------------------------------
Budget feasibility agent.
Analyzes cost categories, budget risk areas, and money-saving suggestions.
"""
from __future__ import annotations

import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.agents.budget")


def _trim_context(text, max_chars: int = 1200) -> str:
    if not text:
        return ""
    text_str = str(text).strip()
    return text_str[:max_chars] if len(text_str) <= max_chars else text_str[:max_chars] + "\n... [trimmed]"


def budget_agent(state: TravelState) -> dict:
    start_time = time.time()
    logger.info("[BUDGET_AGENT] -> Analyzing budget feasibility...")
    effective_query = state.get("resolved_query") or state.get("user_query", "")
    prompt = f"""\
Analyze whether this trip is realistic for the user's budget.

User Query:
{effective_query}

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
        response = resilient_llm.invoke(
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
