"""
src/agents/final.py
-------------------
Final response generator & Human Approval Node.
Resumes after HITL interrupt, incorporates revision feedback, and formats
the final plan with the output guardrail applied.
"""
from __future__ import annotations

import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.guardrails.output_guardrail import check_output
from src.reliability.llm_factory import resilient_llm

logger = get_logger("TripMateAI.agents.final")


def _trim_context(text, max_chars: int = 1200) -> str:
    if not text:
        return ""
    text_str = str(text).strip()
    return text_str[:max_chars] if len(text_str) <= max_chars else text_str[:max_chars] + "\n... [trimmed]"


def human_approval_agent(state: TravelState) -> dict:
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


def final_agent(state: TravelState) -> dict:
    start_time = time.time()
    logger.info("[FINAL_AGENT] -> Polishing final travel plan with user feedback...")

    if state.get("approved", False):
        review_instruction = "The user approved the draft. Preserve its decisions while polishing it."
    else:
        review_instruction = f"""\
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""\
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
        response = resilient_llm.invoke(
            [
                SystemMessage(content="You are a professional AI travel booking assistant."),
                HumanMessage(content=final_prompt),
            ]
        )
        raw_final = response.content

        # Run Output Guardrail to validate pricing disclaimers and safety
        guardrail_result = check_output(
            raw_final,
            flight_results=state.get("flight_results", ""),
            hotel_results=state.get("hotel_results", ""),
        )
        final_response_content = guardrail_result["response"]

        elapsed = time.time() - start_time
        logger.info("[FINAL_AGENT] OK Final travel plan generated (%.1fs)", elapsed)
        msg = AIMessage(content=final_response_content)
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("[FINAL_AGENT] FAIL Synthesis failed after %.1fs: %s", elapsed, exc)
        final_response_content = f"Failed to generate final travel plan: {exc}"
        msg = AIMessage(content=final_response_content)

    return {
        "final_response": final_response_content,
        "messages": [msg],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
