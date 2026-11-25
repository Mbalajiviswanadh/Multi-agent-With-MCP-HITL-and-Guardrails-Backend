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
from mymcp.mcp_client_main import extract_destination, weather_mcp_search, forecast_mcp_search
import asyncio

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
    selected_agents = list(state.get("selected_agents", []))
    logger.info("[FINAL_AGENT] -> Generating response for agents: %s", selected_agents)

    user_query = state.get("resolved_query") or state.get("user_query", "")
    feedback_text = state.get("human_feedback", "").lower()
    weather_results = state.get("weather_results", "")
    flight_results = state.get("flight_results", "")
    hotel_results = state.get("hotel_results", "")
    budget_results = state.get("budget_results", "")
    itinerary_draft = state.get("itinerary", "")

    # If the user asked for weather in the revision feedback and weather wasn't originally called:
    if ("weather" in feedback_text or "climate" in feedback_text or "forecast" in feedback_text) and not weather_results:
        try:
            logger.info("[FINAL_AGENT] -> User requested weather in feedback. Triggering live Weather MCP...")
            city = (
                state.get("trip_constraints", {}).get("destination")
                or extract_destination(user_query)
            )
            w_curr = asyncio.run(weather_mcp_search(city))
            w_fc = asyncio.run(forecast_mcp_search(city))
            weather_results = f"Current Weather:\n{w_curr}\n\nForecast:\n{w_fc}"
            if "weather_agent" not in selected_agents:
                selected_agents.append("weather_agent")
            logger.info("[FINAL_AGENT] OK Live weather retrieved for %s", city)
        except Exception as w_exc:
            logger.warning("[FINAL_AGENT] Could not fetch live weather on feedback: %s", w_exc)

    # Determine user approval context
    is_approved = state.get("approved", False)
    human_feedback = state.get("human_feedback", "").strip()

    if is_approved:
        review_context = (
            "The user approved the draft. Present the approved recommendations clearly and cleanly. "
            "Preserve all confirmed choices, prices, and specifics from the draft."
        )
    elif human_feedback:
        review_context = f"The user requested changes: '{human_feedback}'. Carefully incorporate this feedback."
    else:
        review_context = "Provide a direct, helpful, and beautifully formatted response answering the user request."

    # Build agent-specific context sections
    has_itinerary = "itinerary_agent" in selected_agents and bool(itinerary_draft)
    has_hotel = "hotel_agent" in selected_agents or bool(hotel_results)
    has_flight = "flight_agent" in selected_agents or bool(flight_results)
    has_weather = "weather_agent" in selected_agents or bool(weather_results)
    has_budget = "budget_agent" in selected_agents or bool(budget_results)

    sections_instruction = []
    if has_itinerary:
        sections_instruction.append("- Trip Overview / Summary")
        if has_flight:
            sections_instruction.append("- Flight Options & Recommendations")
        if has_hotel:
            sections_instruction.append("- Hotel & Accommodation Options")
        if has_weather:
            sections_instruction.append("- Weather & Packing Advice")
        if has_budget:
            sections_instruction.append("- Budget Breakdown & Estimates")
        sections_instruction.append("- Day-by-Day Itinerary")
        sections_instruction.append("- Practical Travel Tips")
    else:
        # Focused single or specialist responses (NO fake itinerary or unrequested sections!)
        if has_hotel and not has_flight:
            sections_instruction.append("- Top Recommended Hotels (with proximity, key amenities, estimated rates, and standout features)")
            sections_instruction.append("- Location / Neighborhood Highlights")
            sections_instruction.append("- Booking Recommendations & Tips")
        elif has_flight and not has_hotel:
            sections_instruction.append("- Flight Routes & Airlines")
            sections_instruction.append("- Timing, Schedules & Airport Advice")
            sections_instruction.append("- Booking Tips")
        elif has_weather:
            sections_instruction.append("- Current Climate & Forecast Overview")
            sections_instruction.append("- Recommended Clothing & Packing Essentials")
            sections_instruction.append("- Best Seasons to Visit")
        elif has_budget:
            sections_instruction.append("- Cost Breakdown & Feasibility Assessment")
            sections_instruction.append("- Budget Saving Tips")
        else:
            sections_instruction.append("- Direct, detailed answer to the user's specific request")

    formatting_rules = "\n".join(sections_instruction)

    data_sections = []
    if has_hotel:
        data_sections.append(f"### Hotel Search Results:\n{_trim_context(hotel_results, 1500)}")
    if has_flight:
        data_sections.append(f"### Flight Search Results:\n{_trim_context(flight_results, 1500)}")
    if has_weather:
        data_sections.append(f"### Weather Results:\n{_trim_context(weather_results, 1000)}")
    if has_budget:
        data_sections.append(f"### Budget Analysis:\n{_trim_context(budget_results, 1200)}")
    if has_itinerary:
        data_sections.append(f"### Draft Itinerary:\n{_trim_context(itinerary_draft, 2000)}")

    retrieved_data_str = "\n\n".join(data_sections) if data_sections else "No specialist tool output."

    final_prompt = f"""\
You are TripMate AI, a professional and helpful travel assistant.
Generate the response strictly tailored to what the user asked.

USER REQUEST:
{user_query}

REVIEW STATUS:
{review_context}

RETRIEVED SPECIALIST DATA:
{retrieved_data_str}

STRICT FORMATTING REQUIREMENTS:
1. ONLY discuss information that was requested by the user or retrieved in the data above.
2. DO NOT invent or include sections for unrequested categories (e.g., NEVER add a 'Flight Information' section if no flight search was requested or performed).
3. Structure your response cleanly using these relevant sections:
{formatting_rules}
4. Write in a warm, professional, and clear tone with markdown formatting (bullet points, bold highlights).
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
        "weather_results": weather_results,
        "selected_agents": selected_agents,
        "messages": [msg],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
