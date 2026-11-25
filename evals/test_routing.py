"""
evals/test_routing.py
----------------------
Evaluates the supervisor agent's routing decisions against the golden dataset.

Metric: Routing Precision
  = Number of queries where selected_agents matches expected_agents / Total queries

Metric: Guardrail Accuracy
  = Number of queries where guardrail decision matches expected_guardrail / Total queries

This runs WITHOUT actually calling external MCP tools — it only tests the
supervisor's LLM-based planning and guardrail classification.

Usage:
    python -m evals.test_routing
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from src.guardrails.input_guardrail import check_input
from src.reliability.llm_factory import resilient_llm

DATASET_PATH = Path(__file__).parent / "datasets" / "golden_queries.jsonl"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_guardrail_eval(dataset: list[dict]) -> dict:
    """Test guardrail allow/block decisions."""
    correct = 0
    errors = []

    for case in dataset:
        query = case["query"]
        expected = case["expected_guardrail"]  # "allow" or "block"

        result = check_input(query, resilient_llm)
        actual = "allow" if result["allowed"] else "block"

        if actual == expected:
            correct += 1
        else:
            errors.append({
                "query": query,
                "expected": expected,
                "actual": actual,
                "category": case.get("category"),
            })

    accuracy = correct / len(dataset)
    return {"accuracy": accuracy, "correct": correct, "total": len(dataset), "errors": errors}


def run_routing_eval(dataset: list[dict]) -> dict:
    """
    Test supervisor routing by calling _llm_text with the supervisor prompt.
    Only tests queries that are expected to be ALLOWED.
    """
    from src.core.logging_config import get_logger
    logger = get_logger("TripMateAI.evals.routing")

    import json as json_mod
    from langchain_core.messages import HumanMessage, SystemMessage

    AGENT_ORDER = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"]

    allowed_cases = [c for c in dataset if c["expected_guardrail"] == "allow"]
    correct = 0
    errors = []

    SUPERVISOR_PROMPT_TEMPLATE = """\
You are the supervisor of a multi-agent travel-planning system.
Analyze the user request and choose ONLY the specialist agents that are strictly needed.

Available agents:
- flight_agent: ONLY when user asks for flights, airlines, routes, airfare, or flight booking advice.
- hotel_agent: ONLY when user asks for hotels, accommodation, resorts, neighborhoods, or places to stay.
- weather_agent: ONLY when user asks for weather, climate, season, forecast, or packing advice.
- budget_agent: ONLY when user asks about budget feasibility, cost estimation, or price limits.
- itinerary_agent: ONLY when user asks for a multi-day trip plan, day-by-day itinerary, schedule, tour, or full travel planning.

CRITICAL ROUTING RULES:
1. SPECIFIC INQUIRIES: If the user is only asking for one specific topic (e.g., "hotels near Guntur railway station", "flights from Delhi to Dubai", "weather in Shimla in December"), select ONLY that specific agent (e.g. ["hotel_agent"]). DO NOT include itinerary_agent unless the user explicitly requested an itinerary or schedule!
2. COMPLETE TRIP PLANNING: If the user asks for a complete trip, vacation, holiday, or multi-day itinerary (e.g. "Plan a 3-day trip to Goa with flights and hotel under 20k"), select all relevant specialists plus "itinerary_agent".
3. MINIMAL CALLS: Only select agents that have relevant tools to answer the request. Never call flight_agent or hotel_agent if the user never asked for flights or accommodation.

Return strict JSON only:
{{"selected_agents": ["agent1"], "reasoning": "..."}}

User request:
{query}
"""

    for case in allowed_cases:
        query = case["query"]
        expected_agents = set(case["expected_agents"])

        try:
            raw = resilient_llm.invoke([
                SystemMessage(content="You route work to travel specialist agents. Return strict JSON only."),
                HumanMessage(content=SUPERVISOR_PROMPT_TEMPLATE.format(query=query)),
            ])
            parsed = json_mod.loads(raw.content[raw.content.find("{"):raw.content.rfind("}") + 1])
            selected = set(parsed.get("selected_agents", []))

            # Match criteria: expected agents must be selected
            if expected_agents.issubset(selected):
                correct += 1
            else:
                errors.append({
                    "query": query,
                    "expected": sorted(expected_agents),
                    "actual": sorted(selected),
                    "missing": sorted(expected_agents - selected),
                    "extra": sorted(selected - expected_agents),
                })
        except Exception as exc:
            errors.append({"query": query, "error": str(exc)})

    precision = correct / len(allowed_cases) if allowed_cases else 0
    return {"precision": precision, "correct": correct, "total": len(allowed_cases), "errors": errors}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TripMate AI — Routing Evaluation Suite")
    print("=" * 60)

    dataset = load_dataset()
    print(f"\nLoaded {len(dataset)} test cases from golden_queries.jsonl")

    print("\n[1/2] Running guardrail accuracy eval...")
    t0 = time.time()
    guardrail_results = run_guardrail_eval(dataset)
    print(f"      Guardrail Accuracy: {guardrail_results['accuracy']:.1%}  "
          f"({guardrail_results['correct']}/{guardrail_results['total']}) in {time.time()-t0:.1f}s")

    if guardrail_results["errors"]:
        print(f"      Misclassified ({len(guardrail_results['errors'])}):")
        for e in guardrail_results["errors"][:5]:
            print(f"        - [{e['category']}] '{e['query'][:60]}' → expected={e['expected']} got={e['actual']}")

    print("\n[2/2] Running supervisor routing precision eval...")
    t0 = time.time()
    routing_results = run_routing_eval(dataset)
    print(f"      Routing Precision:  {routing_results['precision']:.1%}  "
          f"({routing_results['correct']}/{routing_results['total']}) in {time.time()-t0:.1f}s")

    if routing_results["errors"]:
        print(f"      Wrong routing ({len(routing_results['errors'])}):")
        for e in routing_results["errors"][:5]:
            if "error" in e:
                print(f"        - ERROR: '{e['query'][:60]}' → {e['error']}")
            else:
                print(f"        - '{e['query'][:60]}'")
                print(f"          expected: {e.get('expected')} | actual: {e.get('actual')}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Guardrail Accuracy: {guardrail_results['accuracy']:.1%}")
    print(f"  Routing Precision:  {routing_results['precision']:.1%}")
    print("=" * 60 + "\n")
