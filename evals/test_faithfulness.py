"""
evals/test_faithfulness.py
---------------------------
LLM-as-a-judge: evaluates whether the final itinerary hallucinated
facts (airlines, prices, flight numbers) that were NOT present in the
MCP tool output.

This is a Ragas-inspired faithfulness metric:
  Faithfulness Score = statements in response grounded in context / total statements

Usage:
    python -m evals.test_faithfulness
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from src.reliability.llm_factory import resilient_llm

# ---------------------------------------------------------------------------
# Synthetic test cases: (tool_output, final_response, should_pass)
# ---------------------------------------------------------------------------
FAITHFULNESS_CASES = [
    {
        "id": "F001",
        "tool_output": "IndiGo and Air India operate HYD-GOI routes. No direct pricing available.",
        "final_response": "IndiGo and Air India offer flights from Hyderabad to Goa. Prices typically range ₹3,000–₹7,000.",
        "should_pass": True,
        "notes": "Response grounded in tool output with appropriate qualifier",
    },
    {
        "id": "F002",
        "tool_output": "IndiGo and Air India operate HYD-GOI routes. No direct pricing available.",
        "final_response": "SpiceJet offers the cheapest flights at guaranteed ₹2,999 from Hyderabad to Goa.",
        "should_pass": False,
        "notes": "SpiceJet not in tool output + fabricated price guarantee",
    },
    {
        "id": "F003",
        "tool_output": "Hotel Taj Vivanta and Marriott Goa are popular options near Calangute.",
        "final_response": "Top hotels: Taj Vivanta and Marriott Goa near Calangute beach. Great options!",
        "should_pass": True,
        "notes": "Hotel names grounded in tool output",
    },
    {
        "id": "F004",
        "tool_output": "Hotel Taj Vivanta and Marriott Goa are popular options near Calangute.",
        "final_response": "Four Seasons Goa and Leela Resort are the best options at ₹8,000/night.",
        "should_pass": False,
        "notes": "Four Seasons and Leela not in tool output — hallucinated",
    },
    {
        "id": "F005",
        "tool_output": "Weather in December: pleasant, 24–30°C, dry season. Light clothing recommended.",
        "final_response": "December in Goa has wonderful weather (24–30°C). Pack light clothes and sunscreen.",
        "should_pass": True,
        "notes": "Weather facts correctly grounded",
    },
]

_JUDGE_SYSTEM = """You are an expert AI evaluator assessing response faithfulness.
Your task: determine if a final travel response is faithful to the provided tool output context.
- Factual claims about specific airlines, specific hotel names, or strict price guarantees MUST be grounded in the context.
- Common-sense travel packing advice (e.g. sunscreen for sunny weather, umbrella for rain) and appropriately qualified general estimates (e.g. 'typically range ₹3,000–₹7,000') are acceptable natural inferences and are considered faithful.
- Completely fabricated hotel names, unmentioned competitor airlines, or guaranteed fake prices = NOT faithful.
Return strict JSON only: {"faithful": true/false, "reason": "brief explanation"}"""

_JUDGE_PROMPT = """
Tool Output (ground truth):
{tool_output}

Final Response to evaluate:
{final_response}

Is every factual claim in the final response grounded in the tool output?
Return: {{"faithful": true or false, "reason": "..."}}
"""


def evaluate_faithfulness(case: dict) -> dict:
    raw = resilient_llm.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=_JUDGE_PROMPT.format(
            tool_output=case["tool_output"],
            final_response=case["final_response"],
        )),
    ])
    text = str(raw.content)
    parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
    actual_faithful = bool(parsed.get("faithful", True))
    expected_pass = case["should_pass"]
    judge_correct = actual_faithful == expected_pass
    return {
        "id": case["id"],
        "faithful": actual_faithful,
        "expected": expected_pass,
        "judge_correct": judge_correct,
        "reason": parsed.get("reason", ""),
        "notes": case["notes"],
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TripMate AI — Faithfulness Evaluation (LLM-as-a-Judge)")
    print("=" * 60)

    results = [evaluate_faithfulness(c) for c in FAITHFULNESS_CASES]

    judge_accuracy = sum(r["judge_correct"] for r in results) / len(results)
    faithful_pass = sum(r["faithful"] == r["expected"] for r in results)

    for r in results:
        status = "✓ PASS" if r["judge_correct"] else "✗ FAIL"
        print(f"\n  [{r['id']}] {status}")
        print(f"    Expected faithful={r['expected']} | Got faithful={r['faithful']}")
        print(f"    Reason: {r['reason'][:100]}")
        print(f"    Note:   {r['notes']}")

    print("\n" + "=" * 60)
    print(f"  Faithfulness Judge Accuracy: {judge_accuracy:.1%}  ({faithful_pass}/{len(results)})")
    print("=" * 60 + "\n")
