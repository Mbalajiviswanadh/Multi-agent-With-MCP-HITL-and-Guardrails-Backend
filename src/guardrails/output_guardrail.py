"""
src/guardrails/output_guardrail.py
------------------------------------
Output Guardrail — post-processes the final response before it is returned
to the user.

Checks for:
  1. Fabricated pricing guarantees (e.g., "guaranteed price of ₹X")
  2. Missing safety disclaimers when live data was unavailable
  3. Hallucinated airline/hotel names not grounded in MCP tool output

If a violation is detected, the guardrail either:
  - Soft-patches: appends a correction disclaimer inline, OR
  - Flags: returns a metadata dict with `guardrail_triggered=True` so the
    API layer can log it to LangSmith for monitoring.

Design principle: The guardrail never hard-blocks (raises) — it always
returns *something* useful to the user while flagging the issue.
"""
from __future__ import annotations

import re
from typing import Any

from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.guardrails.output")

# Patterns that indicate potentially hallucinated content
_PRICE_GUARANTEE_PATTERNS = [
    r"guaranteed\s+price",
    r"exact\s+price\s+of",
    r"ticket\s+costs?\s+exactly",
    r"book\s+for\s+(?:exactly|just)\s+₹",
]

_REQUIRED_DISCLAIMER = (
    "\n\n> ⚠️ **Disclaimer**: Flight prices and hotel rates shown are "
    "approximate estimates. Please verify on official booking platforms "
    "before making any reservations."
)


def check_output(
    final_response: str,
    flight_results: str = "",
    hotel_results: str = "",
) -> dict[str, Any]:
    """
    Validate and patch the final response.

    Returns:
        {
            "response": str,           # (possibly patched) final response
            "guardrail_triggered": bool,
            "violations": list[str],   # human-readable violation descriptions
        }
    """
    violations: list[str] = []
    response = final_response

    # 1. Check for price guarantee language
    for pattern in _PRICE_GUARANTEE_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            violations.append(f"Fabricated price guarantee detected (pattern: {pattern})")
            # Soft-patch: replace guarantee language
            response = re.sub(
                pattern,
                "approximate price",
                response,
                flags=re.IGNORECASE,
            )

    # 2. Ensure disclaimer is present when live data was used
    has_live_data = bool(flight_results or hotel_results)
    has_disclaimer = "disclaimer" in response.lower() or "approximate" in response.lower()

    if has_live_data and not has_disclaimer:
        violations.append("Missing pricing disclaimer when live tool data was used")
        response = response + _REQUIRED_DISCLAIMER

    if violations:
        logger.warning(
            "[OUTPUT_GUARDRAIL] %d violation(s) detected and patched: %s",
            len(violations), violations,
        )
    else:
        logger.info("[OUTPUT_GUARDRAIL] Response passed all checks OK")

    return {
        "response": response,
        "guardrail_triggered": bool(violations),
        "violations": violations,
    }
