"""
src/reliability/circuit_breaker.py
------------------------------------
Circuit breakers for external MCP tool services using pybreaker.

Each external tool (Tavily, AviationStack, Weather) gets its own breaker.
When a service fails 3 consecutive times, the breaker trips to OPEN:
  - All requests instantly return a fallback message (no network wait).
  - After 60 seconds the breaker moves to HALF-OPEN and probes recovery.

Also provides `call_with_timeout()` — an async helper that wraps any
coroutine with a strict time ceiling to prevent hung worker threads.

Install: pip install pybreaker
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine

from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.reliability.circuit_breaker")


# ---------------------------------------------------------------------------
# Optional pybreaker import — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import pybreaker

    class _LogListener(pybreaker.CircuitBreakerListener):
        def state_change(self, cb, old_state, new_state):
            logger.warning(
                "[CIRCUIT] %s breaker: %s -> %s",
                cb.name, old_state.name, new_state.name,
            )

    _listener = _LogListener()

    tavily_breaker = pybreaker.CircuitBreaker(
        fail_max=3,
        reset_timeout=60,
        name="Tavily",
        listeners=[_listener],
    )
    aviation_breaker = pybreaker.CircuitBreaker(
        fail_max=3,
        reset_timeout=60,
        name="AviationStack",
        listeners=[_listener],
    )
    weather_breaker = pybreaker.CircuitBreaker(
        fail_max=3,
        reset_timeout=60,
        name="Weather",
        listeners=[_listener],
    )
    _PYBREAKER_AVAILABLE = True
    logger.info("[CIRCUIT] pybreaker circuit breakers initialised (Tavily, AviationStack, Weather)")

except ImportError:
    logger.warning(
        "[CIRCUIT] pybreaker not installed — circuit breakers disabled. "
        "Run: pip install pybreaker"
    )
    _PYBREAKER_AVAILABLE = False
    # Provide no-op stubs so the rest of the code doesn't need to branch on availability
    tavily_breaker = None      # type: ignore[assignment]
    aviation_breaker = None    # type: ignore[assignment]
    weather_breaker = None     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

async def call_with_timeout(
    coro: Coroutine[Any, Any, Any],
    timeout_seconds: float = 8.0,
    fallback_message: str = "Live data unavailable due to server timeout. Providing general estimates.",
) -> Any:
    """
    Await `coro` with a strict timeout.
    Returns `fallback_message` if the coroutine exceeds `timeout_seconds`.
    Never raises — agents receive graceful fallback text instead.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning("[TIMEOUT] Coroutine timed out after %.1fs", timeout_seconds)
        return fallback_message
    except Exception as exc:
        logger.error("[TIMEOUT] Coroutine raised unexpected error: %s", exc)
        return fallback_message
