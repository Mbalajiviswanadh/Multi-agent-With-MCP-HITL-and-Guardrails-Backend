"""
src/reliability/llm_factory.py
-------------------------------
Builds a resilient LLM chain using LangChain's `.with_fallbacks()`.

Both primary and fallback are Groq-hosted models using your single GROQ_API_KEY.

Primary:  openai/gpt-oss-120b   (120B — highest quality, your original model)
Fallback: openai/gpt-oss-20b    (20B  — same OpenAI OSS family, much higher TPM quota)
                                  Activates transparently on 429/413 rate limit errors.

Why gpt-oss-20b as fallback?
  - Same model family as primary — consistent response style and formatting.
  - Being a smaller 20B model, Groq allocates it a significantly higher TPM budget,
    so it absorbs overflow when the 120B model hits its per-minute token limit.
  - Zero extra cost or API keys — both run on your existing GROQ_API_KEY.

Circuit Breaker:
  When the 120B model exhausts its daily TPD quota (200k tokens/day), subsequent
  calls fail immediately with 429. The circuit breaker detects this and routes
  all calls directly to the 20B fallback for the remainder of that hour,
  avoiding wasted latency retrying an exhausted quota.
"""
from __future__ import annotations

import threading
import time

from langchain_groq import ChatGroq

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.reliability.llm")

# ---------------------------------------------------------------------------
# Model configuration  (both available in your Groq account)
# ---------------------------------------------------------------------------

# Primary: Highest quality model — best reasoning for travel planning
_PRIMARY_MODEL = "openai/gpt-oss-120b"

# Fallback: Lighter model from the same OSS family — higher TPM quota
# Kicks in automatically if primary hits 429 / 413 rate limits
_FALLBACK_MODEL = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# TPD Circuit Breaker
# Reset after 1 hour — Groq TPD is a rolling 24h window, but 1h is usually
# enough for a dev session reset.
# ---------------------------------------------------------------------------
_TPD_LOCK = threading.Lock()
_primary_tpd_exhausted: bool = False
_primary_tpd_reset_at: float = 0.0
_TPD_RESET_SECONDS = 3600  # 1 hour


def _mark_primary_tpd_exhausted() -> None:
    """Call when 120B returns a TPD error (Used >= Limit)."""
    global _primary_tpd_exhausted, _primary_tpd_reset_at
    with _TPD_LOCK:
        _primary_tpd_exhausted = True
        _primary_tpd_reset_at = time.time() + _TPD_RESET_SECONDS
        logger.warning(
            "[LLM] 120B daily quota (TPD) exhausted — routing to 20B fallback for 1h"
        )


def _is_primary_available() -> bool:
    global _primary_tpd_exhausted, _primary_tpd_reset_at
    with _TPD_LOCK:
        if _primary_tpd_exhausted and time.time() >= _primary_tpd_reset_at:
            _primary_tpd_exhausted = False
            logger.info("[LLM] 120B circuit breaker reset — primary model re-enabled")
        return not _primary_tpd_exhausted


def _build_primary() -> ChatGroq:
    return ChatGroq(
        model=_PRIMARY_MODEL,
        api_key=settings.groq_api_key,
        request_timeout=settings.groq_request_timeout,
    )


def _build_fallback() -> ChatGroq:
    return ChatGroq(
        model=_FALLBACK_MODEL,
        api_key=settings.groq_api_key,
        request_timeout=settings.groq_request_timeout,
    )


def build_resilient_llm():
    """
    Returns a ChatGroq primary with a ChatGroq fallback.
    Both use the same GROQ_API_KEY — no additional credentials required.

    If the 120B daily quota is exhausted, returns the 20B model directly
    (skips the failed primary to avoid extra latency).
    """
    if not _is_primary_available():
        logger.info("[LLM] Using 20B direct (120B TPD circuit open)")
        return _build_fallback()

    primary = _build_primary()
    fallback = _build_fallback()
    logger.info(
        "[LLM] Primary  : Groq/%s (timeout=%.1fs)",
        _PRIMARY_MODEL,
        settings.groq_request_timeout,
    )
    logger.info(
        "[LLM] Fallback : Groq/%s (activates on 429/413 rate limits)",
        _FALLBACK_MODEL,
    )

    # Wrap so TPD errors on primary auto-mark the circuit breaker
    base = primary.with_fallbacks([fallback])
    return _TpdAwareLLM(base, fallback_runner=fallback)


class _TpdAwareLLM:
    """
    Thin wrapper that intercepts 429 TPD errors, marks the circuit breaker,
    then delegates all other attribute access to the inner chain.
    """

    def __init__(self, inner, fallback_runner=None):
        self._inner = inner
        self._fallback_runner = fallback_runner

    def invoke(self, *args, **kwargs):
        try:
            return self._inner.invoke(*args, **kwargs)
        except Exception as exc:
            err = str(exc)
            if "429" in err and ("TPD" in err or "tokens per day" in err or "Used 1" in err):
                _mark_primary_tpd_exhausted()
                # Retry immediately on the fallback runner (which preserves bound tools if any)
                runner = self._fallback_runner if self._fallback_runner is not None else _build_fallback()
                return runner.invoke(*args, **kwargs)
            raise

    def bind_tools(self, tools, **kwargs):
        return self._inner.bind_tools(tools, **kwargs)

    def with_fallbacks(self, fallbacks, **kwargs):
        return self._inner.with_fallbacks(fallbacks, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Module-level singleton — import this everywhere instead of constructing ChatGroq directly
resilient_llm = build_resilient_llm()


def bind_tools_resilient(tools: list):
    """
    Bind tools to BOTH primary (120B) and fallback (20B) BEFORE `.with_fallbacks()`
    so 429/413 rate-limit fallback works reliably for tool-calling agents.

    If the 120B daily quota is exhausted (circuit open), binds tools only to
    the 20B model and returns it directly.
    """
    if not _is_primary_available():
        logger.info("[LLM] bind_tools_resilient: 20B direct (120B TPD circuit open)")
        return _build_fallback().bind_tools(tools, tool_choice="auto")

    primary_with_tools = _build_primary().bind_tools(tools, tool_choice="auto")
    fallback_with_tools = _build_fallback().bind_tools(tools, tool_choice="auto")

    base = primary_with_tools.with_fallbacks([fallback_with_tools])

    # Wrap with TPD-awareness, preserving fallback_with_tools so tools stay bound
    return _TpdAwareLLM(base, fallback_runner=fallback_with_tools)
