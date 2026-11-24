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
"""
from __future__ import annotations

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


def build_resilient_llm():
    """
    Returns a ChatGroq primary with a ChatGroq fallback.
    Both use the same GROQ_API_KEY — no additional credentials required.
    """
    primary = ChatGroq(
        model=_PRIMARY_MODEL,
        api_key=settings.groq_api_key,
        request_timeout=settings.groq_request_timeout,
    )
    logger.info(
        "[LLM] Primary  : Groq/%s (timeout=%.1fs)",
        _PRIMARY_MODEL,
        settings.groq_request_timeout,
    )

    fallback = ChatGroq(
        model=_FALLBACK_MODEL,
        api_key=settings.groq_api_key,
        request_timeout=settings.groq_request_timeout,
    )
    logger.info(
        "[LLM] Fallback : Groq/%s (activates on 429/413 rate limits)",
        _FALLBACK_MODEL,
    )

    return primary.with_fallbacks([fallback])


# Module-level singleton — import this everywhere instead of constructing ChatGroq directly
resilient_llm = build_resilient_llm()
