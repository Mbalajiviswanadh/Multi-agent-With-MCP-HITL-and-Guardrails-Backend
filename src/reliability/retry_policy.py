"""
src/reliability/retry_policy.py
--------------------------------
LangGraph RetryPolicy instances used for each agent node.

Two policies are defined:
  MCP_RETRY_POLICY   — for agents that call external MCP servers
                       (AviationStack, Tavily, Weather) which may hit
                       transient network errors or brief API outages.
  LLM_RETRY_POLICY   — for LLM invocations that may hit rate-limits (429)
                       or transient provider errors (503).

Usage in graph.py:
    graph.add_node("hotel_agent", hotel_agent, retry=MCP_RETRY_POLICY)
"""
from __future__ import annotations

from langgraph.types import RetryPolicy

# Retry policy for MCP tool calls:
#   attempts: 3  |  backoff: 1s, 2s, 4s  |  max wait: 10s
MCP_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
    retry_on=(
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
    ),
)

# Retry policy for LLM calls:
#   attempts: 2  |  backoff: 2s, 4s  |  catches HTTP 429 / 503 from providers
LLM_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=8.0,
    jitter=True,
    retry_on=(
        TimeoutError,
        ConnectionError,
    ),
)
