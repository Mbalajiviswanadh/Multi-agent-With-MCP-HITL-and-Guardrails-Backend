"""
src/agents/specialists/user_mcp_agent.py
----------------------------------------
Dynamic User MCP Agent.

Runs ONLY when the Supervisor detects that the user wants to use one of their
connected external MCP servers (e.g., Google Drive, Google Calendar, Notion,
or any custom remote MCP server).

Zero overhead on normal travel/chat queries: external MCP clients are only
initialized here when `user_mcp_agent` is actually invoked.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.mcp_dynamic.client import load_user_mcp_tools_sync
from src.memory.thread_memory import filter_real_messages
from src.memory.vector_store import _get_embedding_model
from src.reliability.llm_factory import bind_tools_resilient, resilient_llm

logger = get_logger("TripMateAI.agents.user_mcp")

_MAX_TOOL_ITERATIONS = 6
_MAX_BOUND_TOOLS = 6
# Thread pool for running async MCP tools (remote MCP adapters) from sync context
_ASYNC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mcp_async")

# In-process cache for FastEmbed model & tool description vectors (0 LLM tokens)
_EMBED_MODEL = None
_TOOL_EMBED_CACHE: dict[str, list[float]] = {}


def _get_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = _get_embedding_model()
    return _EMBED_MODEL


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _filter_target_servers(query: str, available_meta: list[dict[str, Any]]) -> set[str] | None:
    """
    Stage 1 (Server Router): Detect if the user explicitly named one or more connected
    MCP servers (e.g. 'notion', 'google drive', 'calendar') so we only connect to and
    load tools from those servers instead of all enabled servers.
    Returns None if no specific server was matched (falls back to loading all enabled).
    """
    if not available_meta or len(available_meta) <= 1:
        return None

    q_lower = query.lower()
    matched_ids: set[str] = set()

    for srv in available_meta:
        srv_id = str(srv.get("id", ""))
        if not srv_id:
            continue
        display = str(srv.get("display_name", "")).lower()
        slug = str(srv.get("server_slug", "")).lower()
        tokens = {w for w in re.split(r"[\s_-]+", f"{display} {slug}") if len(w) >= 3 and w not in ("google", "custom", "remote", "mcp")}

        # Common domain aliases for built-in Google servers
        if slug == "google_drive":
            tokens.update({"drive", "doc", "docs", "document", "sheet", "sheets", "spreadsheet", "excel"})
        elif slug == "google_calendar":
            tokens.update({"calendar", "calnder", "event", "schedule"})

        if (display and display in q_lower) or any(re.search(rf"\b{re.escape(tok)}\b", q_lower) for tok in tokens):
            matched_ids.add(srv_id)

    if matched_ids:
        matched_names = [s.get("display_name", s.get("server_slug")) for s in available_meta if str(s.get("id", "")) in matched_ids]
        logger.info("[USER_MCP_AGENT] Stage-1 Server Filter -> %s", matched_names)
        return matched_ids
    return None


def _trim(text: str, limit: int = 3000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _is_tool_error(output: str) -> bool:
    low = output.lower()
    if low.startswith("error") or "api error" in low or "tool execution error" in low:
        return True
    error_markers = (
        "apiresponseerror",
        "validation_error",
        '"iserror":true',
        '"iserror": true',
        '"status":400',
        '"status": 400',
        '"status":401',
        '"status":403',
        '"status":404',
        '"object":"error"',
    )
    return any(m in low for m in error_markers)


def _fallback_trace_summary(tool_trace: list[dict[str, Any]]) -> str:
    if not tool_trace:
        return "Completed external integration actions."
    urls: list[str] = []
    errors: list[str] = []
    for item in tool_trace:
        out = str(item.get("output", ""))
        found = re.findall(r"https?://[^\s\"'<>}\]]+", out)
        for u in found:
            if u not in urls:
                urls.append(u)
        if item.get("status") == "error":
            errors.append(f"- `{item.get('tool')}`: {out[:160]}")
    parts: list[str] = []
    if urls:
        parts.append("Created / updated items:\n" + "\n".join(f"- {u}" for u in urls))
    if errors and not urls:
        parts.append("Some tool calls returned errors:\n" + "\n".join(errors))
    return "\n\n".join(parts) if parts else "Completed external integration actions."
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _compact_schema(obj: Any, depth: int = 0) -> Any:
    """Trim verbose JSON schema descriptions so large MCP servers fit inside Groq's 8k TPM limit
    while preserving short parameter hints (like 'title' and 'content') at nested depths."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                if depth >= 8:
                    continue
                limit = 100 if depth <= 1 else 55
                out[k] = v[:limit] + "…" if len(v) > limit else v
            else:
                out[k] = _compact_schema(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_compact_schema(item, depth + 1) for item in obj]
    return obj


def _compact_tool_for_llm(tool: Any) -> Any:
    """Return a lightweight copy of a tool for LLM schema binding (execution still uses original)."""
    desc = (getattr(tool, "description", "") or "").strip()
    short_desc = desc[:180] + "…" if len(desc) > 180 else desc
    args_schema = getattr(tool, "args_schema", None)
    compact_args = _compact_schema(args_schema) if isinstance(args_schema, dict) else args_schema

    if isinstance(tool, StructuredTool):
        try:
            return tool.model_copy(update={"description": short_desc, "args_schema": compact_args})
        except Exception:
            pass
    return tool


def _select_relevant_tools(
    query: str,
    tools: list[Any],
    tool_origin: dict[str, str],
    max_tools: int = _MAX_BOUND_TOOLS,
) -> list[Any]:
    """
    Stage 2 (Semantic Tool RAG): Rank tools using local FastEmbed cosine similarity
    + exact keyword boost so only the top `max_tools` are bound to the LLM.
    Uses 0 LLM tokens and caches tool vectors in-memory for instant retrieval.
    """
    if len(tools) <= max_tools:
        return tools

    q_words = {w for w in re.split(r"[\W_]+", query.lower()) if len(w) >= 3}

    try:
        embedder = _get_embedder()
        q_vec = embedder.embed_query(query)

        # Embed any uncached tools in a single batch
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        tool_keys: list[str] = []

        for t in tools:
            srv = tool_origin.get(t.name, "MCP")
            short_d = (getattr(t, "description", "") or "").replace("\n", " ").strip()[:240]
            text = f"{t.name} ({srv}): {short_d}"
            key = f"{srv}::{t.name}::{short_d[:60]}"
            tool_keys.append(key)
            if key not in _TOOL_EMBED_CACHE:
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            vecs = embedder.embed_documents(missing_texts)
            for k, v in zip(missing_keys, vecs):
                _TOOL_EMBED_CACHE[k] = v

        scored: list[tuple[float, Any]] = []
        for t, key in zip(tools, tool_keys):
            sim = _cosine(q_vec, _TOOL_EMBED_CACHE[key])
            # Hybrid boost: weight exact word matches in tool.name higher than description
            t_name_low = t.name.lower()
            name_hits = sum(1 for w in q_words if w in t_name_low)
            hay = f"{tool_origin.get(t.name, '')} {getattr(t, 'description', '')[:160]}".lower()
            desc_hits = sum(1 for w in q_words if w in hay and w not in t_name_low)
            scored.append((sim + 0.08 * min(name_hits, 4) + 0.03 * min(desc_hits, 3), t))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored[:max_tools]]

    except Exception as exc:
        logger.warning("[USER_MCP_AGENT] Semantic tool RAG fallback (%s); using keyword match", exc)
        scored = []
        for t in tools:
            hay = f"{t.name} {tool_origin.get(t.name, '')} {getattr(t, 'description', '')}".lower()
            score = sum(1 for w in q_words if w in hay)
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored[:max_tools]]

    logger.info(
        "[USER_MCP_AGENT] Stage-2 Semantic Tool RAG (%d -> %d tools): %s",
        len(tools),
        len(selected),
        [t.name for t in selected],
    )
    return selected


def _invoke_tool(tool: Any, args: dict) -> str:
    """
    Invoke a LangChain tool safely regardless of whether it is sync or async-only.

    - StructuredTool built from a sync function: call .invoke() directly.
    - Async-only MCP adapter tool: run its coroutine in a fresh event loop
      on a background thread (avoids RuntimeError: event loop already running).
    """
    # Prefer sync invoke for our built-in Drive/Calendar tools (they use httpx.Client)
    if isinstance(tool, StructuredTool) and tool.coroutine is None:
        # Purely sync tool — fast path
        return str(tool.invoke(args))

    # Async-only tool (langchain_mcp_adapters StructuredTool has .coroutine set).
    # Run in a background thread with its own event loop to avoid nesting issues.
    import asyncio

    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(tool.ainvoke(args))
        finally:
            loop.close()

    future = _ASYNC_POOL.submit(_run_in_new_loop)
    return str(future.result(timeout=30))


def user_mcp_agent(state: TravelState) -> dict[str, Any]:
    start_time = time.time()
    user_id = state.get("user_id", "anonymous")
    query = state.get("resolved_query") or state.get("user_query", "")
    available_meta = state.get("connected_mcp_tools") or []
    servers_available = [s.get("display_name", s.get("server_slug", "MCP")) for s in available_meta]

    logger.info("[USER_MCP_AGENT] -> Loading connected MCP tools for user=%s...", user_id)

    if not user_id or user_id == "anonymous":
        msg = "Please sign in to connect and use external MCP integrations."
        return {
            "user_mcp_results": msg,
            "final_response": state.get("final_response") or msg,
            "mcp_trace": {
                "invoked": True,
                "servers_available": [],
                "servers_used": [],
                "tool_calls": [],
                "error": "User not signed in",
            },
            "messages": [AIMessage(content=msg)],
        }

    target_server_ids = _filter_target_servers(query, available_meta)
    tools, tool_origin = load_user_mcp_tools_sync(user_id, allowed_server_ids=target_server_ids)
    if not tools:
        msg = (
            "No active external MCP tools were found for your account. "
            "Open **MCP Integrations** in the sidebar to connect Google Drive, Google Calendar, or a custom MCP server."
        )
        existing_final = state.get("final_response", "")
        combined = f"{existing_final}\n\n---\n{msg}" if existing_final else msg
        return {
            "user_mcp_results": msg,
            "final_response": combined,
            "mcp_trace": {
                "invoked": True,
                "servers_available": servers_available,
                "servers_used": [],
                "tool_calls": [],
                "error": "No enabled MCP tools available",
            },
            "messages": [AIMessage(content=msg)],
        }

    tool_map = {t.name: t for t in tools}

    # Build context so external tools have access to the generated plan OR recent thread history
    context_blocks: list[str] = []
    if state.get("final_response"):
        context_blocks.append(f"Current Trip Plan / Response:\n{_trim(state['final_response'], 1800)}")
    elif state.get("itinerary"):
        context_blocks.append(f"Current Itinerary:\n{_trim(state['itinerary'], 1800)}")
    else:
        recent = filter_real_messages(state.get("messages", []))[-3:]
        if recent:
            hist = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_trim(str(m.content), 900)}"
                for m in recent
            )
            context_blocks.append(f"Recent Conversation:\n{hist}")

    constraints = state.get("trip_constraints") or {}
    if any(constraints.get(k) for k in ("destination", "origin", "duration", "budget")):
        context_blocks.append(f"Trip Constraints: {json.dumps(constraints)}")

    system_prompt = (
        "You are an external integration execution agent for TripMate AI.\n"
        "Use the user's connected MCP tools to fulfill their request (for example: "
        "creating a Google Doc/Sheet, adding events to Google Calendar, or creating/updating Notion pages).\n"
        "IMPORTANT RULES:\n"
        "1. Always populate BOTH the title/name property (e.g. properties={'title': '...'}) AND the full content body — never leave a page blank or unnamed.\n"
        "2. When creating a new top-level page, omit `parent` unless the user named a specific existing database/page. To nest a child page inside a page you just created, use `parent={'page_id': '<created_id>'}`.\n"
        "3. If a tool returns a validation or property error (e.g. 'Property Name not found'), "
        "read the error carefully and use the exact property name it suggests (e.g. 'Title').\n"
        "4. Always include every created page/document/event URL returned by the tools in your final response."
    )
    user_prompt = f"User Request: {query}\n\n" + "\n\n".join(context_blocks)

    selected_tools = _select_relevant_tools(query, tools, tool_origin, max_tools=_MAX_BOUND_TOOLS)
    compact_tools = [_compact_tool_for_llm(t) for t in selected_tools]
    llm_with_tools = bind_tools_resilient(compact_tools)
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    llm_calls = state.get("llm_calls", 0)
    tool_trace: list[dict[str, Any]] = []
    servers_used: list[str] = []
    final_action_text = ""

    try:
        for _ in range(_MAX_TOOL_ITERATIONS):
            ai_msg = llm_with_tools.invoke(messages)
            llm_calls += 1
            messages.append(ai_msg)

            calls = getattr(ai_msg, "tool_calls", None) or []
            if not calls:
                final_action_text = str(ai_msg.content or "").strip()
                break

            for tc in calls:
                t_name = tc.get("name", "")
                t_args = tc.get("args", {}) or {}
                t_id = tc.get("id", t_name)
                server_label = tool_origin.get(t_name, "External MCP")
                if server_label not in servers_used:
                    servers_used.append(server_label)

                tool_obj = tool_map.get(t_name)
                if not tool_obj:
                    output = f"Tool '{t_name}' is not available."
                    status = "error"
                else:
                    try:
                        output = _invoke_tool(tool_obj, t_args)
                        status = "error" if _is_tool_error(output) else "success"
                    except Exception as t_exc:
                        sub_excs = getattr(t_exc, "exceptions", None)
                        if sub_excs:
                            details = "; ".join(f"{type(e).__name__}: {e}" for e in sub_excs)
                            output = f"Tool execution error: {details}"
                        else:
                            output = f"Tool execution error: {t_exc}"
                        status = "error"

                logger.info("[USER_MCP_AGENT] Executed %s (%s) -> %s", t_name, server_label, status)
                tool_trace.append({
                    "server": server_label,
                    "tool": t_name,
                    "args": {k: (str(v)[:120] + "…" if len(str(v)) > 120 else v) for k, v in t_args.items()},
                    "status": status,
                    "output": output[:350],
                })
                messages.append(ToolMessage(content=output, tool_call_id=t_id))

        if not final_action_text:
            # One final text-only summary call so the user always gets created URLs & status
            try:
                summary_prompt = (
                    "Summarize what was done for the user based on these tool results. "
                    "Include every created page/document/event URL so the user can click it, "
                    "and mention any errors if a step failed:\n\n"
                    + json.dumps(tool_trace, indent=2)[:2500]
                )
                sum_msg = resilient_llm.invoke([
                    SystemMessage(content="Summarize tool execution results concisely with clickable links."),
                    HumanMessage(content=summary_prompt),
                ])
                llm_calls += 1
                final_action_text = str(sum_msg.content or "").strip()
            except Exception:
                pass
            if not final_action_text:
                final_action_text = _fallback_trace_summary(tool_trace)

    except Exception as exc:
        logger.error("[USER_MCP_AGENT] Execution failed: %s", exc)
        if tool_trace:
            # Tools already ran (e.g. create_google_doc succeeded before summary hit 429) — preserve links!
            partial = _fallback_trace_summary(tool_trace)
            final_action_text = f"{partial}\n\n_(Note: summary generation hit an API limit: {str(exc)[:140]})_"
        else:
            final_action_text = f"Could not complete external MCP action: {exc}"

    elapsed = time.time() - start_time
    logger.info("[USER_MCP_AGENT] OK Finished (%.1fs) | Tools called: %d", elapsed, len(tool_trace))

    existing_final = state.get("final_response", "").strip()
    if existing_final:
        combined_response = f"{existing_final}\n\n---\n\n### Connected MCP Actions\n{final_action_text}"
    else:
        combined_response = final_action_text

    return {
        "user_mcp_results": final_action_text,
        "final_response": combined_response,
        "llm_calls": llm_calls,
        "mcp_trace": {
            "invoked": True,
            "servers_available": servers_available,
            "servers_used": servers_used,
            "tool_calls": tool_trace,
        },
        "messages": [AIMessage(content=final_action_text)],
    }
