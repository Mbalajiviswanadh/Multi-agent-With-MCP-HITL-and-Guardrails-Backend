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

import asyncio
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.state import TravelState
from src.core.logging_config import get_logger
from src.mcp_dynamic.client import load_user_mcp_tools
from src.memory.thread_memory import filter_real_messages
from src.reliability.llm_factory import bind_tools_resilient

logger = get_logger("TripMateAI.agents.user_mcp")

_MAX_TOOL_ITERATIONS = 4


def _trim(text: str, limit: int = 3000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


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

    tools, tool_origin = load_user_mcp_tools(user_id)
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
        "creating a Google Doc with their trip plan, adding events to Google Calendar, "
        "or calling a connected remote MCP tool).\n"
        "After executing the tools, return a clear, concise confirmation including any "
        "created document/event links returned by the tools."
    )
    user_prompt = f"User Request: {query}\n\n" + "\n\n".join(context_blocks)

    llm_with_tools = bind_tools_resilient(tools)
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
                        # MCP adapter StructuredTools are async-only (ainvoke works for both sync & async tools)
                        raw_out = asyncio.run(tool_obj.ainvoke(t_args))
                        output = str(raw_out)
                        status = "error" if output.lower().startswith("error") or "api error" in output.lower() else "success"
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
            final_action_text = "Completed external integration actions."

    except Exception as exc:
        logger.error("[USER_MCP_AGENT] Execution failed: %s", exc)
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
