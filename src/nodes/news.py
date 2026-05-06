"""News + local events node: a single ReAct agent backed by MCP tools.

Tools (stdio MCP servers):
  - tavily-mcp:        web search
  - mcp-server-fetch:  retrieve full article text

The agent's job: scan general topical news, then specifically search for
events in Charlotte / Belmont / Lake Wylie for today and the upcoming weekend.

Returns a list of {topic, headline, summary, url}.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ..llm import get_llm


SYSTEM_PROMPT = """You are a news researcher for a daily personal briefing.

For today, produce a curated list of items in this order:
  1. 3-5 major topical news stories (US/world, finance/markets, technology).
  2. Up to 5 local events happening today or this coming weekend in:
     Charlotte NC, Belmont NC, and Lake Wylie SC. Concerts, festivals,
     farmers markets, sports, community events, etc.

Use the search tool to find candidates and the fetch tool to confirm details
when needed. Skip clickbait, paywalled-only sources, and outdated content.

Output ONLY a JSON array (no prose, no code fences) where each element is:
  {"topic": "national" | "local" | "tech" | "finance",
   "headline": str,
   "summary": str (1-2 sentences),
   "url": str}
"""


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


async def _run_agent() -> list[dict[str, Any]]:
    client = MultiServerMCPClient(
        {
            "tavily": {
                "command": "npx",
                "args": ["-y", "tavily-mcp"],
                "transport": "stdio",
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent(get_llm(temperature=0.2), tools)

    result = await agent.ainvoke(
        {"messages": [("system", SYSTEM_PROMPT), ("user", "Build today's list.")]}
    )
    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    return _extract_json_array(content)


def news_node(state: dict) -> dict:
    try:
        items = asyncio.run(_run_agent())
        return {"news_items": items, "errors": []}
    except Exception as e:
        return {"news_items": [], "errors": [f"News agent failed: {e}"]}
