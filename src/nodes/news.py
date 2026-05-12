"""News + local events node using Gemini with Google Search grounding.

Requires GEMINI_KEY secret. Uses gemini-2.5-flash with native Google Search
to reliably find both national news and local Charlotte/Belmont/Lake Wylie events.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any

from google import genai
from google.genai import types


PROMPT = """You are a news researcher for a daily personal briefing.

Produce a curated list of items covering BOTH of the following:

1. NATIONAL NEWS — Find 3-5 major stories from today covering US/world news,
   finance/markets, and technology.

2. LOCAL EVENTS — Search specifically for events happening TODAY or this coming
   weekend in Charlotte NC, Belmont NC, and Lake Wylie SC. Look for concerts,
   festivals, farmers markets, sports, community events, outdoor activities.
   Find up to 5 local events. Search "Charlotte NC events this weekend",
   "Belmont NC events today", and "Lake Wylie SC events this weekend" separately.

Output ONLY a valid JSON array (no prose, no code fences) where each element is:
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
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def news_node(state: dict) -> dict:
    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        return {"news_items": [], "errors": ["GEMINI_KEY not set"]}

    client = genai.Client(api_key=api_key)

    delays = [20, 40, 60]
    last_error = ""
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=PROMPT,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            items = _extract_json_array(response.text)
            return {"news_items": items, "errors": []}
        except Exception as e:
            last_error = str(e)
            if attempt == len(delays):
                break

    return {"news_items": [], "errors": [f"News agent failed: {last_error}"]}
