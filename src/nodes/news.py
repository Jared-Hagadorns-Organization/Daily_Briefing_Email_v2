"""News + local events node using Gemini with Google Search grounding.

Makes 4 separate calls:
  1. National news
  2. Charlotte NC local events
  3. Belmont NC local events
  4. Lake Wylie SC local events
"""
from __future__ import annotations
import json
import os
import time
from typing import Any

from google import genai
from google.genai import types


NATIONAL_PROMPT = """Search for today's top news stories and return ONLY a valid JSON array.
Find 4-5 major stories from today covering US/world news, finance/markets, and technology.

Output ONLY a JSON array (no prose, no markdown, no code fences):
[{"topic": "national" | "tech" | "finance", "headline": str, "summary": str, "url": str}]
"""

CHARLOTTE_PROMPT = """Search for events happening TODAY and this coming weekend in Charlotte NC.
Look for concerts, festivals, farmers markets, sports, community events, outdoor activities.
Search "Charlotte NC events this weekend" and "Charlotte NC things to do today".

Return ONLY a valid JSON array (no prose, no markdown, no code fences) with up to 3 results:
[{"topic": "local", "headline": str, "summary": str, "url": str}]

If nothing is found, return: []
"""

BELMONT_PROMPT = """Search for events happening TODAY and this coming weekend in Belmont NC.
Look for concerts, festivals, farmers markets, sports, community events, outdoor activities.
Search "Belmont NC events this weekend" and "Belmont NC things to do today".

Return ONLY a valid JSON array (no prose, no markdown, no code fences) with up to 3 results:
[{"topic": "local", "headline": str, "summary": str, "url": str}]

If nothing is found, return: []
"""

LAKE_WYLIE_PROMPT = """Search for events happening TODAY and this coming weekend at Lake Wylie SC.
Look for concerts, festivals, boat events, outdoor activities, community events, sports.
Search "Lake Wylie SC events this weekend" and "Lake Wylie SC things to do today".

Return ONLY a valid JSON array (no prose, no markdown, no code fences) with up to 3 results:
[{"topic": "local", "headline": str, "summary": str, "url": str}]

If nothing is found, return: []
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


def _call_gemini(client: genai.Client, prompt: str) -> list[dict[str, Any]]:
    delays = [20, 40, 60]
    for delay in [0] + delays:
        if delay:
            time.sleep(delay)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            return _extract_json_array(response.text)
        except Exception:
            pass
    return []


def news_node(state: dict) -> dict:
    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        return {"news_items": [], "errors": ["GEMINI_KEY not set"]}

    client = genai.Client(api_key=api_key)

    national = _call_gemini(client, NATIONAL_PROMPT)
    charlotte = _call_gemini(client, CHARLOTTE_PROMPT)
    belmont = _call_gemini(client, BELMONT_PROMPT)
    lake_wylie = _call_gemini(client, LAKE_WYLIE_PROMPT)

    items = national + charlotte + belmont + lake_wylie

    return {"news_items": items, "errors": [] if items else ["News agent returned no results"]}
