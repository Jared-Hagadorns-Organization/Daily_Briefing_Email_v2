"""News + local events node using Gemini with Google Search grounding.

Makes two separate calls to guarantee both national news and local events
are fetched — one call per topic so neither gets skipped.
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

LOCAL_PROMPT = """Search for local events happening TODAY and this coming weekend near Charlotte NC,
Belmont NC, and Lake Wylie SC. Search for concerts, festivals, farmers markets, sports,
community events, and outdoor activities. Run searches like:
- "Charlotte NC events this weekend"
- "Belmont NC events today"
- "Lake Wylie SC events this weekend"

Return ONLY a valid JSON array (no prose, no markdown, no code fences) with up to 5 results:
[{"topic": "local", "headline": str, "summary": str, "url": str}]

If no events are found, return an empty array: []
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
        except Exception as e:
            last = str(e)
    return []


def news_node(state: dict) -> dict:
    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        return {"news_items": [], "errors": ["GEMINI_KEY not set"]}

    client = genai.Client(api_key=api_key)
    errors: list[str] = []

    national = _call_gemini(client, NATIONAL_PROMPT)
    local = _call_gemini(client, LOCAL_PROMPT)

    items = national + local
    if not items:
        errors.append("News agent returned no results")

    return {"news_items": items, "errors": errors}
