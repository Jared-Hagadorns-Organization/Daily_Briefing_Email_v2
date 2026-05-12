"""News + local events node using Gemini with Google Search grounding.

Makes 4 separate calls:
  1. National/finance/tech news → structured JSON
  2-4. Local events for Charlotte, Belmont, Lake Wylie → HTML snippets
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

Return a simple HTML list of up to 3 events you find, like this format:
<ul>
<li><a href="URL">Event Name</a> — brief description, date/time if known</li>
</ul>

If you find no events, return: <p>No local events found for Charlotte this weekend.</p>
"""

BELMONT_PROMPT = """Search for events happening TODAY and this coming weekend in Belmont NC.
Look for concerts, festivals, farmers markets, sports, community events, outdoor activities.

Return a simple HTML list of up to 3 events you find, like this format:
<ul>
<li><a href="URL">Event Name</a> — brief description, date/time if known</li>
</ul>

If you find no events, return: <p>No local events found for Belmont this weekend.</p>
"""

LAKE_WYLIE_PROMPT = """Search for events happening TODAY and this coming weekend at Lake Wylie SC.
Look for concerts, festivals, boat events, outdoor activities, community events, sports.

Return a simple HTML list of up to 3 events you find, like this format:
<ul>
<li><a href="URL">Event Name</a> — brief description, date/time if known</li>
</ul>

If you find no events, return: <p>No local events found for Lake Wylie this weekend.</p>
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


def _call_gemini_text(client: genai.Client, prompt: str) -> str:
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
            return response.text or ""
        except Exception:
            pass
    return ""


def _call_gemini_json(client: genai.Client, prompt: str) -> list[dict[str, Any]]:
    return _extract_json_array(_call_gemini_text(client, prompt))


def news_node(state: dict) -> dict:
    api_key = os.environ.get("GEMINI_KEY")
    if not api_key:
        return {"news_items": [], "errors": ["GEMINI_KEY not set"]}

    client = genai.Client(api_key=api_key)

    national = _call_gemini_json(client, NATIONAL_PROMPT)

    charlotte_html = _call_gemini_text(client, CHARLOTTE_PROMPT)
    belmont_html = _call_gemini_text(client, BELMONT_PROMPT)
    lake_wylie_html = _call_gemini_text(client, LAKE_WYLIE_PROMPT)

    local_html = (
        "<h3>Local — Charlotte / Belmont / Lake Wylie</h3>"
        f"<h4>Charlotte NC</h4>{charlotte_html}"
        f"<h4>Belmont NC</h4>{belmont_html}"
        f"<h4>Lake Wylie SC</h4>{lake_wylie_html}"
    )

    return {
        "news_items": national,
        "errors": [],
        "local_events_html": local_html,
    }
