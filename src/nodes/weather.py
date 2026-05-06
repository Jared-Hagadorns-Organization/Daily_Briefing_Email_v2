"""Weather node.

Returns a dict like:
    {
      "summary": "Partly cloudy",
      "high_f": 78,
      "low_f": 62,
      "precip_chance": 20,
      "wind_mph": 8,
      "alerts": []
    }
"""
from __future__ import annotations
import os
import httpx


# Lake Wylie / Belmont area
LAT = 35.10
LON = -81.05


def weather_node(state: dict) -> dict:
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return {"weather": None, "errors": ["OPENWEATHER_API_KEY not set"]}

    try:
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            "lat": LAT,
            "lon": LON,
            "exclude": "minutely,hourly",
            "units": "imperial",
            "appid": api_key,
        }
        r = httpx.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        today = (data.get("daily") or [{}])[0]
        weather_summary = (today.get("weather") or [{}])[0].get("description", "")

        return {
            "weather": {
                "summary": weather_summary,
                "high_f": (today.get("temp") or {}).get("max"),
                "low_f": (today.get("temp") or {}).get("min"),
                "precip_chance": int((today.get("pop") or 0) * 100),
                "wind_mph": today.get("wind_speed"),
                "alerts": [a.get("event") for a in data.get("alerts", [])],
            },
            "errors": [],
        }
    except Exception as e:
        return {"weather": None, "errors": [f"Weather fetch failed: {e}"]}
