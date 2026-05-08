"""Weather node using Open-Meteo (free, no API key required).

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
import httpx


# Lake Wylie / Belmont area
LAT = 35.10
LON = -81.05

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def weather_node(state: dict) -> dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max,weathercode",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone": "America/New_York",
            "forecast_days": 1,
        }
        r = httpx.get(url, params=params, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})

        code = (daily.get("weathercode") or [None])[0]
        summary = WMO_DESCRIPTIONS.get(code, "Unknown") if code is not None else "Unknown"

        return {
            "weather": {
                "summary": summary,
                "high_f": (daily.get("temperature_2m_max") or [None])[0],
                "low_f": (daily.get("temperature_2m_min") or [None])[0],
                "precip_chance": (daily.get("precipitation_probability_max") or [None])[0],
                "wind_mph": (daily.get("windspeed_10m_max") or [None])[0],
                "alerts": [],
            },
            "errors": [],
        }
    except Exception as e:
        return {"weather": None, "errors": [f"Weather fetch failed: {e}"]}
