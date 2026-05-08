"""Sports node: ESPN public JSON for the four tracked teams.

For each team we hit two endpoints:
  - /teams/{slug}/schedule   -> last completed + next upcoming game
  - /news?team={slug}&limit  -> top 3 articles by date

No LLM. No MCP. Eight HTTP calls per run.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

import httpx


ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

TEAMS: list[dict[str, str]] = [
    {
        "team": "Carolina Panthers",
        "league": "NFL",
        "sport_path": "football/nfl",
        "team_slug": "car",
    },
    {
        "team": "Charlotte Hornets",
        "league": "NBA",
        "sport_path": "basketball/nba",
        "team_slug": "cha",
    },
    {
        "team": "Carolina Hurricanes",
        "league": "NHL",
        "sport_path": "hockey/nhl",
        "team_slug": "car",
    },
]


def _summarize_event(ev: dict[str, Any], completed: bool) -> dict[str, Any]:
    competitions = ev.get("competitions") or [{}]
    comp = competitions[0]
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})

    summary: dict[str, Any] = {
        "date": ev.get("date"),
        "name": ev.get("name", ""),
        "short_name": ev.get("shortName", ""),
        "venue": (comp.get("venue") or {}).get("fullName"),
        "home_team": (home.get("team") or {}).get("displayName"),
        "away_team": (away.get("team") or {}).get("displayName"),
    }
    if completed:
        summary["home_score"] = home.get("score")
        summary["away_score"] = away.get("score")
        if home.get("winner"):
            summary["winner"] = "home"
        elif away.get("winner"):
            summary["winner"] = "away"
        else:
            summary["winner"] = "tie"
    return summary


def _parse_last_and_next(
    schedule: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    events = schedule.get("events", [])
    now = datetime.now(timezone.utc)
    last: dict[str, Any] | None = None
    next_: dict[str, Any] | None = None

    for ev in events:
        date_str = ev.get("date")
        if not date_str:
            continue
        try:
            event_time = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        completed = (
            ev.get("status", {}).get("type", {}).get("completed", False)
            or event_time < now
        )
        if completed and event_time < now:
            last = _summarize_event(ev, completed=True)
        elif not completed and event_time >= now and next_ is None:
            next_ = _summarize_event(ev, completed=False)

    return last, next_


def _summarize_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": article.get("headline", ""),
        "summary": article.get("description", ""),
        "url": ((article.get("links") or {}).get("web") or {}).get("href", ""),
        "published": article.get("published"),
    }


def _fetch_schedule(client: httpx.Client, sport_path: str, team_slug: str) -> dict[str, Any]:
    url = f"{ESPN_BASE}/{sport_path}/teams/{team_slug}/schedule"
    r = client.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def _fetch_news(
    client: httpx.Client, sport_path: str, team_slug: str, limit: int = 3
) -> list[dict[str, Any]]:
    url = f"{ESPN_BASE}/{sport_path}/news"
    r = client.get(url, params={"team": team_slug, "limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return (data.get("articles") or [])[:limit]


def sports_node(state: dict) -> dict:
    updates: list[dict[str, Any]] = []
    errors: list[str] = []

    with httpx.Client(headers={"User-Agent": "DailyBriefing/1.0"}) as client:
        for cfg in TEAMS:
            try:
                schedule = _fetch_schedule(client, cfg["sport_path"], cfg["team_slug"])
                last, next_ = _parse_last_and_next(schedule)
            except Exception as e:
                errors.append(f"sports schedule failed for {cfg['team']}: {e}")
                last, next_ = None, None

            try:
                articles = _fetch_news(client, cfg["sport_path"], cfg["team_slug"], limit=3)
                news = [_summarize_article(a) for a in articles]
            except Exception as e:
                errors.append(f"sports news failed for {cfg['team']}: {e}")
                news = []

            updates.append(
                {
                    "team": cfg["team"],
                    "league": cfg["league"],
                    "last_game": last,
                    "next_game": next_,
                    "news": news,
                }
            )

    return {"sports": updates, "errors": errors}
