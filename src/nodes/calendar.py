"""Google Calendar node using OAuth refresh token flow.

One-time setup (run locally):
  1. GCP Console -> APIs & Services -> Enable Google Calendar API.
  2. OAuth consent screen -> External, add yourself as test user.
  3. Credentials -> Create OAuth client ID -> Application type: Desktop app.
  4. Run scripts/get_refresh_token.py once locally to obtain a refresh_token.
  5. Store these as GitHub repo secrets:
       GOOGLE_CLIENT_ID
       GOOGLE_CLIENT_SECRET
       GOOGLE_REFRESH_TOKEN
"""
from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _get_credentials() -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def _today_window() -> tuple[str, str]:
    """Today 00:00 -> 23:59:59 in local time, returned as RFC3339 with offset."""
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    today = datetime.now(local_tz).date()
    start = datetime.combine(today, time.min, tzinfo=local_tz)
    end = datetime.combine(today, time.max, tzinfo=local_tz)
    return start.isoformat(), end.isoformat()


def calendar_node(state: dict) -> dict:
    errors: list[str] = []
    try:
        creds = _get_credentials()
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        return {"calendar_events": None, "errors": [f"Calendar auth failed: {e}"]}

    time_min, time_max = _today_window()

    events_out: list[dict[str, Any]] = []
    try:
        result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=50,
            )
            .execute()
        )
        for ev in result.get("items", []):
            start = ev.get("start", {})
            end = ev.get("end", {})
            events_out.append(
                {
                    "summary": ev.get("summary", "(no title)"),
                    "start": start.get("dateTime") or start.get("date"),
                    "end": end.get("dateTime") or end.get("date"),
                    "location": ev.get("location"),
                    "description": ev.get("description"),
                }
            )
    except Exception as e:
        errors.append(f"Calendar list failed: {e}")

    return {"calendar_events": events_out, "errors": errors}
