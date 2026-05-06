"""Google Calendar + iCloud Calendar node.

Google setup (one-time):
  1. GCP Console -> APIs & Services -> Enable Google Calendar API.
  2. OAuth consent screen -> External, add yourself as test user.
  3. Credentials -> Create OAuth client ID -> Application type: Desktop app.
  4. Run scripts/get_refresh_token.py once locally to obtain a refresh_token.
  5. Store as GitHub secrets: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN

iCloud setup:
  1. appleid.apple.com -> Sign-In and Security -> App-Specific Passwords -> Generate one.
  2. Store as GitHub secrets: ICLOUD_USERNAME, ICLOUD_APP_PASSWORD

Either source is optional — if its secrets are missing it is silently skipped.
"""
from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _get_google_credentials() -> Credentials:
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
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    today = datetime.now(local_tz).date()
    start = datetime.combine(today, time.min, tzinfo=local_tz)
    end = datetime.combine(today, time.max, tzinfo=local_tz)
    return start.isoformat(), end.isoformat()


def _fetch_google_events(time_min: str, time_max: str) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    events_out: list[dict[str, Any]] = []
    try:
        creds = _get_google_credentials()
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        return [], [f"Google Calendar auth failed: {e}"]

    try:
        calendars = service.calendarList().list().execute().get("items", [])
    except Exception as e:
        return [], [f"Google Calendar list failed: {e}"]

    for cal in calendars:
        cal_id = cal["id"]
        cal_name = cal.get("summary", cal_id)
        try:
            result = (
                service.events()
                .list(
                    calendarId=cal_id,
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
                events_out.append({
                    "summary": ev.get("summary", "(no title)"),
                    "start": start.get("dateTime") or start.get("date"),
                    "end": end.get("dateTime") or end.get("date"),
                    "location": ev.get("location"),
                    "description": ev.get("description"),
                    "calendar": cal_name,
                })
        except Exception as e:
            errors.append(f"Google Calendar '{cal_name}' fetch failed: {e}")

    return events_out, errors


def _fetch_icloud_events(time_min: str, time_max: str) -> tuple[list[dict], list[str]]:
    import caldav
    from caldav.elements import dav

    errors: list[str] = []
    events_out: list[dict[str, Any]] = []

    username = os.environ.get("ICLOUD_USERNAME")
    password = os.environ.get("ICLOUD_APP_PASSWORD")
    if not username or not password:
        return [], []

    try:
        client = caldav.DAVClient(
            url="https://caldav.icloud.com",
            username=username,
            password=password,
        )
        principal = client.principal()
        calendars = principal.calendars()
    except Exception as e:
        return [], [f"iCloud Calendar auth failed: {e}"]

    start_dt = datetime.fromisoformat(time_min)
    end_dt = datetime.fromisoformat(time_max)

    for cal in calendars:
        try:
            cal_name = str(cal.name) if cal.name else "iCloud"
            events = cal.date_search(start=start_dt, end=end_dt, expand=True)
            for ev in events:
                vevent = ev.vobject_instance.vevent
                summary = str(getattr(vevent, "summary", None) and vevent.summary.value or "(no title)")
                start_val = vevent.dtstart.value if hasattr(vevent, "dtstart") else None
                end_val = vevent.dtend.value if hasattr(vevent, "dtend") else None
                location = str(vevent.location.value) if hasattr(vevent, "location") else None
                description = str(vevent.description.value) if hasattr(vevent, "description") else None
                events_out.append({
                    "summary": summary,
                    "start": start_val.isoformat() if start_val else None,
                    "end": end_val.isoformat() if end_val else None,
                    "location": location,
                    "description": description,
                    "calendar": cal_name,
                })
        except Exception as e:
            errors.append(f"iCloud Calendar '{getattr(cal, 'name', '?')}' fetch failed: {e}")

    return events_out, errors


def calendar_node(state: dict) -> dict:
    time_min, time_max = _today_window()
    all_events: list[dict] = []
    all_errors: list[str] = []

    if os.environ.get("GOOGLE_REFRESH_TOKEN"):
        events, errors = _fetch_google_events(time_min, time_max)
        all_events.extend(events)
        all_errors.extend(errors)

    if os.environ.get("ICLOUD_APP_PASSWORD"):
        events, errors = _fetch_icloud_events(time_min, time_max)
        all_events.extend(events)
        all_errors.extend(errors)

    all_events.sort(key=lambda e: e.get("start") or "")

    return {"calendar_events": all_events, "errors": all_errors}
