"""Shared LangGraph state for the daily briefing.

Each parallel branch writes into its own slot. The synthesizer reads the whole
state and composes the email body.
"""
from __future__ import annotations
from typing import TypedDict, Annotated
from operator import add


# ---- Stocks ----

class StockMover(TypedDict):
    ticker: str
    pct_change: float
    price: float | None
    volume_ratio: float | None


class StockEvent(TypedDict, total=False):
    ticker: str
    event_type: str  # "earnings" | "news"
    summary: str
    url: str
    published: str


class StockPrediction(TypedDict):
    ticker: str
    factor_summary: dict
    reasoning: str
    confidence: str  # "low" | "medium" | "high"


class StockGrade(TypedDict, total=False):
    ticker: str
    predicted_date: str
    actual_return: float | None
    spy_return: float | None
    alpha: float | None
    correct: bool | None
    note: str


class StockReport(TypedDict):
    movers_top: list[StockMover]
    movers_bottom: list[StockMover]
    watchlist_today: list[StockEvent]
    predictions_today: list[StockPrediction]
    grading_yesterday: list[StockGrade]
    grading_summary: str


# ---- Sports ----

class GameSummary(TypedDict, total=False):
    date: str
    name: str
    short_name: str
    venue: str | None
    home_team: str | None
    away_team: str | None
    home_score: str | None
    away_score: str | None
    winner: str | None  # "home" | "away" | "tie"


class NewsArticle(TypedDict, total=False):
    headline: str
    summary: str
    url: str
    published: str


class SportsTeamUpdate(TypedDict):
    team: str
    league: str
    last_game: GameSummary | None
    next_game: GameSummary | None
    news: list[NewsArticle]


# ---- Other ----

class CalendarEvent(TypedDict, total=False):
    summary: str
    start: str
    end: str
    location: str | None
    description: str | None


class NewsItem(TypedDict):
    topic: str
    headline: str
    summary: str
    url: str


class Recipe(TypedDict, total=False):
    title: str
    protein_source: str
    ingredients: list[str]
    instructions: list[str]
    prep_minutes: int
    macros: dict


# ---- Top-level state ----

class BriefingState(TypedDict, total=False):
    # Inputs (set by init_node)
    date: str
    day_of_week: str
    recipients: list[str]

    # Filled by parallel branches
    weather: dict | None
    calendar_events: list[CalendarEvent] | None
    stocks: StockReport | None
    sports: list[SportsTeamUpdate] | None
    news_items: list[NewsItem] | None
    local_events_html: str | None
    recipes: list[Recipe] | None

    # Output
    email_subject: str | None
    email_body_html: str | None

    # Parallel-safe error accumulator
    errors: Annotated[list[str], add]
