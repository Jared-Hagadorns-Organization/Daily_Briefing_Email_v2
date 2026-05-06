"""LangGraph builder: init -> parallel branches -> synthesize -> send."""
from __future__ import annotations
import os
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from .state import BriefingState
from .nodes.weather import weather_node
from .nodes.calendar import calendar_node
from .nodes.stocks import stocks_node
from .nodes.sports import sports_node
from .nodes.news import news_node
from .nodes.recipes import recipes_node
from .nodes.synthesize import synthesize_node
from .nodes.send import send_node


def init_node(state: dict) -> dict:
    now = datetime.now()
    recipients = [
        r.strip()
        for r in os.environ.get("MAIL_RECIPIENTS", "").split(",")
        if r.strip()
    ]
    return {
        "date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
        "recipients": recipients,
        "errors": [],
    }


def build_graph():
    g = StateGraph(BriefingState)
    g.add_node("init", init_node)
    g.add_node("weather", weather_node)
    g.add_node("calendar", calendar_node)
    g.add_node("stocks", stocks_node)
    g.add_node("sports", sports_node)
    g.add_node("news", news_node)
    g.add_node("recipes", recipes_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("send", send_node)

    g.add_edge(START, "init")

    parallel = ["weather", "calendar", "stocks", "sports", "news", "recipes"]
    for branch in parallel:
        g.add_edge("init", branch)
        g.add_edge(branch, "synthesize")

    g.add_edge("synthesize", "send")
    g.add_edge("send", END)
    return g.compile()
