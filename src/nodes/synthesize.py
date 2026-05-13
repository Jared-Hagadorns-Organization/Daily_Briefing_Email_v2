"""Synthesize node: turns state into a clean HTML email body.

Strategy: pre-render the structured sections (stocks tables, sports, calendar,
recipes) deterministically in Python so the data is always correct, and let
the LLM write only the connective prose: greeting, weather/calendar lead-in,
news prose summary, and sign-off.
"""
from __future__ import annotations
import json
from typing import Any

from ..llm import get_llm


# =============================================================================
# Deterministic section renderers
# =============================================================================

def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.2f}%"


def _fmt_price(x: float | None) -> str:
    return "—" if x is None else f"${x:,.2f}"


def _fmt_volume_ratio(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}×"


def _render_stocks(stocks: dict[str, Any] | None) -> str:
    if not stocks:
        return "<p><em>Stocks data unavailable today.</em></p>"

    parts: list[str] = ["<h2>📈 Markets</h2>"]

    grades = stocks.get("grading_yesterday") or []
    parts.append("<h3>Yesterday's predictions, graded</h3>")
    parts.append(f"<p><strong>{stocks.get('grading_summary', '')}</strong></p>")
    if grades:
        rows = "".join(
            f"<tr>"
            f"<td>{g['ticker']}</td>"
            f"<td>{_fmt_pct(g.get('actual_return'))}</td>"
            f"<td>{_fmt_pct(g.get('spy_return'))}</td>"
            f"<td>{_fmt_pct(g.get('alpha'))}</td>"
            f"<td>{'✅' if g.get('correct') else ('❌' if g.get('correct') is False else '—')}</td>"
            f"</tr>"
            for g in grades
        )
        parts.append(
            "<table><tr><th>Ticker</th><th>Return</th><th>SPY</th>"
            f"<th>Alpha</th><th>Beat SPY?</th></tr>{rows}</table>"
        )

    top = stocks.get("movers_top") or []
    bot = stocks.get("movers_bottom") or []
    parts.append("<h3>Yesterday's top movers (S&P 500)</h3>")
    if top or bot:
        gain_rows = "".join(
            f"<tr><td>{m['ticker']}</td><td>{_fmt_pct(m['pct_change'])}</td>"
            f"<td>{_fmt_price(m.get('price'))}</td>"
            f"<td>{_fmt_volume_ratio(m.get('volume_ratio'))}</td></tr>"
            for m in top
        )
        loss_rows = "".join(
            f"<tr><td>{m['ticker']}</td><td>{_fmt_pct(m['pct_change'])}</td>"
            f"<td>{_fmt_price(m.get('price'))}</td>"
            f"<td>{_fmt_volume_ratio(m.get('volume_ratio'))}</td></tr>"
            for m in bot
        )
        parts.append(
            f"<h4>Gainers</h4><table><tr><th>Ticker</th><th>Change</th>"
            f"<th>Price</th><th>Vol vs 20d</th></tr>{gain_rows}</table>"
            f"<h4>Losers</h4><table><tr><th>Ticker</th><th>Change</th>"
            f"<th>Price</th><th>Vol vs 20d</th></tr>{loss_rows}</table>"
        )

    watch = stocks.get("watchlist_today") or []
    if watch:
        parts.append("<h3>Watchlist today (events / news)</h3><ul>")
        for w in watch:
            label = "📊 Earnings" if w.get("event_type") == "earnings" else "📰 News"
            link = f' — <a href="{w["url"]}">link</a>' if w.get("url") else ""
            parts.append(
                f'<li>{label} · <strong>{w["ticker"]}</strong>: '
                f"{w.get('summary', '')}{link}</li>"
            )
        parts.append("</ul>")

    preds = stocks.get("predictions_today") or []
    if preds:
        parts.append("<h3>Projected outperformers vs SPY today</h3><ol>")
        for p in preds:
            f = p["factor_summary"]
            parts.append(
                f"<li><strong>{p['ticker']}</strong> "
                f'<span style="color:#666">(confidence: {p["confidence"]})</span>'
                f"<br>{p['reasoning']}"
                f'<br><span style="font-size:0.9em;color:#666">'
                f"5d mom {f['momentum_5d_pct']:+.2f}%, "
                f"vol {f['volume_ratio_vs_20d']:.2f}×, "
                f"{f['distance_from_52w_high_pct']:+.2f}% from 52w high, "
                f"RSI-14 {f['rsi_14']:.1f}, "
                f"score {f['composite_score']:+.2f}</span></li>"
            )
        parts.append("</ol>")

    return "\n".join(parts)


def _render_news_items(news_items: list[dict[str, Any]]) -> str:
    if not news_items:
        return ""
    topic_labels = {"national": "National", "finance": "Finance", "tech": "Tech"}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in news_items:
        topic = item.get("topic", "national")
        grouped.setdefault(topic, []).append(item)

    parts = []
    for topic in ["national", "finance", "tech"]:
        if topic not in grouped:
            continue
        parts.append(f"<h3>{topic_labels.get(topic, topic.title())}</h3><ul>")
        for item in grouped[topic]:
            url = item.get("url", "")
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            link = f'<a href="{url}">{headline}</a>' if url else headline
            parts.append(f"<li>{link}<br><span style='font-size:0.9em'>{summary}</span></li>")
        parts.append("</ul>")
    return "\n".join(parts)


def _parse_score(score: Any) -> str:
    if score is None:
        return ""
    if isinstance(score, dict):
        return str(score.get("displayValue") or score.get("value") or "")
    return str(score)


def _render_sports(sports: list[dict[str, Any]] | None) -> str:
    if not sports:
        return ""
    parts: list[str] = ["<h2>🏈 Sports</h2>"]
    for team in sports:
        parts.append(f"<h3>{team['team']} ({team['league']})</h3>")
        last = team.get("last_game")
        if last:
            score = ""
            if last.get("home_score") and last.get("away_score"):
                score = f" — Final: {last['away_team']} {_parse_score(last['away_score'])}, {last['home_team']} {_parse_score(last['home_score'])}"
            parts.append(f"<p><em>Last:</em> {last.get('short_name', '')}{score}</p>")
        nxt = team.get("next_game")
        if nxt:
            parts.append(
                f"<p><em>Next:</em> {nxt.get('short_name', '')} · "
                f"{nxt.get('date', '')[:10]}"
                f"{(' · ' + nxt['venue']) if nxt.get('venue') else ''}</p>"
            )
        news = team.get("news") or []
        if news:
            parts.append("<ul>")
            for n in news[:3]:
                link = n["url"] if n.get("url") else "#"
                parts.append(
                    f'<li><a href="{link}"><strong>{n["headline"]}</strong></a>'
                    f"<br><span style=\"font-size:0.9em\">{n.get('summary', '')}</span></li>"
                )
            parts.append("</ul>")
    return "\n".join(parts)


def _render_calendar(events: list[dict[str, Any]] | None) -> str:
    if not events:
        return "<h2>📅 Calendar</h2><p>Nothing on the calendar today.</p>"
    rows = []
    for ev in events:
        start = (ev.get("start") or "")[11:16] if "T" in (ev.get("start") or "") else "All day"
        loc = f" — {ev['location']}" if ev.get("location") else ""
        rows.append(f"<li><strong>{start}</strong> · {ev['summary']}{loc}</li>")
    return "<h2>📅 Calendar</h2><ul>" + "".join(rows) + "</ul>"


def _render_recipes(recipes: list[dict[str, Any]] | None) -> str:
    if not recipes:
        return ""
    parts = ["<h2>🍳 This week's high-protein recipes</h2>"]
    for r in recipes:
        macros = r.get("macros", {})
        parts.append(f"<h3>{r['title']} <span style=\"color:#666;font-size:0.85em\">({r.get('protein_source', '')})</span></h3>")
        parts.append(
            f"<p><em>{r.get('prep_minutes', '?')} min · "
            f"{macros.get('calories', '?')} kcal · "
            f"{macros.get('protein_g', '?')}g protein · "
            f"{macros.get('carbs_g', '?')}g carbs · "
            f"{macros.get('fat_g', '?')}g fat</em></p>"
        )
        ing = "".join(f"<li>{i}</li>" for i in r.get("ingredients", []))
        steps = "".join(f"<li>{s}</li>" for s in r.get("instructions", []))
        parts.append(f"<details><summary>Ingredients & steps</summary><ul>{ing}</ul><ol>{steps}</ol></details>")
    return "\n".join(parts)


# =============================================================================
# Synthesizer entry point
# =============================================================================

def synthesize_node(state: dict) -> dict:
    date = state.get("date", "")
    weather = state.get("weather")
    news_items = state.get("news_items") or []

    llm = get_llm(temperature=0.4)
    intro_prompt = (
        f"Write a brief 2-3 sentence personal greeting for a daily briefing email. "
        f"Today is {state.get('day_of_week')}, {date}. "
        f"Weather snapshot: {json.dumps(weather)}. "
        f"Use a warm but concise tone. Plain HTML only (use <p>, no <html>/<body>). "
        f"Mention the day's weather naturally."
    )
    try:
        intro_html = llm.invoke(intro_prompt).content
    except Exception:
        intro_html = f"<p>Good morning — here's your briefing for {date}.</p>"

    local_events_html = state.get("local_events_html") or ""
    if news_items or local_events_html:
        news_html = "<h2>📰 News</h2>"
        if news_items:
            news_html += _render_news_items(news_items)
        if local_events_html:
            news_html += local_events_html
    else:
        news_html = ""

    body = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
            max-width:720px;margin:0 auto;color:#222;line-height:1.5">
  <h1>Daily Briefing — {date}</h1>
  {intro_html}
  {_render_calendar(state.get('calendar_events'))}
  {_render_stocks(state.get('stocks'))}
  {_render_sports(state.get('sports'))}
  {news_html}
  {_render_recipes(state.get('recipes'))}
  <hr>
  <p style="font-size:0.8em;color:#888">Generated by Daily_Briefing_Email · LangGraph + GPT-5-mini</p>
</div>
""".strip()

    subject = f"Daily Briefing — {date}"
    return {"email_subject": subject, "email_body_html": body, "errors": []}
