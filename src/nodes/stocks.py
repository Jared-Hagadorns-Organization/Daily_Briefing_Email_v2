"""Stocks node: 4-stage pipeline.

  1. Grade yesterday's predictions vs SPY (loads from history file).
  2. Yesterday's top 5 gainers + top 5 losers from the S&P 500.
  3. Today's watchlist: tickers with earnings today or significant news in 24h.
  4. Today's projected outperformers via a transparent factor screen,
     narrated by the LLM (selection is rule-based; LLM only explains).

Predictions are appended to `data/stock_predictions_history.json`. The Action
commits this file back so tomorrow's run can grade today's picks.

Factor menu (z-scored composite):
  + 1.0 * 5-day momentum
  + 0.5 * volume ratio (yesterday vs 20d avg)
  - 0.7 * distance from 52-week high  (closer to high contributes positively)
  + 0.4 * RSI mean-reversion bonus when oversold (RSI_14 < 35)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from ..llm import get_llm
from ..persistence import append_history, load_history


PREDICTIONS_FILE = "data/stock_predictions_history.json"
SPY = "SPY"

EVENT_SCAN_DEPTH = 75


# =============================================================================
# Universe
# =============================================================================

def _sp500_universe() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    return [t.replace(".", "-") for t in df["Symbol"].astype(str).tolist()]


# =============================================================================
# Bulk price download
# =============================================================================

def _download_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="column",
        auto_adjust=True,
        threads=True,
        progress=False,
    )


# =============================================================================
# Stage 1: Grade yesterday's predictions
# =============================================================================

def _grade_yesterday(
    history: list[dict[str, Any]], universe_data: pd.DataFrame
) -> tuple[list[dict[str, Any]], str]:
    if not history:
        return [], "No prior predictions on file — grading starts after the first run."

    last_batch = history[-1]
    predicted_date = last_batch.get("date", "?")
    picks = last_batch.get("picks", [])
    if not picks:
        return [], "Last prediction batch was empty."

    closes = universe_data["Close"]
    if SPY not in closes.columns:
        return [], "SPY data unavailable; cannot compute alpha."

    spy_returns = closes[SPY].pct_change()
    if len(spy_returns) < 1 or pd.isna(spy_returns.iloc[-1]):
        return [], "Not enough recent price data to grade yet."

    spy_return = float(spy_returns.iloc[-1])
    grades: list[dict[str, Any]] = []
    alphas: list[float] = []
    correct_count = 0

    for pick in picks:
        ticker = pick["ticker"]
        if ticker not in closes.columns:
            grades.append(
                {
                    "ticker": ticker,
                    "predicted_date": predicted_date,
                    "actual_return": None,
                    "spy_return": spy_return,
                    "alpha": None,
                    "correct": None,
                    "note": "Ticker missing from current pull.",
                }
            )
            continue

        actual = closes[ticker].pct_change().iloc[-1]
        if pd.isna(actual):
            grades.append(
                {
                    "ticker": ticker,
                    "predicted_date": predicted_date,
                    "actual_return": None,
                    "spy_return": spy_return,
                    "alpha": None,
                    "correct": None,
                    "note": "Missing close.",
                }
            )
            continue

        actual = float(actual)
        alpha = actual - spy_return
        correct = alpha > 0
        if correct:
            correct_count += 1
        alphas.append(alpha)
        grades.append(
            {
                "ticker": ticker,
                "predicted_date": predicted_date,
                "actual_return": actual,
                "spy_return": spy_return,
                "alpha": alpha,
                "correct": correct,
            }
        )

    rated = [g for g in grades if g.get("alpha") is not None]
    if rated:
        avg_alpha = sum(alphas) / len(alphas)
        summary = (
            f"{correct_count}/{len(rated)} beat SPY · "
            f"avg alpha {avg_alpha * 100:+.2f}% · "
            f"SPY {spy_return * 100:+.2f}%"
        )
    else:
        summary = "Could not grade — actual returns unavailable."

    return grades, summary


# =============================================================================
# Stage 2: Yesterday's movers
# =============================================================================

def _yesterday_movers(
    universe_data: pd.DataFrame, top_n: int = 5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    closes = universe_data["Close"]
    volumes = universe_data["Volume"]

    last_returns = closes.pct_change().iloc[-1]
    avg_vol_20 = volumes.rolling(20).mean().iloc[-2]
    last_vol = volumes.iloc[-1]
    last_close = closes.iloc[-1]

    valid = last_returns.dropna()
    if SPY in valid.index:
        valid = valid.drop(index=SPY)

    gainers = valid.nlargest(top_n)
    losers = valid.nsmallest(top_n)

    def _row(ticker: str) -> dict[str, Any]:
        vol_ratio: float | None = None
        if (
            ticker in last_vol.index
            and ticker in avg_vol_20.index
            and pd.notna(avg_vol_20[ticker])
            and avg_vol_20[ticker] > 0
        ):
            vol_ratio = float(last_vol[ticker] / avg_vol_20[ticker])

        price: float | None = None
        if ticker in last_close.index and pd.notna(last_close[ticker]):
            price = float(last_close[ticker])

        return {
            "ticker": ticker,
            "pct_change": float(valid[ticker]),
            "price": price,
            "volume_ratio": vol_ratio,
        }

    return [_row(t) for t in gainers.index], [_row(t) for t in losers.index]


# =============================================================================
# Stage 3: Today's watchlist (earnings + 24h news)
# =============================================================================

def _todays_events(
    universe_data: pd.DataFrame, top_n: int = 5, scan_depth: int = EVENT_SCAN_DEPTH
) -> list[dict[str, Any]]:
    closes = universe_data["Close"]
    volumes = universe_data["Volume"]

    last_returns = closes.pct_change().iloc[-1].abs()
    avg_vol_20 = volumes.rolling(20).mean().iloc[-2]
    vol_ratio = (volumes.iloc[-1] / avg_vol_20).replace([float("inf"), float("-inf")], 0).fillna(0)

    combined = (last_returns.fillna(0) * vol_ratio).dropna()
    if SPY in combined.index:
        combined = combined.drop(index=SPY)
    candidates = combined.nlargest(scan_depth).index.tolist()

    today = datetime.now(timezone.utc).date()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    found: list[dict[str, Any]] = []

    for ticker in candidates:
        try:
            t = yf.Ticker(ticker)
        except Exception:
            continue

        try:
            cal = t.calendar
            edate = None
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if ed:
                    edate = ed[0] if isinstance(ed, list) else ed
            elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
                edate = cal.loc["Earnings Date"].iloc[0]

            if edate is not None and hasattr(edate, "date") and edate.date() == today:
                found.append(
                    {
                        "ticker": ticker,
                        "event_type": "earnings",
                        "summary": "Earnings report scheduled today",
                        "score": 3.0,
                    }
                )
                continue
        except Exception:
            pass

        try:
            news = t.news or []
            for item in news[:3]:
                ts = item.get("providerPublishTime")
                if ts is None:
                    content = item.get("content") or {}
                    pub_iso = content.get("pubDate")
                    if pub_iso:
                        try:
                            published = datetime.fromisoformat(pub_iso.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                    else:
                        continue
                else:
                    published = datetime.fromtimestamp(ts, tz=timezone.utc)

                if published >= cutoff:
                    title = item.get("title") or (item.get("content") or {}).get("title", "")
                    link = item.get("link") or (
                        ((item.get("content") or {}).get("clickThroughUrl") or {}).get("url", "")
                    )
                    found.append(
                        {
                            "ticker": ticker,
                            "event_type": "news",
                            "summary": title,
                            "url": link,
                            "published": published.isoformat(),
                            "score": 1.5,
                        }
                    )
                    break
        except Exception:
            pass

    found.sort(key=lambda c: (-c["score"], c.get("published", "")))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in found:
        if c["ticker"] in seen:
            continue
        seen.add(c["ticker"])
        deduped.append(c)
        if len(deduped) >= top_n:
            break
    return deduped


# =============================================================================
# Stage 4: Project today's outperformers (rule-based screen + LLM narration)
# =============================================================================

def _compute_factors(universe_data: pd.DataFrame) -> pd.DataFrame:
    closes = universe_data["Close"]
    volumes = universe_data["Volume"]

    momentum_5d = closes.pct_change(5).iloc[-1]

    vol_20 = volumes.rolling(20).mean().iloc[-2]
    vol_ratio = volumes.iloc[-1] / vol_20.replace(0, pd.NA)

    high_52w = closes.rolling(252, min_periods=60).max().iloc[-1]
    dist_52w = (closes.iloc[-1] - high_52w) / high_52w

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi_14 = (100 - 100 / (1 + rs)).iloc[-1]

    df = pd.DataFrame(
        {
            "momentum_5d": momentum_5d,
            "volume_ratio": vol_ratio,
            "distance_from_52w_high": dist_52w,
            "rsi_14": rsi_14,
        }
    ).dropna()

    if SPY in df.index:
        df = df.drop(index=SPY)

    z = (df - df.mean()) / df.std(ddof=0)

    rsi_signal = pd.Series(0.0, index=df.index)
    oversold = df["rsi_14"] < 35
    if oversold.any():
        rsi_signal.loc[oversold] = -z.loc[oversold, "rsi_14"]

    df["composite"] = (
        z["momentum_5d"]
        + 0.5 * z["volume_ratio"]
        - 0.7 * z["distance_from_52w_high"]
        + 0.4 * rsi_signal
    )
    return df


def _narrate_picks(picks: pd.DataFrame, today: str) -> list[dict[str, Any]]:
    llm = get_llm(temperature=0.4)

    rows: list[dict[str, Any]] = []
    for ticker, row in picks.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "momentum_5d_pct": round(float(row["momentum_5d"]) * 100, 2),
                "volume_ratio_vs_20d": round(float(row["volume_ratio"]), 2),
                "distance_from_52w_high_pct": round(float(row["distance_from_52w_high"]) * 100, 2),
                "rsi_14": round(float(row["rsi_14"]), 1),
                "composite_score": round(float(row["composite"]), 2),
            }
        )

    prompt = (
        f"Date: {today}\n"
        "You are explaining a factor-based screen's top 5 picks for today.\n"
        "Factors used: 5-day momentum, volume ratio vs 20d average, distance "
        "from 52-week high (negative = below high), and RSI-14. The screen "
        "rewards momentum, volume confirmation, proximity to 52w highs, and "
        "oversold RSI mean-reversion.\n"
        "For each ticker, write 1-2 sentences explaining what the factors say. "
        "Stick to the numbers — do NOT add fundamental analysis, news takes, "
        "or price targets. Be concise and specific.\n\n"
        f"Picks:\n{json.dumps(rows, indent=2)}\n\n"
        'Return JSON: {"picks": [{"ticker": str, "reasoning": str, '
        '"confidence": "low|medium|high"}]}'
    )

    schema = {
        "title": "Picks",
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["ticker", "reasoning", "confidence"],
                },
            }
        },
        "required": ["picks"],
    }

    try:
        response = llm.with_structured_output(schema).invoke(prompt)
        narrated = {p["ticker"]: p for p in response.get("picks", [])}
    except Exception:
        narrated = {}

    out: list[dict[str, Any]] = []
    for ticker, row in picks.iterrows():
        n = narrated.get(ticker, {})
        out.append(
            {
                "ticker": ticker,
                "factor_summary": {
                    "momentum_5d_pct": round(float(row["momentum_5d"]) * 100, 2),
                    "volume_ratio_vs_20d": round(float(row["volume_ratio"]), 2),
                    "distance_from_52w_high_pct": round(
                        float(row["distance_from_52w_high"]) * 100, 2
                    ),
                    "rsi_14": round(float(row["rsi_14"]), 1),
                    "composite_score": round(float(row["composite"]), 2),
                },
                "reasoning": n.get("reasoning", "Factor-driven pick (narration unavailable)."),
                "confidence": n.get("confidence", "medium"),
            }
        )
    return out


# =============================================================================
# Helpers
# =============================================================================

def _is_weekday(date_str: str) -> bool:
    return datetime.strptime(date_str, "%Y-%m-%d").weekday() < 5


# =============================================================================
# Node entry point
# =============================================================================

def stocks_node(state: dict) -> dict:
    today = state["date"]
    errors: list[str] = []

    try:
        tickers = _sp500_universe()
    except Exception as e:
        errors.append(f"S&P 500 universe fetch failed: {e}")
        return {"stocks": None, "errors": errors}

    universe = sorted(set(tickers) | {SPY})

    try:
        data = _download_history(universe, period="1y")
    except Exception as e:
        errors.append(f"Bulk price download failed: {e}")
        return {"stocks": None, "errors": errors}

    history = load_history(PREDICTIONS_FILE)
    try:
        grades, grading_summary = _grade_yesterday(history, data)
    except Exception as e:
        errors.append(f"Grading failed: {e}")
        grades, grading_summary = [], "Grading skipped due to error."

    try:
        gainers, losers = _yesterday_movers(data, top_n=5)
    except Exception as e:
        errors.append(f"Movers calc failed: {e}")
        gainers, losers = [], []

    watchlist: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    if _is_weekday(today):
        try:
            watchlist = _todays_events(data, top_n=5)
        except Exception as e:
            errors.append(f"Watchlist build failed: {e}")

        try:
            factor_df = _compute_factors(data)
            top_picks = factor_df.nlargest(5, "composite")
            predictions = _narrate_picks(top_picks, today)
        except Exception as e:
            errors.append(f"Prediction screen failed: {e}")

        if predictions:
            try:
                append_history(
                    PREDICTIONS_FILE,
                    {
                        "date": today,
                        "picks": [
                            {"ticker": p["ticker"], "factor_summary": p["factor_summary"]}
                            for p in predictions
                        ],
                    },
                )
            except Exception as e:
                errors.append(f"History save failed: {e}")
    else:
        grading_summary = (grading_summary + " · Markets closed today.").strip(" ·")

    return {
        "stocks": {
            "movers_top": gainers,
            "movers_bottom": losers,
            "watchlist_today": watchlist,
            "predictions_today": predictions,
            "grading_yesterday": grades,
            "grading_summary": grading_summary,
        },
        "errors": errors,
    }
