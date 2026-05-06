# Daily Briefing Email — LangGraph + GitHub Models

A daily personal briefing email assembled by a LangGraph pipeline. Runs free in GitHub Actions using `GITHUB_TOKEN` for LLM inference via GitHub Models (GPT-5-mini).

## Architecture

```
init → [weather, calendar, stocks, sports, news, recipes] → synthesize → send
```

Six branches run in parallel. Each branch writes its slice into shared state. The synthesizer pre-renders structured sections deterministically and only uses the LLM for connective prose (greeting, news section). Recipes only run on Sundays.

## What lives where

| Path | Purpose |
|---|---|
| `main.py` | Entry point — invokes the compiled graph |
| `src/graph.py` | Graph builder and `init_node` |
| `src/state.py` | Typed shared state |
| `src/llm.py` | GPT-5-mini client (GitHub Models) |
| `src/persistence.py` | JSON history file helpers |
| `src/nodes/weather.py` | OpenWeatherMap fetch |
| `src/nodes/calendar.py` | Google Calendar via OAuth refresh token |
| `src/nodes/stocks.py` | 4-stage pipeline: grade → movers → watchlist → predictions |
| `src/nodes/sports.py` | ESPN schedule + top 3 news for Panthers / Hornets / Hurricanes / Charlotte FC |
| `src/nodes/news.py` | ReAct agent w/ MCP tools (Tavily search + fetch) |
| `src/nodes/recipes.py` | Sunday-only high-protein recipes with weekly variety |
| `src/nodes/synthesize.py` | HTML email composition |
| `src/nodes/send.py` | SMTP delivery |
| `data/*.json` | Cross-run memory — Action commits these back |
| `.github/workflows/daily_briefing.yml` | Cron schedule + secrets wiring |
| `scripts/get_refresh_token.py` | One-time Google OAuth token helper |

## Required secrets

```
# LLM — auto-provided in Actions, no secret needed
# GITHUB_TOKEN

# Weather
OPENWEATHER_API_KEY

# Calendar (run scripts/get_refresh_token.py once)
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
GOOGLE_REFRESH_TOKEN

# News agent
TAVILY_API_KEY

# Email delivery
SMTP_HOST
SMTP_PORT
SMTP_USER
SMTP_PASSWORD
SMTP_FROM
MAIL_RECIPIENTS    # comma-separated
```

## One-time Calendar setup

1. GCP Console → APIs & Services → enable Google Calendar API
2. OAuth consent screen → External, add yourself as test user
3. Credentials → Create OAuth client ID → Desktop app → download JSON
4. `python scripts/get_refresh_token.py path/to/client_secret.json`
5. Copy the printed values into repo secrets

## Stocks pipeline notes

The factor screen is intentionally transparent and rule-based. Selection is *not* delegated to the LLM — the model only narrates why each pick made the cut. Composite score:

```
+ 1.0 * z(5-day momentum)
+ 0.5 * z(volume ratio vs 20-day avg)
- 0.7 * z(distance from 52-week high)
+ 0.4 * RSI mean-reversion bonus when oversold (RSI_14 < 35)
```

Predictions are written to `data/stock_predictions_history.json`. Each run grades the prior batch against SPY (alpha = pick return − SPY return; correct = alpha > 0). Over time this becomes a slow-running paper-trade backtest you can tune by editing the factor weights.

## Tweaking ideas

- **Universe**: swap S&P 500 in `_sp500_universe()` for Russell 1000 or sector subsets.
- **Factor menu**: edit `_compute_factors` and the weights in the composite. Add earnings revision direction, sector strength, short interest, etc.
- **Recipe protein bias**: edit the system prompt in `recipes.py` to bias toward specific cuisines or constraints.
- **Sports**: ESPN team slugs are easy to swap; add UNC/Duke basketball during March, etc.
