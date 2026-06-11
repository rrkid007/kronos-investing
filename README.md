# Local AI Investment Research Platform

A fully local, AI-powered stock research and **paper-trading** system targeting an
NVIDIA DGX Spark. Multi-agent research (technical, Kronos forecast, fundamentals,
news, SEC filings) feeds a deterministic risk engine and a human approval queue.
No real-money trading.

**Design pillars:** Agents analyze. Risk engine controls. Human approves. Broker executes.

- Spec: [trading.md](trading.md)
- Build plan & architecture: [PLAN.md](PLAN.md)

## Setup

```bash
uv sync                              # install deps (Python >= 3.12)
uv run python scripts/init_db.py     # create/migrate the SQLite database
```

Secrets are environment variables, never config files:

```bash
export FINNHUB_API_KEY=...           # news coverage (free tier)
```

## Usage

```bash
uv run python scripts/run_daily_analysis.py            # analyze today
uv run python scripts/run_daily_analysis.py --date 2026-06-11
uv run pytest                                          # test suite
uv run ruff check .                                    # lint
```

Runs are idempotent: re-invoking the same date resumes an incomplete run and
never duplicates scores or trades.

## Configuration

Everything tunable lives in `config/` (validated at startup — bad config fails fast):

| File | Controls |
|------|----------|
| `settings.yaml` | paths, LLM endpoint (Ollama/Qwen2.5), Kronos model + horizon, news sources, schedule |
| `watchlist.yaml` | tickers + sector tags |
| `weights.yaml` | signal weights (must sum to 1.0), buy/watchlist/exit thresholds, exit policy |
| `risk_limits.yaml` | position/sector/cash limits, approval policy, paper account, fill model |

## Status

- **Phase 0 (foundations)** — complete: config system, SQLite schema (WAL,
  versioned migrations), agent data contracts, resumable run lifecycle,
  pipeline skeleton, test suite.
- **Phase 1 (market data + quality gate)** — complete: yfinance ingestion into
  the `price_cache` (full-window upsert so retroactive split/dividend
  adjustments are always correct), 8-check data-quality gate (history depth,
  NaN, price sanity, high/low consistency, date gaps, staleness, split
  artifacts, zero volume). Tickers failing the gate are skipped and flagged in
  the daily report — agents never score bad data.
  `scripts/refresh_market_data.py` refreshes the watchlist manually.

- **Phase 2 (Technical Agent)** — complete: deterministic pandas analysis on
  adjusted close with a documented 4-component rubric (trend structure 30,
  momentum 30, trend quality 20, risk 20). Golden-file tests pin exact scores;
  agent crashes are isolated to a neutral zero-confidence score so the run
  survives. The daily report now includes a live score leaderboard.

- **Phase 3 (Fundamentals Agent)** — complete: yfinance snapshot scored by an
  explicit worst→best anchor rubric across the spec's five sub-scores (growth,
  profitability, balance sheet, cash flow, valuation; 16 components total).
  Missing fields degrade to neutral sub-scores and reduce confidence
  (0.3 + 0.6 × coverage) — never guesses. Snapshots are current-only, never
  fed to the backtester (PLAN.md S2).

- **Phase 4 (Kronos Forecast Agent)** — complete: vendored the MIT-licensed
  Kronos model code (`src/trading_platform/vendor/kronos/`, patched to expose
  per-sample forecast paths), lazy one-load-per-run forecaster with
  cuda→mps→cpu auto-detect, and an agent that maps the mean 10-day forecast
  return onto the score (±8% anchors) with confidence from path agreement.
  Heavy deps are opt-in: `uv sync --extra kronos`. Without them the agent
  degrades to an ignorable neutral score. CPU benchmark: ~10s/ticker
  (8 paths, Kronos-small); switch `model_id` to Kronos-base on the DGX.

See [PLAN.md](PLAN.md) §3 for the phase roadmap. Next: Phase 5 (News + SEC Filing agents).
