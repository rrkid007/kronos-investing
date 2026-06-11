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

- **Phase 5 (News + SEC Filing agents)** — complete: Ollama client with
  grammar-constrained JSON (pydantic schema as Ollama `format`), validation
  retry, and hard LLMError → neutral fallback. News: yfinance + Finnhub
  ingestion, headline-hash dedupe, LLM classification where every key driver
  must cite a real headline index (ungrounded drivers dropped; no grounding
  caps confidence at 0.4). SEC: edgartools 10-K/10-Q section extraction, LLM
  risk findings that must quote the filing verbatim (ungrounded findings
  dropped), analyses cached per accession number so each filing hits the LLM
  once. Dev LLM: any local Ollama model (tested with gpt-oss:20b); DGX target
  is qwen2.5:32b-instruct per config.

- **Phase 6 (Decision + Portfolio agents)** — complete: confidence-weighted
  aggregation (`Σ w·c·s / Σ w·c`) where dead signals drop out instead of
  dragging the score to 50, a coverage floor that blocks entries on thin
  signal, the full exit policy (stop-loss → take-profit → max-hold →
  score-decay, in priority order; held tickers re-evaluated every run even
  off-watchlist), and greedy portfolio sizing (base % of equity scaled by
  score and inverse vol, trimmed by position cap, sector room, and the cash
  reserve floor). Daily report now leads with the decision table. Also fixed
  live: Yahoo's intraday partial bars (which carry the prior session's open)
  are dropped and purged — daily analysis only ever sees completed bars.

- **Phase 7 (Risk Engine)** — complete: pure-Python deterministic safety
  layer (no AI) that independently re-derives every constraint rather than
  trusting upstream sizing — final-score floor, per-agent floors (config
  `min_agent_scores`, e.g. never buy against a deeply negative Kronos
  forecast; dead agents are not blocking), restricted assets, position/sector
  caps, cash sufficiency, and the reserve floor. All violations are named
  (not just the first), every evaluation lands in `risk_events`, and sells
  are exempt from exposure rules by design — blocking a stop-loss is itself
  a risk. 30 dedicated tests covering pass/fail/boundary per rule.

- **Phase 8 (Paper Trading + approval queue)** — complete: full order
  lifecycle (awaiting_approval → approved/rejected/expired → filled) with
  UNIQUE(run_id, ticker, side) idempotency; T+1-open fills with slippage and
  commission; weighted-average cost positions; realized/unrealized P&L;
  daily mark-to-market snapshots. Each run starts by expiring stale
  unapproved orders and filling approved ones at today's open BEFORE
  analysis, so decisions always see post-fill positions and cash.
  `scripts/approve_trades.py` lists/approves/rejects the queue; the daily
  report shows the account and pending orders.

- **Phase 9 (orchestrator + deployment)** — complete: market-data downloads
  fan out across a thread pool (workers get their own SQLite connections;
  all writes stay on the main thread), fundamentals analysis runs parallel
  across tickers, GPU/LLM stages stay serial by design (one model, one
  Ollama). ntfy-compatible notifications fire on every run — success summary
  (decisions, equity, pending approvals) or failure with the error — and the
  report gained a Run Health section (duration, stage failures, data
  freshness). `deploy/` has the DGX kit: systemd service + timer (17:30 ET
  weekdays, `Persistent=true` catch-up), model prefetch script, and
  [SETUP.md](deploy/SETUP.md).

- **Phase 10 (performance analytics)** — complete: equity-curve metrics
  (total/annualized return, Sharpe at zero risk-free over 252d, max
  drawdown), closed-trade reconstruction from fills (weighted-average cost,
  win rate, avg return), SPY/QQQ benchmark comparison over the live window
  (benchmarks auto-maintained in the price cache each run), and per-signal
  hit rates (S6): every directional call (score ≥60 or ≤40, confidence > 0)
  graded against the 10-day forward return — the evidence base for future
  weight tuning. Reports now have a machine-readable JSON twin and a
  Performance section once history accumulates; `scripts/show_performance.py`
  prints the summary. Every metric is pinned by hand-computed test values,
  including a simulated 21-day month.

- **Phase 11 (dashboard)** — complete: FastAPI + Jinja2 + HTMX, server-
  rendered terminal aesthetic, zero build step. Account KPIs, server-rendered
  SVG equity curve, performance + per-signal hit rates, the approval queue
  with one-click approve/reject (HTMX partial swaps; the only write path),
  open positions with live unrealized P&L, decisions with risk verdicts,
  the agent score leaderboard (zero-confidence signals visibly dotted), run
  health, and a report browser. `uv run python scripts/run_dashboard.py`
  → http://127.0.0.1:8420 (localhost-only by default; no auth layer).

- **Phase 12 (backtester)** — complete: day-by-day historical replay driving
  the *same* TechnicalAgent / DecisionEngine / PortfolioAgent / RiskEngine /
  order / fill / snapshot code the live pipeline uses, against a scratch
  SQLite db that becomes a full per-backtest audit trail. PIT-safe by
  construction: only technical (+ opt-in kronos) replays; agents see history
  sliced to each simulated day; orders auto-approve and fill at next open.
  Sensitivity sweep (weight split or entry threshold) produces a comparison
  table; reports state every assumption. First real 2-year result: technical-
  only is flat (−0.5%, 42% win rate) vs SPY +39% — evidence the multi-signal
  blend has to earn its keep, exactly what the platform exists to measure.
  `uv run python scripts/run_backtest.py --years 2 --sweep`

- **Phase 13 (Alpaca paper broker)** — complete: opt-in
  `execution.broker: alpaca_paper` routes approved orders to Alpaca's PAPER
  endpoint as market-on-open orders (submitted at approval time, so the
  T+1-open fill convention is preserved; `client_order_id` makes retries
  idempotent). The next run syncs real fill prices through the same
  accounting path as simulated fills and reconciles positions/cash against
  Alpaca, surfacing (never auto-fixing) drift. The paper base URL is
  hard-coded — live trading is structurally impossible, and a test enforces
  it. Schema migrated v1→v2 (broker columns) via the versioned migration
  path. Live round-trip verification awaits Alpaca paper keys (see
  [deploy/SETUP.md](deploy/SETUP.md) §6).

- **Phase 14 (Discovery Agent)** — complete: weekly screen of the S&P 500
  universe (Wikipedia constituents with GICS sectors, cached monthly) for
  watchlist candidates. Funnel: gap-sector-first candidate selection rotating
  through the universe, the standard quality gate + the SAME technical and
  fundamentals rubrics the daily pipeline uses, then ranking at live relative
  weights plus a sector-gap bonus, with a fundamentals quality floor blocking
  momentum junk. Suggestions are proposals only — `discover_stocks.py --add`
  is the human's pen (appends to watchlist.yaml, capacity-capped with the
  weakest incumbent flagged for replacement). Watchlist sector tags now use
  official GICS names — the live run caught a taxonomy mismatch ("Technology"
  vs GICS "Information Technology") that was mis-awarding gap bonuses.
  `uv run python scripts/discover_stocks.py`

**All planned phases (0–14) are complete.** See [PLAN.md](PLAN.md) §3 for the roadmap.
