# Build Plan — Local AI Investment Research Platform

**Source spec:** [trading.md](trading.md)
**Status:** Greenfield rebuild (all prior agent work discarded)
**Target runtime:** NVIDIA DGX Spark (ARM64, DGX OS/Ubuntu, 128GB unified memory)
**Date:** 2026-06-11

---

## 1. Review of the Spec — Shortcomings & Improvements

The spec in trading.md is a solid product description, but it has gaps that will bite during implementation. Each item below is incorporated into the build plan.

### 1.1 Architecture corrections

| # | Issue | Improvement |
|---|-------|-------------|
| A1 | **Pipeline drawn as strictly sequential.** Technical, Kronos, Fundamentals, News, and SEC agents have no dependencies on each other. Running them in a chain wastes wall-clock time (LLM + Kronos inference dominate). | Fan-out: run the 5 research agents **in parallel per ticker**, then converge at Trade Decision → Portfolio fit → Risk Engine. Orchestrator uses `asyncio` + a process pool for GPU-bound work. |
| A2 | **No exit logic anywhere in the spec.** Only buy decisions are described. Without sell rules, the paper account fills up and never frees capital. | Add a **Position Management policy** to the Trade Decision Agent: stop-loss %, take-profit %, max holding period, score-decay exit (sell when a held ticker's final score drops below an exit threshold), and rebalancing rules. Every daily run re-evaluates *held* positions, not just the watchlist. |
| A3 | **No human approval mechanism**, despite "Human approves" being a core design pillar. | Add a **pending-trades approval queue**: Risk-approved trades land in `pending_orders` with status `awaiting_approval`; a CLI command and a dashboard view approve/reject; unapproved orders **auto-expire at next market open**. An optional config flag (`auto_approve: true`) allows fully automated paper trading once trusted. |
| A4 | **Idempotency / partial-failure handling unspecified.** A crashed or re-run daily job must not duplicate trades or scores. | Every pipeline execution gets a `run_id`; all writes are keyed by `(run_id, ticker)`; re-running a day upserts instead of inserting; the orchestrator records per-agent success/failure and can resume a partial run. |
| A5 | **No monitoring/alerting.** A silent cron failure means days of missing data. | Run-status table + structured logging + a notification hook (ntfy/email/webhook) on run completion or failure. The daily report includes "data freshness" health checks. |

### 1.2 Signal-quality corrections

| # | Issue | Improvement |
|---|-------|-------------|
| S1 | **Score semantics undefined.** What makes a Kronos "74" comparable to a fundamentals "78"? Static weights (35/25/20/10/10) over un-normalized scores are meaningless. | Define a **scoring contract**: every agent outputs `score ∈ [0,100]` with a documented mapping (e.g., percentile vs. ticker's own history, or fixed rubric), plus `confidence ∈ [0,1]`. The Decision Agent uses **confidence-weighted aggregation**, and weights live in config — not code. |
| S2 | **Backtester has look-ahead bias as specced.** yfinance fundamentals, news, and analyst data are *current snapshots*, not point-in-time. Backtesting the full 5-signal stack on them produces invalid results. | v1 backtester replays **only point-in-time-safe signals** (Technical + Kronos on historical OHLCV). Fundamentals/News/SEC backtesting is explicitly out of scope until point-in-time data exists; instead, those signals are validated **forward** via the paper-trading track record. The backtest report must state which signals were included. |
| S3 | **No data validation layer.** Bad OHLCV (missing days, unadjusted splits, stale quotes) silently corrupts every downstream score. | A data-quality gate after ingestion: split/dividend adjustment check, gap detection, staleness check, NaN/zero-volume screens. A ticker that fails validation is **skipped and flagged**, never scored on bad data. |
| S4 | **LLM outputs (News/SEC agents) are unguarded.** Free-text LLM scoring hallucinates and drifts. | LLM agents must return **schema-validated JSON** (retry on parse failure), cite the specific headline/filing excerpt supporting each claim ("grounding"), and fall back to `score=50, confidence=0` (neutral, ignorable) when sources are unavailable — never a guessed score. |
| S5 | **Position sizing unspecified.** "Position limits" exist but nothing says how large a new position is. | Config-driven sizing: fixed-fraction base (e.g., 10% of equity), scaled by final score and inverse volatility, capped by risk limits. Deterministic, in the Portfolio Agent. |
| S6 | **Static weights never validated.** | Backtester supports **weight sensitivity sweeps** (grid over weight vectors on PIT-safe signals) and the analytics module tracks per-signal hit rates from live paper trades, so weights can be tuned from evidence. |

### 1.3 Engineering corrections

| # | Issue | Improvement |
|---|-------|-------------|
| E1 | No configuration management; watchlist, weights, limits implied as code constants. | Single `config/` directory of YAML files (watchlist, weights, risk limits, schedule, model choices), validated by pydantic at startup. Changing the watchlist must never require a code change. |
| E2 | No tests mentioned. | pytest suite from phase 0; the Risk Engine and Paper Trading Engine (the money-touching code) get **exhaustive unit tests**; agents get golden-file tests on fixed fixtures. |
| E3 | Fill model unspecified — paper trades at what price? | Convention: decisions made after close on day T **fill at day T+1 open**, with configurable slippage (bps) and commission. This is also the backtester's fill model, so paper and backtest results are comparable. |
| E4 | SQLite under concurrent writes. | Keep SQLite (right choice for single-node) but enable WAL mode, use a single writer (the orchestrator), and wrap each run in a transaction. |
| E5 | News source unspecified — hard problem for a local system. | v1: yfinance news + RSS feeds (free, no key). Optional: Finnhub free tier. The News Agent must work degraded (neutral score) when no news is found. |
| E6 | Kronos integration details missing. | Use the open-source **Kronos** foundation model (NeoQuasar, HuggingFace) — `Kronos-base` + its tokenizer, max context 512 candles, PyTorch. Pin the model revision. Wrap in a service class so the model loads once per run, not per ticker. |
| E7 | Dev machine (Windows) ≠ deploy target (DGX Spark, ARM Linux). | Everything pure-Python and platform-neutral; `uv` for env management; a `deploy/` doc + systemd timer unit for the Spark (systemd timer > cron: logging, catch-up on missed runs). LLM inference via **Ollama** on the Spark (can serve Qwen2.5 32B-class models comfortably in 128GB unified memory); dev machine can point at the Spark's Ollama endpoint or a smaller local model. |

---

## 2. Target Architecture

```
                        ┌────────────────────────────────────────────┐
                        │              Orchestrator (run_id)         │
                        └────────────────────────────────────────────┘
 Market Data Layer ──► Data Quality Gate
        │
        │   per ticker, IN PARALLEL
        ├──────────┬──────────────┬───────────────┬─────────────┐
        ▼          ▼              ▼               ▼             ▼
   Technical    Kronos       Fundamentals      News          SEC Filing
     Agent      Agent          Agent           Agent           Agent
        └──────────┴──────┬───────┴───────────────┴─────────────┘
                          ▼
                 Trade Decision Agent  ◄── re-evaluates HELD positions too (exits)
                          ▼
                  Portfolio Agent (sizing, fit)
                          ▼
                   Risk Engine (deterministic, no AI)
                          ▼
              Pending Orders (human approval queue)*
                          ▼
                Paper Trading Engine (T+1 open fills)
                          ▼
                  SQLite (WAL, single writer)
                          ▼
        Reports / Analytics / Dashboard / Notifications

  * skippable via auto_approve config flag
```

**Module layout:**

```
trading/
├── config/
│   ├── settings.yaml          # paths, schedule, model endpoints, notification hook
│   ├── watchlist.yaml         # tickers + sector tags
│   ├── weights.yaml           # signal weights + thresholds
│   └── risk_limits.yaml       # position/sector/cash limits, approval policy
├── src/trading_platform/      # ("platform" collides with the Python stdlib module)
│   ├── core/                  # config loader, logging, run lifecycle, db engine
│   ├── data/                  # market_data.py, fundamentals.py, news.py, sec.py, quality.py
│   ├── agents/                # technical.py, kronos.py, fundamentals.py, news.py,
│   │                          # sec_filing.py, decision.py, portfolio.py  (common base: AgentResult)
│   ├── risk/                  # engine.py, rules.py
│   ├── execution/             # paper_broker.py, orders.py, approval.py, alpaca_broker.py (phase 13)
│   ├── analytics/             # performance.py, metrics.py (Sharpe, drawdown, win rate)
│   ├── backtest/              # replay.py, fills.py, benchmark.py
│   ├── reporting/             # daily_report.py, leaderboard.py
│   └── dashboard/             # FastAPI app + templates (HTMX) or small React app
├── scripts/
│   ├── run_daily_analysis.py
│   ├── approve_trades.py      # CLI approval queue
│   ├── run_backtest.py
│   └── init_db.py
├── db/                        # investment_research.sqlite (gitignored)
├── reports/                   # daily/, backtests/ (gitignored)
├── deploy/                    # systemd units, DGX Spark setup notes, Ollama model pulls
└── tests/
```

**Core data contracts (pydantic):**

- `AgentResult { agent, ticker, run_id, score: 0–100, confidence: 0–1, direction, details: dict, data_as_of: datetime }` — every agent returns exactly this shape.
- `TradeDecision { ticker, action: buy|sell|hold|watchlist, final_score, signal_breakdown, sizing_hint }`
- `Order { id, run_id, ticker, side, qty, status: awaiting_approval|approved|rejected|expired|filled, limit_notes }`

**SQLite schema (tables):** `runs`, `agent_scores`, `decisions`, `orders`, `fills`, `positions`, `account_snapshots`, `price_cache`, `news_items`, `filings`, `risk_events`.

**Tech stack:** Python 3.12, `uv`, pydantic v2, pandas, yfinance, `edgartools` (SEC EDGAR), Kronos (HuggingFace/PyTorch), Ollama (local LLM for News/SEC reasoning), FastAPI + HTMX dashboard, APScheduler for dev / systemd timer for the Spark, pytest.

---

## 3. Build Phases

Ordered so every phase produces something runnable and testable, money-touching code gets built deterministically before any AI is wired in, and the LLM/GPU work (highest uncertainty) doesn't block the rest.

### Phase 0 — Foundations (scaffolding)
- Repo layout above; `uv` project; pydantic config loader for all YAML; structured logging.
- SQLite schema + `init_db.py` + migration convention; WAL mode; single-writer discipline.
- `AgentResult` / `TradeDecision` / `Order` models; agent base class.
- Run lifecycle: `run_id` creation, per-agent status tracking, resume-on-partial-failure.
- pytest harness + CI-style local check script (`ruff`, `pytest`).
- **Done when:** `init_db.py` builds the schema; a no-op pipeline run writes a `runs` row and a report stub.

### Phase 1 — Market Data Layer + Quality Gate
- OHLCV ingestion via yfinance with on-disk price cache (incremental daily updates into `price_cache`).
- Corporate-action adjustment verification; gap/staleness/NaN screens (`data/quality.py`).
- Ticker failure → skip + flag, never score.
- **Done when:** full watchlist refreshes idempotently; deliberately corrupted fixture data is caught by the gate.

### Phase 2 — Technical Agent (deterministic, no AI)
- 50/200 DMA, momentum (multi-window), realized volatility, max drawdown, trend strength (e.g., ADX or regression slope).
- Documented score rubric mapping metrics → 0–100; golden-file tests on fixture data.
- **Done when:** repeatable identical scores on fixed fixtures.

### Phase 3 — Fundamentals Agent
- yfinance fundamentals (income, balance sheet, cash flow, valuation ratios).
- Sub-scores (growth/profitability/balance sheet/cash flow/valuation) per spec, each with an explicit rubric; missing data → neutral sub-score with reduced confidence.
- **Done when:** all 10 watchlist tickers score with a stored breakdown and `data_as_of` stamps.

### Phase 4 — Kronos Forecast Agent
- Download + pin Kronos model/tokenizer; inference wrapper that loads once per run.
- Feed last N candles → forecast horizon (configurable, e.g., 10 trading days); derive `expected_return_pct`, direction, score, confidence (e.g., from sample-path dispersion).
- Benchmark inference time on the Spark; CPU fallback for dev.
- **Done when:** forecasts produced for the watchlist within an acceptable run budget; outputs stored as `AgentResult`.

### Phase 5 — News Agent + SEC Filing Agent (local LLM)
- Ollama client wrapper with: JSON-schema-enforced outputs, parse-retry, neutral-fallback (S4).
- News: yfinance news + RSS ingestion → dedupe → LLM classification (event types per spec) → grounded score with cited headlines.
- SEC: `edgartools` to pull latest 10-K/10-Q → section extraction (Risk Factors, MD&A) → LLM risk-term analysis → grounded score with cited excerpts.
- **Done when:** both agents run against live data; pulling the network cable yields neutral scores, not crashes.

### Phase 6 — Trade Decision + Portfolio Agents
- Confidence-weighted aggregation with config weights (S1); thresholds for buy/watchlist/hold from `weights.yaml`.
- **Exit policy (A2):** stop-loss, take-profit, max-hold, score-decay exit; held positions re-scored every run.
- Portfolio Agent: position sizing (S5), sector exposure, cash reserve, diversification checks.
- **Done when:** synthetic agent-score fixtures produce correct buy/sell/hold decisions and sizes in unit tests, including exit scenarios.

### Phase 7 — Risk Engine
- Pure deterministic rule evaluation from `risk_limits.yaml`: position/sector caps, cash minimum, score thresholds, asset restrictions, approval requirement.
- Returns structured pass/fail with every violated rule named; all evaluations logged to `risk_events`.
- **Exhaustive unit tests** — this is the safety layer; aim for full branch coverage.
- **Done when:** every rule has tests for pass, fail, and boundary cases.

### Phase 8 — Paper Trading Engine + Approval Queue
- Account ($100k start), positions, T+1-open fill model with slippage/commission config (E3).
- Orders lifecycle: `awaiting_approval → approved/rejected/expired → filled`; `approve_trades.py` CLI; auto-expiry at next open; `auto_approve` flag (A3).
- Daily `account_snapshots` (equity, cash, unrealized/realized P&L).
- **Done when:** a full simulated buy→hold→exit cycle round-trips correctly with exact P&L math verified in tests.

### Phase 9 — Orchestrator, Scanner & Scheduler
- `run_daily_analysis.py`: parallel agent fan-out (A1), held-position re-evaluation, ranking/leaderboard, decision → risk → order submission, idempotent re-runs (A4).
- Notification hook on success/failure (A5); data-freshness health summary.
- `deploy/`: systemd service + timer for the Spark (weekdays post-close, catch-up enabled), Ollama model pull script, setup notes.
- **Done when:** two consecutive runs of the same day produce no duplicates; a mid-run kill resumes cleanly.

### Phase 10 — Reporting + Performance Analytics
- Daily markdown + JSON reports (leaderboard, decisions, risk results, account state) to `reports/daily/`.
- Analytics: win rate, average return, Sharpe, max drawdown, vs SPY/QQQ benchmark; **per-signal hit-rate tracking** to inform weight tuning (S6).
- **Done when:** a month of simulated history renders correct metrics validated against hand-computed values.

### Phase 11 — Dashboard
- FastAPI + HTMX (server-rendered, lightweight — right-sized for a single-user local tool): leaderboard, positions & P&L, equity curve, pending-approvals view with approve/reject buttons, risk status, report browser, run health.
- Read-only against SQLite except the approval actions.
- **Done when:** the full daily workflow (review → approve → see fills next day) is doable entirely in the browser.

### Phase 12 — Historical Replay Backtester
- Replays **Technical + Kronos only** (PIT-safe, S2) over historical windows using the same decision/risk/fill code paths as live (no parallel reimplementation — reuse, so backtest fidelity = paper fidelity).
- Benchmark comparison vs SPY/QQQ; weight sensitivity sweep mode (S6).
- Report clearly labels included signals and assumptions.
- **Done when:** a 2-year backtest on the watchlist completes and a weight sweep produces a comparison table.

### Phase 13 — Alpaca Paper Broker Integration
- `Broker` interface extracted from the paper engine; `AlpacaPaperBroker` implementation (paper API keys only), order submission, position/fill sync, drift reconciliation between local DB and Alpaca state.
- Live trading remains out of scope, per spec.
- **Done when:** an approved order round-trips through Alpaca paper and reconciles with the local DB.

---

## 4. Build Order Rationale & Effort

- **Deterministic before AI:** Phases 2, 6, 7, 8 (the code that touches money) are fully testable without any model — they're built and locked down before LLM variability enters.
- **Highest-uncertainty items (4, 5)** are isolated behind the `AgentResult` contract; if Kronos or the LLM agents lag, the rest of the platform still ships with neutral scores in their slots.
- **Backtester after live pipeline (12)** so it reuses the identical decision/risk/fill code instead of a drifting reimplementation.

Rough effort (focused sessions): Phases 0–3 ≈ 4–6 days · Phase 4 ≈ 2–3 days · Phase 5 ≈ 3–4 days · Phases 6–8 ≈ 4–5 days · Phase 9 ≈ 2 days · Phases 10–11 ≈ 3–4 days · Phase 12 ≈ 2–3 days · Phase 13 ≈ 2 days. **Total ≈ 4–5 weeks part-time.**

---

## 5. Locked Decisions (confirmed 2026-06-11)

1. **LLM on the Spark:** Qwen2.5-32B-Instruct via Ollama.
2. **Kronos forecast horizon:** 10 trading days.
3. **Approval:** manual human approval required (`auto_approve: false`).
4. **Dashboard:** FastAPI + HTMX server-rendered.
5. **News sources:** yfinance + RSS, plus Finnhub free tier (API key via env var `FINNHUB_API_KEY`).
