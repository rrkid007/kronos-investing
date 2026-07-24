# Build Plan — Local AI Crypto Research Platform

**Sibling spec:** [trading.md](trading.md) · [PLAN.md](PLAN.md) (equity platform this forks from)
**Status:** Greenfield fork of the equity platform — crypto-native rebuild of the asset-specific layers
**Target runtime:** NVIDIA DGX Spark (ARM64, DGX OS/Ubuntu, 128GB unified memory)
**Date:** 2026-06-22

---

## 0. Decision Record (this fork)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | **Fork** into a separate `crypto_platform` repo, not a multi-asset extension of the equity codebase | Speed and isolation. The equity system stays untouchable; fundamentals/SEC are deleted rather than abstracted around; the 24/7 fill model is rewritten cleanly instead of bolted onto an equity-shaped engine. Cost accepted: two codebases that can drift — mitigated by keeping the shared *core contracts* byte-identical so fixes port by copy. |
| **Execution / broker** | **CCXT exchange testnet / paper** (e.g. Binance Spot Testnet, with the same `Broker` interface able to target Coinbase/Kraken sandboxes) | Real crypto market structure — maker/taker fees, order types, partial fills, 24/7 books — without risking capital. Broader coin coverage than a single-broker API. Internal-sim fills remain the default/offline fallback. |
| **Signal stack** | **Full crypto-native stack** | Technical + Kronos + on-chain + tokenomics/unlocks + derivatives (funding/OI) + crypto news + crypto macro. Each isolated behind the `AgentResult` contract, so any lagging signal degrades to a neutral, ignorable score. |
| **No real-money trading** | Enforced structurally | Same pillar as the equity platform: **Agents analyze. Risk engine controls. Human approves. Broker executes (testnet only).** A test pins the broker to sandbox/testnet base URLs so live trading is impossible by construction. |

---

## 1. What Forks Cleanly vs. What Is Net-New

The equity platform's contracts and infrastructure are already asset-agnostic. This plan reuses them verbatim and rebuilds only the asset-specific layers.

### 1.1 Reused verbatim (copy from the equity repo, do not re-derive)

- **Core data contracts** — `AgentResult { asset, score 0-100, confidence 0-1, direction, details, data_as_of }`, `TradeDecision`, `Order`. (Rename the `ticker` field to `asset`/`symbol`; otherwise unchanged.)
- **Run lifecycle** — `run_id`, per-agent status tracking, resume-on-partial-failure, idempotent re-runs.
- **Persistence** — SQLite (WAL, single-writer), versioned migrations.
- **Decision aggregation** — confidence-weighted blend `Σ w·c·s / Σ w·c`, coverage floor, dead-signal dropout.
- **Risk engine *framework*** — deterministic rule evaluation, every violation named, all evaluations logged. (Rules + thresholds are re-authored for crypto.)
- **Paper accounting** — positions, weighted-average cost, realized/unrealized P&L, mark-to-market snapshots, approval queue. (Fill *timing* is rewritten — see §1.2.)
- **LLM client** — Ollama wrapper with grammar-constrained JSON, parse-retry, grounding requirement, hard-fail → neutral fallback.
- **Dashboard / reporting / analytics / backtester frameworks** — server-rendered FastAPI + HTMX, markdown+JSON report twins, Sharpe/drawdown/win-rate, day-by-day replay reusing live code paths.

### 1.2 Net-new or substantially rewritten

| Area | Equity behavior | Crypto rebuild |
|------|-----------------|----------------|
| **Market calendar + fill model** | Decide after close on day T → fill at T+1 **open**; weekdays only | **24/7, no close.** Bars are fixed UTC windows (1d primary; 1h/4h optional). Decision at bar close → fill at **next-bar open** (sim) or **immediate testnet order** (CCXT). No weekend/holiday gating. This is the single biggest engineering change and it touches both the paper engine and the backtester (which share the fill convention by design). |
| **Price/OHLCV data** | yfinance daily bars | **CCXT** unified OHLCV across exchanges (primary), with CoinGecko for market-cap/supply context. Same full-window upsert + quality-gate discipline. |
| **Fundamentals agent** | yfinance financial statements | **Deleted.** Replaced by **On-Chain** + **Tokenomics** agents (no analog exists). |
| **SEC filing agent** | EDGAR 10-K/10-Q risk extraction | **Deleted.** Replaced by **Protocol/Governance** signal (docs, governance forums, audit status) folded into On-Chain or run as its own LLM agent. |
| **Macro regime** | FRED: 10Y-2Y, VIX, HY OAS | **Crypto macro:** BTC dominance, aggregate funding, total market cap trend, stablecoin supply, (optional) spot-ETF net flows. |
| **News agent** | yfinance + Finnhub headlines | **Crypto news:** CoinGecko/CryptoPanic/RSS + exchange announcements; same grounding/dedupe discipline. Social (X) optional, gated behind a flag. |
| **Risk profile** | Sector caps, equity vol bands | **Category** caps (L1/L2/DeFi/infra/memecoin), much wider vol bands, stablecoin de-peg screen, **exchange-counterparty** concentration limit, liquidity/volume floor per asset. |
| **Universe / discovery** | S&P 500 (Wikipedia + GICS) | Top-N by market cap / volume from CoinGecko, category-tagged; gap-category-first rotation like the equity discovery agent. |

---

## 2. Spec Review — Crypto-Specific Shortcomings & Improvements

Carrying forward the equity plan's discipline, these are the gaps the *crypto* domain introduces beyond the original spec. Each is incorporated below.

### 2.1 Architecture / market-structure

| # | Issue | Improvement |
|---|-------|-------------|
| C1 | **No market close.** The entire T+1-open convention is meaningless 24/7. | Define a **bar clock**: a configurable primary timeframe (default `1d` on UTC boundaries). A "daily run" closes the just-completed UTC day; sim fills use the next bar's open; testnet routes a market order immediately at approval. The clock is the single source of truth shared by paper engine and backtester. |
| C2 | **Continuous trading invites overtrading + look-ahead in replay.** | The decision cadence is the bar clock, not wall-clock. Backtester slices history to each simulated bar close; no intra-bar peeking. A minimum re-decision interval and a no-churn band (don't flip on tiny score wiggles) protect against 24/7 thrash. |
| C3 | **Exchange counterparty + custody risk has no equity analog.** | Risk engine adds a per-exchange exposure cap and a venue allow-list; the broker layer records which venue holds each (paper) position. Stablecoins treated as a distinct category with a de-peg guard. |
| C4 | **Self-custody / settlement nuances** (testnet quirks, dust, min-notional, lot-size filters). | CCXT broker honors each market's `precision`/`limits` (min notional, step size); orders below min-notional are rejected pre-submit with a named reason, never silently dropped. |

### 2.2 Signal-quality

| # | Issue | Improvement |
|---|-------|-------------|
| K1 | **On-chain data is noisy, vendor-specific, and rate-limited.** | On-Chain agent reads from **DefiLlama** (TVL/fees/revenue, keyless) first; optional Glassnode/Dune behind env-var keys. Missing metric → neutral sub-score + reduced confidence, exactly like the equity fundamentals degrade path. Every metric carries `data_as_of`; stale-tolerant cache with a freshness ceiling. |
| K2 | **Tokenomics / unlock schedules are the crypto equivalent of dilution and are routinely ignored.** | Tokenomics agent scores circulating-vs-fully-diluted ratio, emission/inflation rate, and **upcoming unlock cliffs** (a large unlock inside the forecast horizon is a strong bearish prior). Sourced from token-unlock datasets; absent data → neutral. |
| K3 | **Derivatives positioning drives short-term crypto moves** (funding, OI, long/short skew) and has no equity parallel here. | Derivatives agent reads funding rate, open interest trend, and basis from CCXT-supported perp venues; extreme funding = crowded-trade contra-signal. PIT-safe subset (funding/OI history) is backtestable; the rest validates forward. |
| K4 | **Kronos was trained on equity candles**, not crypto's fat-tailed, 24/7 regime. | Kronos agent runs on crypto OHLCV but is **explicitly flagged experimental**: benchmark forecast skill against a naive baseline on a crypto holdout before granting it nonzero weight. Until validated, ship it at weight ~0 (present but ignorable). |
| K5 | **News/sentiment is faster and more reflexive in crypto.** | Same grounded-LLM discipline (cite the headline index or drop the driver; no grounding → confidence cap). Shorter lookback default; announcement feeds (listings, hacks, depegs) treated as high-salience event types. |

### 2.3 Engineering

| # | Issue | Improvement |
|---|-------|-------------|
| G1 | **CCXT rate limits + flaky exchange endpoints.** | Centralized CCXT client with built-in `enableRateLimit`, retry/backoff, and per-exchange connection reuse. All network I/O isolated behind a thin adapter so tests patch one seam (mirrors the equity `_download` pattern). |
| G2 | **Fractional, high-precision quantities** (8+ decimals) vs equity whole/large-fraction shares. | Quantities and prices use `Decimal` end-to-end in the money path; rounding follows each market's lot/precision filters. P&L math is unit-tested on fractional fixtures. |
| G3 | **Quote currency ambiguity** (USDT vs USD vs USDC books). | A single configured **account quote currency** (default USDT on testnet); all pairs normalized to it; cross-quote conversions explicit and logged. Benchmarks become BTC and ETH (the crypto "SPY/QQQ"). |
| G4 | **Secrets sprawl across many vendors** (exchange keys, on-chain API keys). | Same env-var-only secrets discipline as the equity repo (`core/secrets.py`); config references env var *names*, never values. Testnet keys clearly namespaced. |

---

## 3. Target Architecture

```
                        ┌────────────────────────────────────────────┐
                        │         Orchestrator (run_id, bar clock)   │
                        └────────────────────────────────────────────┘
 CCXT OHLCV Layer ──► Data Quality Gate (crypto-tuned)
        │
        │   per asset, IN PARALLEL
        ├─────────┬──────────┬────────────┬───────────┬───────────┬──────────┐
        ▼         ▼          ▼            ▼           ▼           ▼          ▼
   Technical   Kronos*    On-Chain    Tokenomics  Derivatives   News     (Protocol/
     Agent     Agent       Agent        Agent        Agent      Agent     Governance)
        └─────────┴────┬─────┴────────────┴───────────┴───────────┴──────────┘
                       ▼
              Trade Decision Agent  ◄── re-evaluates HELD positions every bar (exits)
                       ▼
               Portfolio Agent (sizing, inverse-vol, category fit)
                       ▼
            Risk Engine (deterministic, no AI; crypto rule-set)
                       ▼
           Pending Orders (human approval queue)†
                       ▼
       Broker (sim 24/7 fills  |  CCXT testnet market orders)
                       ▼
               SQLite (WAL, single writer, Decimal money path)
                       ▼
   Reports / Analytics / Dashboard / Notifications (vs BTC/ETH benchmarks)

  * Kronos ships experimental, weight ~0 until validated on a crypto holdout.
  † skippable via auto_approve config flag.
```

**Crypto Macro Regime Agent** runs once per bar (not per asset) and scales **new-position sizing only**, exactly like the equity macro agent — calm/caution/stress from BTC dominance + funding + total-cap trend; exits are never scaled; <2 live dials → neutral ×1.0.

**Module layout (`crypto_platform/`):**

```
crypto_platform/
├── config/
│   ├── settings.yaml          # bar clock, exchanges, LLM/Kronos endpoints, quote ccy
│   ├── universe.yaml          # assets + category tags (L1/L2/DeFi/infra/stable/meme)
│   ├── weights.yaml           # signal weights + buy/watchlist/exit thresholds
│   └── risk_limits.yaml       # position/category/venue caps, vol & liquidity floors
├── src/crypto_platform/
│   ├── core/                  # config, logging, run lifecycle, db, llm, secrets, bar_clock
│   ├── data/                  # ccxt_ohlcv.py, onchain.py (DefiLlama), tokenomics.py,
│   │                          # derivatives.py, news.py, macro.py, quality.py, universe.py
│   ├── agents/                # technical, kronos, onchain, tokenomics, derivatives,
│   │                          # news, governance, decision, portfolio  (base: AgentResult)
│   ├── risk/                  # engine.py, rules.py (crypto rule-set)
│   ├── execution/             # broker.py (interface), sim_broker.py, ccxt_broker.py,
│   │                          # orders.py, approval.py, account.py
│   ├── analytics/             # performance.py, metrics.py (vs BTC/ETH)
│   ├── backtest/              # replay.py, fills.py, benchmark.py
│   ├── reporting/             # daily_report.py, leaderboard.py
│   └── dashboard/             # FastAPI + HTMX
├── scripts/
│   ├── run_cycle.py           # one bar-close cycle (the "daily run" analog)
│   ├── approve_trades.py
│   ├── discover_assets.py
│   ├── run_backtest.py
│   └── init_db.py
├── db/   reports/   deploy/   tests/
```

**Core contracts (pydantic, copied from equity repo):** `AgentResult`, `TradeDecision`, `Order` — `ticker`→`symbol`, plus an optional `venue` on `Order`/`Position`.

**Tech stack:** Python 3.12, `uv`, pydantic v2, pandas, **ccxt**, **DefiLlama** (keyless) + optional Glassnode/Dune, Kronos (HF/PyTorch, experimental for crypto), Ollama (Qwen2.5-32B), FastAPI + HTMX, systemd timer (or short-interval scheduler) for the bar clock, pytest. `Decimal` for all money math.

---

## 4. Build Phases

Same philosophy as the equity build: every phase ships something runnable; money-touching code is deterministic and tested before any AI is wired in; the highest-uncertainty model work (Kronos on crypto, LLM agents, on-chain vendors) is isolated behind `AgentResult` so it never blocks the spine.

### Phase 0 — Fork & Foundations
- Stand up `crypto_platform` repo; copy `core/` contracts, run lifecycle, db engine, config loader, logging, secrets, LLM client **verbatim** (rename `ticker`→`symbol`).
- Add `core/bar_clock.py`: the UTC bar boundary abstraction every stage shares.
- SQLite schema + `init_db.py` (assets, agent_scores, decisions, orders, fills, positions, account_snapshots, ohlcv_cache, onchain_metrics, derivs_metrics, news_items, risk_events, runs) with the versioned-migration convention.
- **Done when:** a no-op cycle writes a `runs` row + report stub; copied core tests pass unchanged.

### Phase 1 — CCXT Market-Data Layer + Quality Gate
- `data/ccxt_ohlcv.py`: unified OHLCV fetch (default Binance Spot Testnet for live keys; public data for history), full-window upsert into `ohlcv_cache`, rate-limited + retried.
- Crypto-tuned quality gate: history depth, NaN/zero-volume, price sanity, **gap detection on a 24/7 calendar** (a missing UTC day is a real gap, unlike equities), staleness, exchange-outage flatline detection. Fail → skip + flag.
- **Done when:** the configured universe refreshes idempotently; corrupted fixtures are caught; a synthetic exchange-gap fixture is flagged.

### Phase 2 — Technical Agent (deterministic, no AI)
- Port the equity rubric (trend structure / momentum / trend quality / risk) onto crypto OHLCV; re-anchor momentum and volatility bands for crypto's wider distribution (config-driven, not hard-coded).
- Golden-file tests pin exact scores; agent crash → neutral zero-confidence isolation.
- **Done when:** repeatable identical scores on fixed crypto fixtures.

### Phase 3 — On-Chain Agent (DefiLlama-first)
- `data/onchain.py`: TVL, fees, revenue, (where available) active addresses; keyless DefiLlama primary, optional keyed vendors behind env vars.
- Rubric with explicit worst→best anchors per metric; missing metric → neutral sub-score, confidence `0.3 + 0.6×coverage` (mirrors equity fundamentals). Never guesses.
- **Done when:** universe assets score with a stored breakdown + `data_as_of`; pulling the network yields neutral, not crash.

### Phase 4 — Tokenomics Agent
- `data/tokenomics.py`: circulating/FDV ratio, emission/inflation rate, **upcoming unlock cliffs** within the forecast horizon (high-salience bearish prior).
- Deterministic rubric; absent supply data → neutral + reduced confidence.
- **Done when:** unlock-cliff fixture produces the expected bearish tilt; clean assets score normally.

### Phase 5 — Derivatives Agent
- `data/derivatives.py`: funding rate, open-interest trend, basis from CCXT perp venues. Extreme funding = crowded-trade contra-signal; PIT-safe funding/OI history stored for backtest.
- **Done when:** funding/OI fixtures map to correct scores; venues without derivs degrade to neutral.

### Phase 6 — Kronos Forecast Agent (experimental)
- Vendor + pin Kronos (as in the equity repo); one-load-per-run forecaster; map mean H-bar forecast return → score, confidence from path dispersion.
- **Validation gate:** benchmark forecast skill vs. a naive baseline on a crypto holdout *before* assigning weight. Ships at weight ~0 until it earns it.
- **Done when:** forecasts produced within run budget; a documented skill comparison exists.

### Phase 7 — News + Protocol/Governance Agents (local LLM)
- Reuse the Ollama grounded-JSON client. News: CoinGecko/CryptoPanic/RSS + exchange announcements → dedupe → grounded classification (listing, hack, depeg, upgrade, regulatory). Governance/protocol: docs + governance-forum/audit signals → grounded risk findings.
- Both must cite sources or drop the claim; no source → neutral fallback.
- **Done when:** both run on live data; offline yields neutral scores, not crashes.

### Phase 8 — Decision + Portfolio Agents
- Confidence-weighted aggregation with config weights; buy/watchlist/hold/exit thresholds.
- **Exit policy** (re-evaluated every bar): stop-loss, take-profit, max-hold (in bars), score-decay, **and an unlock-cliff exit** (trim ahead of a large scheduled unlock). No-churn band + min re-decision interval (C2).
- Portfolio sizing: base % of equity × score × inverse-vol, capped by position/category/venue limits and the cash-reserve floor.
- **Done when:** synthetic fixtures produce correct buy/sell/hold/size decisions incl. exit + no-churn cases.

### Phase 9 — Risk Engine (deterministic, crypto rule-set)
- Re-author rules from `risk_limits.yaml`: final-score floor, per-agent floors, restricted assets, position cap, **category cap**, **per-venue exposure cap**, **liquidity/volume floor**, **stablecoin de-peg guard**, cash sufficiency, reserve floor. Every violation named; all logged to `risk_events`; sells exempt from exposure rules (blocking a stop-loss is itself a risk).
- **Exhaustive unit tests** — pass/fail/boundary per rule (this is the safety layer).
- **Done when:** every rule has pass, fail, and boundary tests.

### Phase 10 — Broker + Approval Queue (sim first, CCXT testnet second)
- `execution/broker.py` interface; `sim_broker.py`: 24/7 next-bar-open fills with maker/taker fee + slippage model, weighted-avg cost, realized/unrealized P&L, per-bar mark-to-market.
- Orders lifecycle `awaiting_approval → approved/rejected/expired → filled`; `UNIQUE(run_id, symbol, side)` idempotency; auto-expiry at next bar; `auto_approve` flag.
- `ccxt_broker.py`: route approved orders to exchange **testnet** as market orders (idempotent `clientOrderId`), honoring min-notional/precision filters; next cycle syncs real fills + reconciles positions/cash, surfacing (never auto-fixing) drift. **Testnet base URL hard-pinned; a test enforces it.**
- **Done when:** a full buy→hold→exit cycle round-trips with exact fractional P&L in tests; (with keys) one order round-trips through testnet and reconciles.

### Phase 11 — Crypto Macro Regime Agent
- `data/macro.py`: BTC dominance, aggregate funding, total market-cap trend, stablecoin supply (+ optional ETF flows). Each dial 0/1/2 vs config thresholds → calm/caution/stress; scales **new** sizing only (×1.0/0.75/0.5), <2 live dials → neutral ×1.0.
- **Done when:** regime surfaced in report/JSON/sizing audit; backtests document that regime scaling is off (needs PIT macro).

### Phase 12 — Orchestrator + Scheduler (bar clock)
- `run_cycle.py`: parallel agent fan-out, held-position re-evaluation every bar, leaderboard, decision→risk→order, idempotent re-runs. Expire stale unapproved orders + fill approved ones at the bar open **before** analysis, so decisions see post-fill state.
- Notification hook on success/failure; run-health (duration, stage failures, data freshness). `deploy/`: systemd service + timer firing on the bar boundary (e.g. 00:05 UTC for the daily clock; shorter interval if intraday).
- **Done when:** two runs of the same bar produce no duplicates; a mid-run kill resumes cleanly.

### Phase 13 — Reporting + Performance Analytics
- Markdown + JSON report twins (leaderboard, decisions, risk verdicts, account, regime, pending approvals). Analytics: total/annualized return, Sharpe, max drawdown, win rate, **vs BTC and ETH** benchmarks (auto-maintained in `ohlcv_cache`), per-signal hit rates graded on H-bar forward return.
- **Done when:** a simulated history window renders metrics validated against hand-computed values.

### Phase 14 — Dashboard
- FastAPI + HTMX, terminal aesthetic, zero build step: account KPIs, equity curve, performance + per-signal hit rates, approval queue (one-click approve/reject, the only write path), open positions w/ live unrealized P&L + venue, decisions w/ risk verdicts, agent leaderboard (zero-confidence dotted), regime, run health, report browser.
- **Done when:** the full review→approve→see-fills loop is doable in the browser.

### Phase 15 — Backtester (PIT-safe replay)
- Day/bar-by-bar replay driving the **same** Technical/Derivatives(+experimental Kronos)/Decision/Portfolio/Risk/order/fill/snapshot code as live, against a scratch SQLite audit-trail db. PIT-safe: only signals with point-in-time history replay (technical, funding/OI, opt-in kronos); on-chain/tokenomics/news validate forward. Benchmarks vs BTC/ETH; weight & threshold sensitivity sweep.
- **Done when:** a multi-year backtest completes and a sweep produces a comparison table stating included signals + assumptions.

### Phase 16 — Discovery Agent
- `discover_assets.py`: weekly screen of a top-N-by-market-cap/volume universe (CoinGecko, category-tagged, cached), gap-category-first rotation, standard quality gate + the same technical/on-chain/tokenomics rubrics, ranked at live weights + category-gap bonus, with a liquidity floor blocking thin/illiquid junk. Proposals only — `--add` is the human's pen (appends to `universe.yaml`, capacity-capped).
- **Done when:** a discovery run proposes ranked candidates with a stored rationale; `--add` updates the universe.

---

## 5. Build-Order Rationale & Effort

- **Deterministic before AI:** Phases 2–5, 8–10 (technical, on-chain/tokenomics/derivs rubrics, decision, risk, broker) are fully testable without a model and are locked down before LLM/Kronos variability enters.
- **Highest-uncertainty items (6 Kronos-on-crypto, 7 LLM agents, 3 on-chain vendors)** sit behind the `AgentResult` seam — if any lag, the platform still ships with neutral scores in their slots.
- **Backtester after the live pipeline (15)** so it reuses identical decision/risk/fill code (backtest fidelity = paper fidelity).
- **The fork's critical path is the bar clock + 24/7 fill model (Phase 0/10).** Everything else is either copied or analogous to the equity build; this is the one piece with no prior art in the source repo.

Rough effort (focused sessions): Phase 0 ≈ 2–3 days (mostly copy + bar clock) · Phases 1–2 ≈ 3–4 days · Phases 3–5 ≈ 5–7 days (new data vendors dominate) · Phase 6 ≈ 2–3 days · Phase 7 ≈ 3–4 days · Phases 8–10 ≈ 5–6 days (24/7 fills + risk) · Phases 11–14 ≈ 4–5 days · Phases 15–16 ≈ 3–4 days. **Total ≈ 4–6 weeks part-time**, comparable to the equity build because ~60% of the spine is copied.

---

## 6. Open Questions to Confirm Before Phase 1

1. **Primary exchange + testnet** — Binance Spot Testnet assumed; confirm, and whether a second venue (Coinbase/Kraken sandbox) is in scope for the venue-exposure logic.
2. **Quote currency** — USDT on testnet assumed; confirm vs USDC/USD.
3. **Bar timeframe** — `1d` UTC primary assumed; is intraday (1h/4h) wanted at v1, or a later phase?
4. **On-chain vendor budget** — DefiLlama keyless covers TVL/fees/revenue; confirm whether to wire paid Glassnode/Dune keys for address-level metrics or stay keyless at v1.
5. **Universe size** — top-N by market cap (N=?) plus any hand-picked majors (BTC/ETH always in).
6. **Kronos** — keep it experimental at weight 0 as specified, or drop it from v1 and add post-validation?

