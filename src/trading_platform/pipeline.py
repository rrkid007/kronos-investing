"""Daily pipeline orchestrator.

Per run:
  pass 1 — per ticker (watchlist + currently held): market data refresh +
           quality gate, then each registered research agent scores the
           ticker and persists an AgentResult. Agent crashes are isolated to
           a neutral zero-confidence score; the run continues.
  pass 2 — Decision Engine: confidence-weighted aggregation of the stored
           scores per ticker, exit checks for held positions. Decisions are
           recomputed (upserted) on every invocation — they're pure functions
           of stored scores.
  pass 3 — Portfolio Agent: buy candidates ranked by final score consume
           cash/sector room greedily; sizing and fit are written back onto
           each decision.

Stages without a registered agent are no-ops until their phase lands.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from trading_platform.agents.decision import DecisionEngine
from trading_platform.analytics.performance import performance_summary
from trading_platform.agents.fundamentals import FundamentalsAgent
from trading_platform.agents.kronos import KronosAgent
from trading_platform.agents.news import NewsAgent
from trading_platform.agents.portfolio import PortfolioAgent, PortfolioState
from trading_platform.agents.sec_filing import SECFilingAgent
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.core.config import AppConfig
from trading_platform.core.db import connect, init_db
from trading_platform.core.llm import OllamaClient
from trading_platform.core.models import Action, AgentResult, TradeDecision
from trading_platform.core.notify import notify
from trading_platform.core.runs import (
    complete_run,
    create_run,
    find_resumable_run,
    mark_stage,
    save_agent_result,
    save_decision,
    stage_status,
)
from trading_platform.data.market_data import MarketDataService
from trading_platform.execution.account import load_cash, load_positions
from trading_platform.execution.orders import (
    expire_stale_orders,
    fillable_orders,
    submit_order,
)
from trading_platform.execution.paper_broker import (
    FillError,
    ensure_account,
    fill_order,
    snapshot_account,
)
from trading_platform.risk.engine import RiskEngine, log_risk_event

logger = logging.getLogger(__name__)

# Research stages in fan-out order. Stages without a registered agent are
# no-ops until their phase lands.
AGENT_STAGES = ["technical", "kronos", "fundamentals", "news", "sec_filing"]
DATA_STAGE = "market_data"
DECISION_STAGE = "decision"
PORTFOLIO_STAGE = "portfolio"
RISK_STAGE = "risk"
EXECUTION_STAGE = "execution"
ORDERS_STAGE = "orders"
SNAPSHOT_STAGE = "snapshot"


def build_agent_registry(config: AppConfig, conn=None) -> dict:
    """One agent instance per run — the Kronos model loads once and is reused
    across every ticker in the scan; News/SEC share one LLM client and use the
    run's connection for persistence and the filing cache."""
    llm = OllamaClient(config.settings.llm)
    return {
        "technical": TechnicalAgent(),
        "fundamentals": FundamentalsAgent(),
        "kronos": KronosAgent(config.settings.kronos),
        "news": NewsAgent(config, llm=llm, conn=conn),
        "sec_filing": SECFilingAgent(config, llm=llm, conn=conn),
    }


def run_daily(
    config: AppConfig,
    run_date: str | None = None,
    market_data: MarketDataService | None = None,
) -> str:
    """Execute one daily analysis run. Returns the run_id.

    Idempotent: re-invoking for the same date resumes the incomplete run and
    skips stages already completed. market_data is injectable for tests.
    """
    run_date = run_date or date.today().isoformat()
    as_of = date.fromisoformat(run_date)
    started = time.perf_counter()
    conn = connect(config.db_path)
    init_db(conn)
    injected_market_data = market_data is not None  # tests inject fakes
    if market_data is None:
        market_data = MarketDataService(conn)

    run_id = find_resumable_run(conn, run_date)
    if run_id:
        logger.info("resuming run %s for %s", run_id, run_date)
    else:
        run_id = create_run(conn, run_date)
        logger.info("created run %s for %s", run_id, run_date)

    registry = build_agent_registry(config, conn=conn)
    skipped: dict[str, str] = {}  # ticker -> reason, for the report
    frames: dict[str, pd.DataFrame] = {}
    try:
        # --- pass 0: execution — expire stale orders, fill approved ones at
        # today's open, BEFORE analysis (fills change positions and cash).
        mark_stage(conn, run_id, EXECUTION_STAGE, "running")
        ensure_account(conn, config.risk.paper_account.starting_cash)
        n_expired = expire_stale_orders(conn, run_date)
        n_filled = _process_fills(conn, config, market_data, run_date, as_of, frames)
        mark_stage(conn, run_id, EXECUTION_STAGE, "completed",
                   detail=f"{n_filled} filled, {n_expired} expired")

        positions = load_positions(conn)  # post-fill view
        held_by_ticker = {p.ticker: p for p in positions}
        # Held tickers are always re-evaluated, even off-watchlist (exit policy).
        symbols = list(config.watchlist.symbols)
        symbols += [t for t in held_by_ticker if t not in symbols]

        # --- pass 1a: market data — downloads fan out in parallel (A1);
        # all SQLite writes stay on this thread.
        _refresh_market_data(conn, config, market_data, injected_market_data,
                             run_id, symbols, as_of, skipped, frames)

        # --- pass 1b: research agents. Network-bound stages fan out across
        # tickers; GPU/LLM stages run serial (single model, single Ollama).
        _run_research_stages(conn, registry, run_id, symbols, frames, skipped)

        # --- pass 2: decisions (pure recompute from stored scores)
        engine = DecisionEngine(config.weights)
        decisions: dict[str, TradeDecision] = {}
        for symbol in symbols:
            mark_stage(conn, run_id, DECISION_STAGE, "running", ticker=symbol)
            df = frames.get(symbol)
            price = float(df["close"].iloc[-1]) if df is not None else None
            decision = engine.decide(
                symbol, run_id,
                signals=_load_signals(conn, run_id, symbol),
                position=held_by_ticker.get(symbol),
                current_price=price,
                today=as_of,
            )
            decisions[symbol] = decision
            save_decision(conn, decision)
            mark_stage(conn, run_id, DECISION_STAGE, "completed", ticker=symbol,
                       detail=f"{decision.action.value}: {decision.reason[:80]}")

        # --- pass 3: portfolio sizing for buys, strongest first
        mark_stage(conn, run_id, PORTFOLIO_STAGE, "running")
        n_approved = _assess_buys(conn, config, decisions, frames, positions)
        mark_stage(conn, run_id, PORTFOLIO_STAGE, "completed",
                   detail=f"{n_approved} buy(s) sized and approved")

        # --- pass 4: risk engine re-validates every proposed trade
        mark_stage(conn, run_id, RISK_STAGE, "running")
        n_cleared = _risk_pass(conn, config, run_id, decisions, frames,
                               positions, held_by_ticker)
        mark_stage(conn, run_id, RISK_STAGE, "completed",
                   detail=f"{n_cleared} trade(s) cleared risk")

        # --- pass 5: risk-cleared trades become orders (approval queue)
        mark_stage(conn, run_id, ORDERS_STAGE, "running")
        n_orders = _submit_orders(conn, config, run_id, run_date, decisions, frames)
        mark_stage(conn, run_id, ORDERS_STAGE, "completed",
                   detail=f"{n_orders} order(s) submitted")

        # --- pass 6: mark-to-market snapshot + benchmark series upkeep
        mark_stage(conn, run_id, SNAPSHOT_STAGE, "running")
        closes = {t: float(df["close"].iloc[-1]) for t, df in frames.items()}
        summary = snapshot_account(conn, run_date, closes)
        _refresh_benchmarks(conn, config, market_data, injected_market_data, as_of)
        mark_stage(conn, run_id, SNAPSHOT_STAGE, "completed",
                   detail=f"equity {summary['equity']:,.0f}")

        duration = time.perf_counter() - started
        _write_report(conn, config, run_id, run_date, skipped, duration)
        complete_run(conn, run_id)
        logger.info("run %s completed (%d tickers skipped)", run_id, len(skipped))
        notify(
            config.settings.notifications,
            f"Trading run {run_date}: OK",
            _run_summary(conn, run_id, run_date, summary, decisions, skipped, duration),
        )
    except Exception as exc:
        complete_run(conn, run_id, status="partial", notes="crashed mid-run; resumable")
        notify(
            config.settings.notifications,
            f"Trading run {run_date}: FAILED",
            f"run {run_id} crashed (resumable on next invocation):\n{exc}",
        )
        raise
    finally:
        conn.close()
    return run_id


def _run_summary(conn, run_id, run_date, account_summary, decisions, skipped,
                 duration) -> str:
    actions: dict[str, int] = {}
    for d in decisions.values():
        actions[d.action.value] = actions.get(d.action.value, 0) + 1
    n_pending = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE status = 'awaiting_approval'"
    ).fetchone()[0]
    lines = [
        f"run {run_id} in {duration:.0f}s",
        "decisions: " + (", ".join(f"{k}={v}" for k, v in sorted(actions.items())) or "none"),
        f"equity ${account_summary['equity']:,.2f} "
        f"(cash ${account_summary['cash']:,.2f}, "
        f"{account_summary['n_positions']} positions)",
    ]
    if n_pending:
        lines.append(f"** {n_pending} order(s) awaiting approval — run approve_trades.py **")
    if skipped:
        lines.append(f"skipped tickers: {', '.join(sorted(skipped))}")
    return "\n".join(lines)


def _load_signals(conn, run_id: str, symbol: str) -> dict[str, tuple[float, float]]:
    rows = conn.execute(
        "SELECT agent, score, confidence FROM agent_scores WHERE run_id = ? AND ticker = ?",
        (run_id, symbol),
    ).fetchall()
    return {r["agent"]: (r["score"], r["confidence"]) for r in rows}


def _portfolio_state(conn, config, frames, positions) -> PortfolioState:
    """Mark-to-market account view: cash + positions valued at last close."""
    sector_values: dict[str, float] = {}
    pos_value = 0.0
    for p in positions:
        df = frames.get(p.ticker)
        price = float(df["close"].iloc[-1]) if df is not None else p.avg_cost
        value = p.qty * price
        pos_value += value
        sector = config.watchlist.sector_of(p.ticker) or "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    cash = load_cash(conn, config)
    return PortfolioState(
        cash=cash, equity=cash + pos_value,
        positions=positions, sector_values=sector_values,
    )


def _assess_buys(conn, config, decisions, frames, positions) -> int:
    """Rank buys by final score and size them against a running state."""
    state = _portfolio_state(conn, config, frames, positions)
    agent = PortfolioAgent(config.risk, config.watchlist)

    buys = sorted(
        (d for d in decisions.values() if d.action == Action.BUY),
        key=lambda d: d.final_score, reverse=True,
    )
    approved = 0
    for decision in buys:
        df = frames.get(decision.ticker)
        price = float(df["close"].iloc[-1]) if df is not None else None
        assessment = agent.assess_buy(decision, price, df, state)
        decision.signal_breakdown["portfolio"] = assessment.model_dump()
        decision.sizing_hint = assessment.target_value if assessment.approved else None
        if assessment.approved:
            sector = config.watchlist.sector_of(decision.ticker) or "Unknown"
            state.apply_buy(decision.ticker, sector, assessment.target_value)
            approved += 1
        save_decision(conn, decision)
    return approved


def _process_fills(conn, config, market_data, run_date, as_of, frames) -> int:
    """Fill approved orders from earlier runs at today's open.

    An order whose ticker has no completed bar for today (halt, data issue)
    stays approved and is retried next run. A fill the broker refuses
    (e.g. selling an absent position) is cancelled with the reason noted.
    """
    filled = 0
    for order in fillable_orders(conn, run_date):
        ticker = order["ticker"]
        df = frames.get(ticker)
        if df is None:
            data = market_data.refresh_and_validate(ticker, as_of=as_of)
            df = data.df
            if data.ok:  # only gate-passed frames may reach the agents
                frames[ticker] = df
        bar = df.loc[df.index.date == as_of] if df is not None and not df.empty else None
        if bar is None or bar.empty:
            logger.warning("no bar for %s on %s; order %s stays pending fill",
                           ticker, run_date, order["order_id"])
            continue
        try:
            result = fill_order(conn, order, as_of, float(bar["open"].iloc[0]),
                                config.risk.fill_model)
            logger.info("filled %s %s x%s @ %.2f", result["side"], ticker,
                        result["qty"], result["exec_price"])
            filled += 1
        except FillError as exc:
            logger.error("fill refused for %s: %s", order["order_id"], exc)
            conn.execute(
                "UPDATE orders SET status = 'cancelled', notes = ? WHERE order_id = ?",
                (f"fill refused: {exc}", order["order_id"]),
            )
            conn.commit()
    return filled


def _submit_orders(conn, config, run_id, run_date, decisions, frames) -> int:
    """Risk-cleared decisions become orders in the approval queue."""
    auto = not config.risk.require_human_approval
    submitted = 0
    for decision in decisions.values():
        risk = decision.signal_breakdown.get("risk")
        if not risk or not risk.get("approved"):
            continue
        if decision.action == Action.SELL:
            qty = decision.sizing_hint
        elif decision.action == Action.BUY:
            qty = decision.signal_breakdown["portfolio"]["qty"]
        else:
            continue
        price = _last_close(frames, decision.ticker)
        submit_order(
            conn, run_id, decision.ticker, decision.action.value, qty, run_date,
            auto_approve=auto,
            context={
                "final_score": decision.final_score,
                "reason": decision.reason,
                "est_price": price,
                "est_value": round(qty * price, 2) if price else None,
            },
        )
        submitted += 1
    return submitted


def _risk_pass(conn, config, run_id, decisions, frames, positions, held_by_ticker) -> int:
    """Independently re-validate every proposed trade; log all evaluations.

    Buys are checked strongest-first against a fresh running state (sell
    proceeds are NOT credited — fills happen T+1, so today's buys must fit
    today's cash). Sells are exits and only sanity-checked.
    """
    engine = RiskEngine(config.risk, config.watchlist)
    state = _portfolio_state(conn, config, frames, positions)
    cleared = 0

    for decision in decisions.values():
        if decision.action != Action.SELL:
            continue
        price = _last_close(frames, decision.ticker)
        result = engine.evaluate_sell(decision, held_by_ticker.get(decision.ticker), price)
        log_risk_event(conn, run_id, result)
        decision.signal_breakdown["risk"] = result.model_dump()
        save_decision(conn, decision)
        cleared += result.approved

    buys = sorted(
        (d for d in decisions.values()
         if d.action == Action.BUY
         and d.signal_breakdown.get("portfolio", {}).get("approved")),
        key=lambda d: d.final_score, reverse=True,
    )
    for decision in buys:
        qty = decision.signal_breakdown["portfolio"]["qty"]
        price = _last_close(frames, decision.ticker)
        result = engine.evaluate_buy(decision, qty, price, state)
        log_risk_event(conn, run_id, result)
        decision.signal_breakdown["risk"] = result.model_dump()
        save_decision(conn, decision)
        if result.approved:
            sector = config.watchlist.sector_of(decision.ticker) or "Unknown"
            state.apply_buy(decision.ticker, sector, qty * price)
            cleared += 1
        else:
            logger.warning("risk blocked %s buy: %s", decision.ticker, result.errors)
    return cleared


def _last_close(frames: dict, ticker: str) -> float | None:
    df = frames.get(ticker)
    return float(df["close"].iloc[-1]) if df is not None else None


def _run_agent_stage(
    conn, registry: dict, run_id: str, stage: str, symbol: str, df: pd.DataFrame
) -> None:
    agent = registry.get(stage)
    if agent is None:
        mark_stage(conn, run_id, stage, "completed", ticker=symbol,
                   detail="no-op (agent pending)")
        return

    mark_stage(conn, run_id, stage, "running", ticker=symbol)
    try:
        result = agent.analyze(symbol, run_id, df)
    except Exception as exc:
        # Isolate agent crashes: store an ignorable neutral score, keep going.
        logger.exception("agent %s crashed on %s", stage, symbol)
        save_agent_result(conn, AgentResult.neutral(stage, symbol, run_id,
                                                    f"agent error: {exc}"))
        mark_stage(conn, run_id, stage, "failed", ticker=symbol, detail=str(exc))
        return

    save_agent_result(conn, result)
    mark_stage(conn, run_id, stage, "completed", ticker=symbol,
               detail=f"score {result.score:.1f} ({result.direction.value})")


MAX_DATA_WORKERS = 8
PARALLEL_AGENT_STAGES = {"fundamentals"}  # network-bound, stateless, no conn use


def _refresh_benchmarks(conn, config, market_data, injected: bool, as_of: date) -> None:
    """Keep SPY/QQQ in the price cache for performance comparison.

    Best-effort: benchmarks aren't scored, so a failed refresh logs and moves
    on. Skipped entirely when a fake service is injected (tests)."""
    if injected:
        return
    for ticker in config.settings.benchmarks:
        result = market_data.refresh(ticker, as_of=as_of)
        if result.error:
            logger.warning("benchmark refresh failed for %s: %s", ticker, result.error)


def _refresh_market_data(
    conn, config, market_data, injected: bool, run_id: str,
    symbols: list[str], as_of: date,
    skipped: dict[str, str], frames: dict[str, pd.DataFrame],
) -> None:
    """Refresh + gate all tickers, downloads in parallel (A1).

    Workers each open their own SQLite connection (WAL serializes writes);
    with an injected fake service (tests) workers share it directly. All
    stage marking happens on the calling thread.
    """
    pending: list[str] = []
    for symbol in symbols:
        if symbol in frames:  # already refreshed during fill processing
            mark_stage(conn, run_id, DATA_STAGE, "completed", ticker=symbol,
                       detail="refreshed during fill processing")
        elif stage_status(conn, run_id, DATA_STAGE, symbol) == "completed":
            frames[symbol] = market_data.load(symbol)  # resumed run
        else:
            mark_stage(conn, run_id, DATA_STAGE, "running", ticker=symbol)
            pending.append(symbol)
    if not pending:
        return

    def refresh(symbol: str):
        if injected:
            return market_data.refresh_and_validate(symbol, as_of=as_of)
        worker_conn = connect(config.db_path)
        try:
            return MarketDataService(worker_conn).refresh_and_validate(symbol, as_of=as_of)
        finally:
            worker_conn.close()

    with ThreadPoolExecutor(max_workers=min(MAX_DATA_WORKERS, len(pending))) as pool:
        futures = {symbol: pool.submit(refresh, symbol) for symbol in pending}

    for symbol in pending:
        try:
            data = futures[symbol].result()
        except Exception as exc:  # worker crash == fetch failure, not run failure
            logger.exception("market data worker crashed for %s", symbol)
            mark_stage(conn, run_id, DATA_STAGE, "failed", ticker=symbol, detail=str(exc))
            skipped[symbol] = f"refresh crashed: {exc}"
            continue
        if data.ok:
            mark_stage(conn, run_id, DATA_STAGE, "completed", ticker=symbol,
                       detail=data.detail())
            frames[symbol] = data.df
        else:
            logger.warning("data gate failed for %s: %s", symbol, data.detail())
            mark_stage(conn, run_id, DATA_STAGE, "failed", ticker=symbol,
                       detail=data.detail())
            skipped[symbol] = data.detail()


def _run_research_stages(
    conn, registry: dict, run_id: str, symbols: list[str],
    frames: dict[str, pd.DataFrame], skipped: dict[str, str],
) -> None:
    """Run each agent stage across tickers; parallel where safe.

    GPU (kronos) and LLM (news, sec_filing) stages stay serial — one model,
    one Ollama, and they write to the shared connection. All persistence
    happens on the calling thread either way.
    """
    for stage in AGENT_STAGES:
        todo = [s for s in symbols
                if stage_status(conn, run_id, stage, s) != "completed"]
        ready = []
        for symbol in todo:
            if symbol not in frames:
                mark_stage(conn, run_id, stage, "skipped", ticker=symbol,
                           detail=skipped.get(symbol, "market data unavailable"))
            else:
                ready.append(symbol)

        agent = registry.get(stage)
        if agent is None:
            for symbol in ready:
                mark_stage(conn, run_id, stage, "completed", ticker=symbol,
                           detail="no-op (agent pending)")
            continue

        if stage in PARALLEL_AGENT_STAGES and len(ready) > 1:
            for symbol in ready:
                mark_stage(conn, run_id, stage, "running", ticker=symbol)
            with ThreadPoolExecutor(max_workers=min(MAX_DATA_WORKERS, len(ready))) as pool:
                futures = {s: pool.submit(agent.analyze, s, run_id, frames[s])
                           for s in ready}
            for symbol in ready:
                _persist_agent_outcome(conn, run_id, stage, symbol, futures[symbol])
        else:
            for symbol in ready:
                _run_agent_stage(conn, registry, run_id, stage, symbol, frames[symbol])


def _persist_agent_outcome(conn, run_id, stage, symbol, future) -> None:
    """Main-thread persistence for a parallel agent call, crash-isolated."""
    try:
        result = future.result()
    except Exception as exc:
        logger.exception("agent %s crashed on %s", stage, symbol)
        save_agent_result(conn, AgentResult.neutral(stage, symbol, run_id,
                                                    f"agent error: {exc}"))
        mark_stage(conn, run_id, stage, "failed", ticker=symbol, detail=str(exc))
        return
    save_agent_result(conn, result)
    mark_stage(conn, run_id, stage, "completed", ticker=symbol,
               detail=f"score {result.score:.1f} ({result.direction.value})")


def _write_report(
    conn, config: AppConfig, run_id: str, run_date: str,
    skipped: dict[str, str], duration: float = 0.0,
) -> Path:
    report_dir = config.reports_dir / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{run_date}.md"
    tickers = ", ".join(config.watchlist.symbols)
    lines = [
        f"# Daily Research Report — {run_date}",
        "",
        f"Run ID: `{run_id}`",
        "",
        f"Watchlist: {tickers}",
        "",
    ]

    import json as _json

    snap = conn.execute(
        "SELECT * FROM account_snapshots WHERE snapshot_date = ?", (run_date,)
    ).fetchone()
    if snap:
        positions = conn.execute(
            "SELECT ticker, qty, avg_cost FROM positions ORDER BY ticker"
        ).fetchall()
        lines += [
            "## Account",
            "",
            f"- Equity: **${snap['equity']:,.2f}**  |  Cash: ${snap['cash']:,.2f}",
            f"- Realized P&L: ${snap['realized_pnl']:,.2f}  |  "
            f"Unrealized P&L: ${snap['unrealized_pnl']:,.2f}",
            f"- Open positions: {len(positions)}"
            + (" — " + ", ".join(f"{p['ticker']} x{p['qty']:g} @ {p['avg_cost']:.2f}"
                                 for p in positions) if positions else ""),
            "",
        ]

    pending = conn.execute(
        "SELECT o.order_id, o.ticker, o.side, o.qty, o.notes FROM orders o "
        "WHERE o.status = 'awaiting_approval' ORDER BY o.ticker"
    ).fetchall()
    if pending:
        lines += [
            "## Pending Orders — approval required",
            "",
            "Run `python scripts/approve_trades.py` to review.",
            "",
            "| Order ID | Ticker | Side | Qty | Est. Value |",
            "|----------|--------|------|----:|-----------:|",
        ]
        for o in pending:
            notes = _json.loads(o["notes"] or "{}")
            value = notes.get("est_value")
            lines.append(
                f"| `{o['order_id']}` | {o['ticker']} | {o['side']} | {o['qty']:g} "
                f"| {'$' + format(value, ',.0f') if value else '—'} |"
            )
        lines += [""]

    decisions = conn.execute(
        "SELECT ticker, action, final_score, sizing_hint, reason, signal_breakdown "
        "FROM decisions WHERE run_id = ? ORDER BY final_score DESC",
        (run_id,),
    ).fetchall()
    if decisions:
        lines += [
            "## Decisions",
            "",
            "| Ticker | Action | Final Score | Size | Risk | Reason |",
            "|--------|--------|------------:|-----:|------|--------|",
        ]
        for d in decisions:
            size = f"${d['sizing_hint']:,.0f}" if d["sizing_hint"] else "—"
            payload = _json.loads(d["signal_breakdown"] or "{}")
            risk = payload.get("risk")
            if risk is None:
                risk_cell = "—"
            elif risk["approved"]:
                risk_cell = "✓ cleared"
            else:
                first_error = (risk.get("checks") and
                               next((c["detail"] for c in risk["checks"] if not c["passed"]), ""))
                risk_cell = f"✗ {first_error[:40]}"
            lines.append(
                f"| {d['ticker']} | **{d['action']}** | {d['final_score']:.1f} "
                f"| {size} | {risk_cell} | {d['reason'][:60]} |"
            )
        lines += [""]

    rows = conn.execute(
        "SELECT ticker, agent, score, confidence, direction FROM agent_scores "
        "WHERE run_id = ? ORDER BY ticker, agent",
        (run_id,),
    ).fetchall()
    if rows:
        lines += [
            "## Agent Scores",
            "",
            "| Ticker | Agent | Score | Confidence | Direction |",
            "|--------|-------|------:|-----------:|-----------|",
        ]
        lines += [
            f"| {r['ticker']} | {r['agent']} | {r['score']:.1f} "
            f"| {r['confidence']:.2f} | {r['direction']} |"
            for r in rows
        ]
        lines += [""]

    if skipped:
        lines += ["## Data Quality Flags", ""]
        lines += [f"- **{t}** — skipped: {reason}" for t, reason in sorted(skipped.items())]
        lines += [""]

    perf = performance_summary(conn, config.settings.benchmarks)
    if perf["n_snapshots"] >= 2:
        fmt_pct = lambda v: f"{v * 100:.2f}%" if v is not None else "n/a"  # noqa: E731
        lines += [
            "## Performance",
            "",
            f"- Total return: **{fmt_pct(perf['total_return'])}** "
            f"({perf['window']['start']} → {perf['window']['end']})",
            f"- Annualized: {fmt_pct(perf['annualized_return'])}  |  "
            f"Sharpe: {perf['sharpe'] if perf['sharpe'] is not None else 'n/a'}  |  "
            f"Max drawdown: {fmt_pct(perf['max_drawdown'])}",
        ]
        t = perf["trades"]
        if t["n_trades"]:
            lines += [
                f"- Trades: {t['n_trades']} closed, win rate "
                f"{fmt_pct(t['win_rate'])}, avg return {t['avg_return_pct']:.2f}%, "
                f"total P&L ${t['total_pnl']:,.2f}",
            ]
        for bench, b in (perf.get("benchmarks") or {}).items():
            lines += [f"- {bench} same window: {fmt_pct(b['total_return'])}"]
        if perf["signal_hit_rates"]:
            lines += ["", "Per-signal hit rates (10-day forward):", ""]
            lines += [
                f"- {agent}: {s['hits']}/{s['n_calls']} = {fmt_pct(s['hit_rate'])}"
                for agent, s in sorted(perf["signal_hit_rates"].items())
            ]
        lines += [""]

    # Run health: silent failures are the enemy (A5).
    failures = conn.execute(
        "SELECT stage, ticker, detail FROM run_stages "
        "WHERE run_id = ? AND status = 'failed' ORDER BY stage, ticker",
        (run_id,),
    ).fetchall()
    freshness = conn.execute(
        "SELECT MIN(last) AS oldest, MAX(last) AS newest FROM "
        "(SELECT MAX(date) AS last FROM price_cache GROUP BY ticker)"
    ).fetchone()
    lines += ["## Run Health", ""]
    lines += [f"- Duration: {duration:.0f}s" if duration else "- Duration: n/a"]
    if failures:
        lines += [f"- Stage failures: {len(failures)}"]
        lines += [f"  - {f['stage']}/{f['ticker']}: {(f['detail'] or '')[:80]}"
                  for f in failures]
    else:
        lines += ["- Stage failures: none"]
    if freshness and freshness["oldest"]:
        lines += [f"- Price data freshness: oldest last bar {freshness['oldest']}, "
                  f"newest {freshness['newest']}"]
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")

    _write_json_report(conn, config, run_id, run_date, skipped, duration, perf)
    return path


def _write_json_report(
    conn, config: AppConfig, run_id: str, run_date: str,
    skipped: dict[str, str], duration: float, perf: dict,
) -> Path:
    """Machine-readable twin of the markdown report (dashboard, tooling)."""
    import json as _json

    decisions = [
        dict(r) for r in conn.execute(
            "SELECT ticker, action, final_score, sizing_hint, reason FROM decisions "
            "WHERE run_id = ? ORDER BY final_score DESC", (run_id,),
        ).fetchall()
    ]
    scores = [
        dict(r) for r in conn.execute(
            "SELECT ticker, agent, score, confidence, direction FROM agent_scores "
            "WHERE run_id = ? ORDER BY ticker, agent", (run_id,),
        ).fetchall()
    ]
    snapshot = conn.execute(
        "SELECT * FROM account_snapshots WHERE snapshot_date = ?", (run_date,)
    ).fetchone()
    pending = [
        dict(r) for r in conn.execute(
            "SELECT order_id, ticker, side, qty FROM orders "
            "WHERE status = 'awaiting_approval' ORDER BY ticker"
        ).fetchall()
    ]
    payload = {
        "run_id": run_id,
        "run_date": run_date,
        "duration_seconds": round(duration, 1),
        "account": dict(snapshot) if snapshot else None,
        "decisions": decisions,
        "agent_scores": scores,
        "pending_orders": pending,
        "skipped_tickers": skipped,
        "performance": perf,
    }
    path = config.reports_dir / "daily" / f"{run_date}.json"
    path.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
