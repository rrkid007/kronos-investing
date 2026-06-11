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
from datetime import date
from pathlib import Path

import pandas as pd

from trading_platform.agents.decision import DecisionEngine
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
from trading_platform.risk.engine import RiskEngine, log_risk_event

logger = logging.getLogger(__name__)

# Research stages in fan-out order. Stages without a registered agent are
# no-ops until their phase lands.
AGENT_STAGES = ["technical", "kronos", "fundamentals", "news", "sec_filing"]
DATA_STAGE = "market_data"
DECISION_STAGE = "decision"
PORTFOLIO_STAGE = "portfolio"
RISK_STAGE = "risk"


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
    conn = connect(config.db_path)
    init_db(conn)
    if market_data is None:
        market_data = MarketDataService(conn)

    run_id = find_resumable_run(conn, run_date)
    if run_id:
        logger.info("resuming run %s for %s", run_id, run_date)
    else:
        run_id = create_run(conn, run_date)
        logger.info("created run %s for %s", run_id, run_date)

    registry = build_agent_registry(config, conn=conn)
    positions = load_positions(conn)
    held_by_ticker = {p.ticker: p for p in positions}
    # Held tickers are always re-evaluated, even off-watchlist (exit policy).
    symbols = list(config.watchlist.symbols)
    symbols += [t for t in held_by_ticker if t not in symbols]

    skipped: dict[str, str] = {}  # ticker -> reason, for the report
    frames: dict[str, pd.DataFrame] = {}
    try:
        # --- pass 1: data + research agents
        for symbol in symbols:
            df = _ensure_market_data(conn, market_data, run_id, symbol, as_of, skipped)
            if df is not None:
                frames[symbol] = df
            for stage in AGENT_STAGES:
                if stage_status(conn, run_id, stage, symbol) == "completed":
                    continue
                if df is None:
                    mark_stage(conn, run_id, stage, "skipped", ticker=symbol,
                               detail=skipped.get(symbol, "market data unavailable"))
                    continue
                _run_agent_stage(conn, registry, run_id, stage, symbol, df)

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

        _write_report(conn, config, run_id, run_date, skipped)
        complete_run(conn, run_id)
        logger.info("run %s completed (%d tickers skipped)", run_id, len(skipped))
    except Exception:
        complete_run(conn, run_id, status="partial", notes="crashed mid-run; resumable")
        raise
    finally:
        conn.close()
    return run_id


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


def _ensure_market_data(
    conn,
    market_data: MarketDataService,
    run_id: str,
    symbol: str,
    as_of: date,
    skipped: dict[str, str],
) -> pd.DataFrame | None:
    """Refresh + validate one ticker. Returns its frame, or None if unscoreable."""
    if stage_status(conn, run_id, DATA_STAGE, symbol) == "completed":
        return market_data.load(symbol)  # resumed run; gate already passed

    mark_stage(conn, run_id, DATA_STAGE, "running", ticker=symbol)
    data = market_data.refresh_and_validate(symbol, as_of=as_of)
    if data.ok:
        mark_stage(conn, run_id, DATA_STAGE, "completed", ticker=symbol, detail=data.detail())
        return data.df

    logger.warning("data gate failed for %s: %s", symbol, data.detail())
    mark_stage(conn, run_id, DATA_STAGE, "failed", ticker=symbol, detail=data.detail())
    skipped[symbol] = data.detail()
    return None


def _write_report(
    conn, config: AppConfig, run_id: str, run_date: str, skipped: dict[str, str]
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

    lines += [
        "_Risk results and account summary will appear here as phases land._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
