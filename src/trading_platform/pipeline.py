"""Daily pipeline orchestrator.

Flow per ticker: market data refresh + quality gate, then each registered
agent scores the ticker and persists an AgentResult. Stages without a
registered agent are no-ops until their phase lands. An agent that crashes is
isolated — the stage is marked failed, a neutral zero-confidence score is
stored so the decision layer can ignore it, and the run continues.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from trading_platform.agents.fundamentals import FundamentalsAgent
from trading_platform.agents.kronos import KronosAgent
from trading_platform.agents.news import NewsAgent
from trading_platform.agents.sec_filing import SECFilingAgent
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.core.llm import OllamaClient
from trading_platform.core.config import AppConfig
from trading_platform.core.db import connect, init_db
from trading_platform.core.models import AgentResult
from trading_platform.core.runs import (
    complete_run,
    create_run,
    find_resumable_run,
    mark_stage,
    save_agent_result,
    stage_status,
)
from trading_platform.data.market_data import MarketDataService

logger = logging.getLogger(__name__)

# Research stages in fan-out order. Stages without a registered agent are
# no-ops until their phase lands.
AGENT_STAGES = ["technical", "kronos", "fundamentals", "news", "sec_filing"]
DATA_STAGE = "market_data"


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
    skipped: dict[str, str] = {}  # ticker -> reason, for the report
    try:
        for symbol in config.watchlist.symbols:
            df = _ensure_market_data(
                conn, market_data, run_id, symbol, date.fromisoformat(run_date), skipped
            )
            for stage in AGENT_STAGES:
                if stage_status(conn, run_id, stage, symbol) == "completed":
                    continue
                if df is None:
                    mark_stage(conn, run_id, stage, "skipped", ticker=symbol,
                               detail=skipped.get(symbol, "market data unavailable"))
                    continue
                _run_agent_stage(conn, registry, run_id, stage, symbol, df)

        _write_report(conn, config, run_id, run_date, skipped)
        complete_run(conn, run_id)
        logger.info("run %s completed (%d tickers skipped)", run_id, len(skipped))
    except Exception:
        complete_run(conn, run_id, status="partial", notes="crashed mid-run; resumable")
        raise
    finally:
        conn.close()
    return run_id


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

    rows = conn.execute(
        "SELECT ticker, agent, score, confidence, direction FROM agent_scores "
        "WHERE run_id = ? ORDER BY score DESC, ticker",
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
        "_Decisions, risk results, and account summary will appear here as "
        "phases land._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
