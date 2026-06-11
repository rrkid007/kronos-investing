"""Daily pipeline orchestrator.

Phase 1: market data refresh + quality gate run per ticker ahead of the
research stages. A ticker that fails the gate has all its agent stages
skipped and is flagged in the report — agents never score bad data.
Research agents themselves are still no-ops; they plug in via AGENT_STAGES
in later phases.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from trading_platform.core.config import AppConfig
from trading_platform.core.db import connect, init_db
from trading_platform.core.runs import (
    complete_run,
    create_run,
    find_resumable_run,
    mark_stage,
    stage_status,
)
from trading_platform.data.market_data import MarketDataService

logger = logging.getLogger(__name__)

# Agent names, in fan-out order. Each maps to a callable in later phases.
AGENT_STAGES = ["technical", "kronos", "fundamentals", "news", "sec_filing"]
DATA_STAGE = "market_data"


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

    skipped: dict[str, str] = {}  # ticker -> reason, for the report
    try:
        for symbol in config.watchlist.symbols:
            data_ok = _ensure_market_data(
                conn, market_data, run_id, symbol, date.fromisoformat(run_date), skipped
            )
            for stage in AGENT_STAGES:
                if stage_status(conn, run_id, stage, symbol) == "completed":
                    continue
                if not data_ok:
                    mark_stage(conn, run_id, stage, "skipped", ticker=symbol,
                               detail=skipped.get(symbol, "market data unavailable"))
                    continue
                mark_stage(conn, run_id, stage, "running", ticker=symbol)
                # Phase 1: no-op. Later phases dispatch to the real agent here
                # and persist its AgentResult via save_agent_result().
                mark_stage(conn, run_id, stage, "completed", ticker=symbol,
                           detail="no-op (phase 1)")

        _write_report_stub(config, run_id, run_date, skipped)
        complete_run(conn, run_id)
        logger.info("run %s completed (%d tickers skipped)", run_id, len(skipped))
    except Exception:
        complete_run(conn, run_id, status="partial", notes="crashed mid-run; resumable")
        raise
    finally:
        conn.close()
    return run_id


def _ensure_market_data(
    conn,
    market_data: MarketDataService,
    run_id: str,
    symbol: str,
    as_of: date,
    skipped: dict[str, str],
) -> bool:
    """Refresh + validate one ticker's data. Returns True if agents may score it."""
    if stage_status(conn, run_id, DATA_STAGE, symbol) == "completed":
        return True  # resumed run; gate already passed for this ticker

    mark_stage(conn, run_id, DATA_STAGE, "running", ticker=symbol)
    data = market_data.refresh_and_validate(symbol, as_of=as_of)
    if data.ok:
        mark_stage(conn, run_id, DATA_STAGE, "completed", ticker=symbol, detail=data.detail())
        return True

    logger.warning("data gate failed for %s: %s", symbol, data.detail())
    mark_stage(conn, run_id, DATA_STAGE, "failed", ticker=symbol, detail=data.detail())
    skipped[symbol] = data.detail()
    return False


def _write_report_stub(
    config: AppConfig, run_id: str, run_date: str, skipped: dict[str, str]
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
    if skipped:
        lines += ["## Data Quality Flags", ""]
        lines += [f"- **{t}** — skipped: {reason}" for t, reason in sorted(skipped.items())]
        lines += [""]
    lines += [
        "_Pipeline (phase 1): agent scores, decisions, and account summary "
        "will appear here as phases land._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
