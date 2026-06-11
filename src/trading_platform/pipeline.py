"""Daily pipeline orchestrator.

Phase 0: skeleton only — creates (or resumes) a run, walks the watchlist with
no-op research stages, writes a report stub, and completes the run. Research
agents plug in via AGENT_STAGES in later phases; the run/resume/idempotency
machinery they rely on is exercised here from day one.
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

logger = logging.getLogger(__name__)

# Agent names, in fan-out order. Each maps to a callable in later phases.
AGENT_STAGES = ["technical", "kronos", "fundamentals", "news", "sec_filing"]


def run_daily(config: AppConfig, run_date: str | None = None) -> str:
    """Execute one daily analysis run. Returns the run_id.

    Idempotent: re-invoking for the same date resumes the incomplete run and
    skips stages already completed.
    """
    run_date = run_date or date.today().isoformat()
    conn = connect(config.db_path)
    init_db(conn)

    run_id = find_resumable_run(conn, run_date)
    if run_id:
        logger.info("resuming run %s for %s", run_id, run_date)
    else:
        run_id = create_run(conn, run_date)
        logger.info("created run %s for %s", run_id, run_date)

    try:
        for symbol in config.watchlist.symbols:
            for stage in AGENT_STAGES:
                if stage_status(conn, run_id, stage, symbol) == "completed":
                    continue
                mark_stage(conn, run_id, stage, "running", ticker=symbol)
                # Phase 0: no-op. Later phases dispatch to the real agent here
                # and persist its AgentResult via save_agent_result().
                mark_stage(conn, run_id, stage, "completed", ticker=symbol,
                           detail="no-op (phase 0)")

        _write_report_stub(config, run_id, run_date)
        complete_run(conn, run_id)
        logger.info("run %s completed", run_id)
    except Exception:
        complete_run(conn, run_id, status="partial", notes="crashed mid-run; resumable")
        raise
    finally:
        conn.close()
    return run_id


def _write_report_stub(config: AppConfig, run_id: str, run_date: str) -> Path:
    report_dir = config.reports_dir / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{run_date}.md"
    tickers = ", ".join(config.watchlist.symbols)
    path.write_text(
        f"# Daily Research Report — {run_date}\n\n"
        f"Run ID: `{run_id}`\n\n"
        f"Watchlist: {tickers}\n\n"
        "_Pipeline skeleton (phase 0): agent scores, decisions, and account "
        "summary will appear here as phases land._\n",
        encoding="utf-8",
    )
    return path
