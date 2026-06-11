"""Run lifecycle: every pipeline execution is tracked, idempotent, resumable.

A run is identified by run_id and tied to a run_date (the trading day being
analyzed). Re-running a date resumes the existing incomplete run instead of
starting a duplicate; completed stages are skipped. All score/decision writes
are upserts keyed by (run_id, ...), so a resumed run never duplicates rows.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

from trading_platform.core.models import AgentResult, TradeDecision, utcnow


def _now() -> str:
    return utcnow().isoformat()


def new_run_id(run_date: str) -> str:
    return f"{run_date}-{uuid.uuid4().hex[:8]}"


def create_run(conn: sqlite3.Connection, run_date: str) -> str:
    run_id = new_run_id(run_date)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) VALUES (?, ?, ?, 'running')",
        (run_id, run_date, _now()),
    )
    conn.commit()
    return run_id


def find_resumable_run(conn: sqlite3.Connection, run_date: str) -> str | None:
    """Return the most recent incomplete run for this date, if any."""
    row = conn.execute(
        "SELECT run_id FROM runs WHERE run_date = ? AND status IN ('running','partial') "
        "ORDER BY started_at DESC LIMIT 1",
        (run_date,),
    ).fetchone()
    return row["run_id"] if row else None


def mark_stage(
    conn: sqlite3.Connection,
    run_id: str,
    stage: str,
    status: str,
    ticker: str = "",
    detail: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO run_stages (run_id, stage, ticker, status, detail, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, stage, ticker)
        DO UPDATE SET status = excluded.status, detail = excluded.detail,
                      updated_at = excluded.updated_at
        """,
        (run_id, stage, ticker, status, detail, _now()),
    )
    conn.commit()


def stage_status(
    conn: sqlite3.Connection, run_id: str, stage: str, ticker: str = ""
) -> str | None:
    row = conn.execute(
        "SELECT status FROM run_stages WHERE run_id = ? AND stage = ? AND ticker = ?",
        (run_id, stage, ticker),
    ).fetchone()
    return row["status"] if row else None


def complete_run(
    conn: sqlite3.Connection, run_id: str, status: str = "completed", notes: str | None = None
) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ?, notes = ? WHERE run_id = ?",
        (status, _now(), notes, run_id),
    )
    conn.commit()


def save_agent_result(conn: sqlite3.Connection, result: AgentResult) -> None:
    """Upsert an agent score — re-running a stage overwrites, never duplicates."""
    conn.execute(
        """
        INSERT INTO agent_scores
            (run_id, agent, ticker, score, confidence, direction, details, data_as_of, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, agent, ticker)
        DO UPDATE SET score = excluded.score, confidence = excluded.confidence,
                      direction = excluded.direction, details = excluded.details,
                      data_as_of = excluded.data_as_of, created_at = excluded.created_at
        """,
        (
            result.run_id,
            result.agent,
            result.ticker,
            result.score,
            result.confidence,
            result.direction.value,
            json.dumps(result.details),
            result.data_as_of.isoformat() if result.data_as_of else None,
            _now(),
        ),
    )
    conn.commit()


def save_decision(conn: sqlite3.Connection, decision: TradeDecision) -> None:
    conn.execute(
        """
        INSERT INTO decisions
            (run_id, ticker, action, final_score, signal_breakdown, sizing_hint, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, ticker)
        DO UPDATE SET action = excluded.action, final_score = excluded.final_score,
                      signal_breakdown = excluded.signal_breakdown,
                      sizing_hint = excluded.sizing_hint, reason = excluded.reason,
                      created_at = excluded.created_at
        """,
        (
            decision.run_id,
            decision.ticker,
            decision.action.value,
            decision.final_score,
            json.dumps(decision.signal_breakdown),
            decision.sizing_hint,
            decision.reason,
            _now(),
        ),
    )
    conn.commit()
