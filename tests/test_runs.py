import pytest

from trading_platform.core.db import connect, init_db
from trading_platform.core.models import Action, AgentResult, Direction, TradeDecision
from trading_platform.core.runs import (
    complete_run,
    create_run,
    find_resumable_run,
    mark_stage,
    save_agent_result,
    save_decision,
    stage_status,
)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    init_db(c)
    yield c
    c.close()


def test_run_lifecycle(conn):
    run_id = create_run(conn, "2026-06-11")
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "running"

    complete_run(conn, run_id)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"
    assert row["finished_at"] is not None


def test_resume_finds_incomplete_run_only(conn):
    run_id = create_run(conn, "2026-06-11")
    assert find_resumable_run(conn, "2026-06-11") == run_id
    assert find_resumable_run(conn, "2026-06-12") is None

    complete_run(conn, run_id)
    assert find_resumable_run(conn, "2026-06-11") is None


def test_stage_tracking_upserts(conn):
    run_id = create_run(conn, "2026-06-11")
    mark_stage(conn, run_id, "technical", "running", ticker="AAPL")
    assert stage_status(conn, run_id, "technical", "AAPL") == "running"

    mark_stage(conn, run_id, "technical", "completed", ticker="AAPL")
    assert stage_status(conn, run_id, "technical", "AAPL") == "completed"

    count = conn.execute("SELECT COUNT(*) FROM run_stages").fetchone()[0]
    assert count == 1  # upsert, not duplicate rows


def test_agent_result_upsert_is_idempotent(conn):
    run_id = create_run(conn, "2026-06-11")
    result = AgentResult(
        agent="technical", ticker="AAPL", run_id=run_id,
        score=90.0, confidence=0.8, direction=Direction.BULLISH,
        details={"trend": "uptrend"},
    )
    save_agent_result(conn, result)
    save_agent_result(conn, result.model_copy(update={"score": 85.0}))

    rows = conn.execute("SELECT * FROM agent_scores").fetchall()
    assert len(rows) == 1
    assert rows[0]["score"] == 85.0


def test_decision_upsert_is_idempotent(conn):
    run_id = create_run(conn, "2026-06-11")
    decision = TradeDecision(
        run_id=run_id, ticker="AAPL", action=Action.BUY,
        final_score=78.0, signal_breakdown={"technical": 90.0},
    )
    save_decision(conn, decision)
    save_decision(conn, decision.model_copy(update={"action": Action.WATCHLIST}))

    rows = conn.execute("SELECT * FROM decisions").fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "watchlist"
