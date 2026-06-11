import pytest

from tests.fixtures import FakeMarketDataService, make_ohlcv
from trading_platform.core.db import connect
from trading_platform.pipeline import AGENT_STAGES, run_daily


@pytest.fixture
def fake_market_data():
    return FakeMarketDataService(default_frame=make_ohlcv())


def test_run_daily_writes_run_row_and_report(tmp_config, fake_market_data):
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"

    n_agent_stages = conn.execute(
        "SELECT COUNT(*) FROM run_stages WHERE run_id = ? AND stage != 'market_data'",
        (run_id,),
    ).fetchone()[0]
    assert n_agent_stages == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)

    n_data_stages = conn.execute(
        "SELECT COUNT(*) FROM run_stages WHERE run_id = ? AND stage = 'market_data' "
        "AND status = 'completed'",
        (run_id,),
    ).fetchone()[0]
    assert n_data_stages == len(tmp_config.watchlist.symbols)
    conn.close()

    report = tmp_config.reports_dir / "daily" / "2026-06-11.md"
    assert report.exists()
    assert run_id in report.read_text(encoding="utf-8")


def test_data_gate_failure_skips_agents_not_run(tmp_config):
    bad = make_ohlcv()
    bad.iloc[-5:, bad.columns.get_loc("volume")] = 0  # fails recent_volume check
    market_data = FakeMarketDataService(
        default_frame=make_ohlcv(),
        frames={"NVDA": bad},
        fetch_errors={"BRK-B": "no data returned"},
    )
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=market_data)

    conn = connect(tmp_config.db_path)
    for ticker, expected in [("NVDA", "skipped"), ("BRK-B", "skipped"), ("AAPL", "completed")]:
        statuses = {
            r["status"]
            for r in conn.execute(
                "SELECT status FROM run_stages WHERE run_id = ? AND ticker = ? "
                "AND stage != 'market_data'",
                (run_id, ticker),
            ).fetchall()
        }
        assert statuses == {expected}, f"{ticker}: {statuses}"

    gate = {
        r["ticker"]: r["status"]
        for r in conn.execute(
            "SELECT ticker, status FROM run_stages WHERE run_id = ? AND stage = 'market_data'",
            (run_id,),
        ).fetchall()
    }
    assert gate["NVDA"] == "failed"
    assert gate["BRK-B"] == "failed"
    assert gate["AAPL"] == "completed"
    conn.close()

    # Flagged tickers must be visible in the report (A5: no silent failures).
    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "Data Quality Flags" in report_text
    assert "NVDA" in report_text
    assert "BRK-B" in report_text


def test_technical_agent_scores_persisted(tmp_config, fake_market_data):
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    rows = conn.execute(
        "SELECT * FROM agent_scores WHERE run_id = ? AND agent = 'technical'", (run_id,)
    ).fetchall()
    conn.close()

    assert len(rows) == len(tmp_config.watchlist.symbols)
    for r in rows:
        assert 0.0 <= r["score"] <= 100.0
        assert r["direction"] in ("bullish", "bearish", "neutral")

    # Scores surface in the daily report
    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "Agent Scores" in report_text
    assert "technical" in report_text


def test_agent_crash_is_isolated(tmp_config, fake_market_data, monkeypatch):
    """A crashing agent yields a neutral score and a failed stage — run continues."""
    import trading_platform.pipeline as pipeline

    def boom(self, ticker, run_id, df):
        raise ValueError("synthetic agent failure")

    monkeypatch.setattr(type(pipeline.AGENT_REGISTRY["technical"]), "analyze", boom)
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    run = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert run["status"] == "completed"  # the run survived

    rows = conn.execute(
        "SELECT * FROM agent_scores WHERE run_id = ? AND agent = 'technical'", (run_id,)
    ).fetchall()
    assert len(rows) == len(tmp_config.watchlist.symbols)
    for r in rows:
        assert r["score"] == 50.0
        assert r["confidence"] == 0.0  # ignorable by the decision layer

    stages = conn.execute(
        "SELECT status FROM run_stages WHERE run_id = ? AND stage = 'technical'", (run_id,)
    ).fetchall()
    assert {s["status"] for s in stages} == {"failed"}
    conn.close()


def test_rerunning_same_date_does_not_duplicate(tmp_config, fake_market_data):
    run_id_1 = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)
    run_id_2 = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)
    # First run completed, so the second invocation is a NEW run (fresh analysis),
    # but stage rows must be scoped per-run — no cross-contamination.
    assert run_id_1 != run_id_2

    conn = connect(tmp_config.db_path)
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_runs == 2
    per_run = len(tmp_config.watchlist.symbols) * (len(AGENT_STAGES) + 1)  # +1 market_data
    for rid in (run_id_1, run_id_2):
        n = conn.execute(
            "SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (rid,)
        ).fetchone()[0]
        assert n == per_run
    conn.close()


def test_incomplete_run_is_resumed_not_duplicated(tmp_config, fake_market_data, monkeypatch):
    # Simulate a crash mid-run: first invocation dies after 3 stage completions.
    import trading_platform.pipeline as pipeline

    original_mark = pipeline.mark_stage
    calls = {"completed": 0}

    def crashing_mark(conn, run_id, stage, status, ticker="", detail=None):
        original_mark(conn, run_id, stage, status, ticker=ticker, detail=detail)
        if status == "completed" and stage in AGENT_STAGES:
            calls["completed"] += 1
            if calls["completed"] >= 3:
                raise RuntimeError("simulated crash")

    monkeypatch.setattr(pipeline, "mark_stage", crashing_mark)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)
    monkeypatch.setattr(pipeline, "mark_stage", original_mark)

    # Second invocation must resume the SAME run and finish it.
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_runs == 1  # resumed, not restarted
    row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"
    n_agent_stages = conn.execute(
        "SELECT COUNT(*) FROM run_stages WHERE run_id = ? AND stage != 'market_data'",
        (run_id,),
    ).fetchone()[0]
    assert n_agent_stages == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)
    conn.close()
