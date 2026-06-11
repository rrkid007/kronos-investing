from trading_platform.core.db import connect
from trading_platform.pipeline import AGENT_STAGES, run_daily


def test_run_daily_writes_run_row_and_report(tmp_config):
    run_id = run_daily(tmp_config, run_date="2026-06-11")

    conn = connect(tmp_config.db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"

    n_stages = conn.execute("SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n_stages == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)
    conn.close()

    report = tmp_config.reports_dir / "daily" / "2026-06-11.md"
    assert report.exists()
    assert run_id in report.read_text(encoding="utf-8")


def test_rerunning_same_date_does_not_duplicate(tmp_config):
    run_id_1 = run_daily(tmp_config, run_date="2026-06-11")
    run_id_2 = run_daily(tmp_config, run_date="2026-06-11")
    # First run completed, so the second invocation is a NEW run (fresh analysis),
    # but stage rows must be scoped per-run — no cross-contamination.
    assert run_id_1 != run_id_2

    conn = connect(tmp_config.db_path)
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_runs == 2
    for rid in (run_id_1, run_id_2):
        n = conn.execute("SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (rid,)).fetchone()[0]
        assert n == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)
    conn.close()


def test_incomplete_run_is_resumed_not_duplicated(tmp_config, monkeypatch):
    # Simulate a crash mid-run: first invocation dies after 3 stage completions.
    import trading_platform.pipeline as pipeline

    original_mark = pipeline.mark_stage
    calls = {"completed": 0}

    def crashing_mark(conn, run_id, stage, status, ticker="", detail=None):
        original_mark(conn, run_id, stage, status, ticker=ticker, detail=detail)
        if status == "completed":
            calls["completed"] += 1
            if calls["completed"] >= 3:
                raise RuntimeError("simulated crash")

    monkeypatch.setattr(pipeline, "mark_stage", crashing_mark)
    try:
        run_daily(tmp_config, run_date="2026-06-11")
    except RuntimeError:
        pass
    monkeypatch.setattr(pipeline, "mark_stage", original_mark)

    # Second invocation must resume the SAME run and finish it.
    run_id = run_daily(tmp_config, run_date="2026-06-11")

    conn = connect(tmp_config.db_path)
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_runs == 1  # resumed, not restarted
    row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"
    n_stages = conn.execute("SELECT COUNT(*) FROM run_stages WHERE run_id = ?", (run_id,)).fetchone()[0]
    assert n_stages == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)
    conn.close()
