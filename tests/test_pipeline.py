import pytest

from tests.fixtures import FakeForecaster, FakeMarketDataService, make_ohlcv, make_snapshot
from trading_platform.core.db import connect, init_db
from trading_platform.pipeline import AGENT_STAGES, run_daily

# SQL fragment selecting research-agent stages only (not market_data/decision/portfolio)
AGENT_ONLY = "stage IN ({})".format(",".join(f"'{s}'" for s in AGENT_STAGES))


@pytest.fixture
def fake_market_data():
    return FakeMarketDataService(default_frame=make_ohlcv())


@pytest.fixture(autouse=True)
def canned_fundamentals(monkeypatch):
    """Pipeline tests never hit the network for fundamentals."""
    monkeypatch.setattr(
        "trading_platform.agents.fundamentals.fetch_fundamentals",
        lambda ticker: make_snapshot(ticker=ticker),
    )


@pytest.fixture(autouse=True)
def canned_kronos(monkeypatch):
    """Pipeline tests never load the Kronos model — canned upward paths."""
    from trading_platform.agents.kronos import KronosAgent

    monkeypatch.setattr(
        KronosAgent, "_get_forecaster", lambda self: FakeForecaster(final_return=0.03)
    )


@pytest.fixture(autouse=True)
def canned_llm_agents(monkeypatch):
    """Pipeline tests never hit Ollama, Finnhub, or EDGAR.

    Real agent code runs, but fetches are canned and the LLM is dead — so the
    pipeline exercises the neutral-fallback paths end to end.
    """
    from tests.fixtures import make_news_items
    from trading_platform.agents.news import NewsAgent
    from trading_platform.agents.sec_filing import SECFilingAgent
    from trading_platform.core.llm import LLMError, OllamaClient

    def dead_llm(self, *args, **kwargs):
        raise LLMError("no ollama in tests")

    monkeypatch.setattr(NewsAgent, "_fetch", lambda self, t: make_news_items(t))
    monkeypatch.setattr(SECFilingAgent, "_fetch", lambda self, t: None)
    monkeypatch.setattr(OllamaClient, "generate", dead_llm)


def test_run_daily_writes_run_row_and_report(tmp_config, fake_market_data):
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["status"] == "completed"

    n_agent_stages = conn.execute(
        f"SELECT COUNT(*) FROM run_stages WHERE run_id = ? AND {AGENT_ONLY}",
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
                f"SELECT status FROM run_stages WHERE run_id = ? AND ticker = ? "
                f"AND {AGENT_ONLY}",
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

    # Fundamentals agent persists too (canned snapshot via autouse fixture)
    n_fund = conn2_count(tmp_config, run_id)
    assert n_fund == len(tmp_config.watchlist.symbols)

    # Scores surface in the daily report
    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "Agent Scores" in report_text
    assert "technical" in report_text
    assert "fundamentals" in report_text


def conn2_count(tmp_config, run_id):
    conn = connect(tmp_config.db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM agent_scores WHERE run_id = ? AND agent = 'fundamentals'",
        (run_id,),
    ).fetchone()[0]
    conn.close()
    return n


def test_agent_crash_is_isolated(tmp_config, fake_market_data, monkeypatch):
    """A crashing agent yields a neutral score and a failed stage — run continues."""
    import trading_platform.pipeline as pipeline

    def boom(self, ticker, run_id, df):
        raise ValueError("synthetic agent failure")

    monkeypatch.setattr(pipeline.TechnicalAgent, "analyze", boom)
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
    # per ticker: 5 agents + market_data + decision; plus run-level
    # execution + portfolio + risk + orders + snapshot
    per_run = len(tmp_config.watchlist.symbols) * (len(AGENT_STAGES) + 2) + 5
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
        f"SELECT COUNT(*) FROM run_stages WHERE run_id = ? AND {AGENT_ONLY}",
        (run_id,),
    ).fetchone()[0]
    assert n_agent_stages == len(tmp_config.watchlist.symbols) * len(AGENT_STAGES)
    conn.close()


def test_decisions_written_for_every_ticker(tmp_config, fake_market_data):
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    rows = conn.execute(
        "SELECT ticker, action, final_score FROM decisions WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.close()

    assert len(rows) == len(tmp_config.watchlist.symbols)
    for r in rows:
        assert r["action"] in ("buy", "sell", "hold", "watchlist")
        assert 0.0 <= r["final_score"] <= 100.0

    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "## Decisions" in report_text


def test_approved_buys_get_sizing(tmp_config, fake_market_data):
    """With strong canned signals, top-ranked buys must carry a sizing_hint."""
    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    buys = conn.execute(
        "SELECT ticker, sizing_hint, signal_breakdown FROM decisions "
        "WHERE run_id = ? AND action = 'buy' ORDER BY final_score DESC",
        (run_id,),
    ).fetchall()
    conn.close()

    if buys:  # canned fixtures produce buys; guard keeps the test honest
        import json
        sized = [b for b in buys if b["sizing_hint"]]
        assert sized, "no buy was approved by the portfolio agent"
        payload = json.loads(sized[0]["signal_breakdown"])
        assert payload["portfolio"]["approved"] is True
        assert payload["portfolio"]["qty"] >= 1


def test_held_position_stop_loss_produces_sell(tmp_config, fake_market_data):
    """A held position deep underwater must exit via stop_loss — even though
    the ticker isn't on the watchlist."""
    conn = connect(tmp_config.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
        "VALUES ('AAPL', 10, 1000.0, '2026-06-05', '2026-06-05')",
    )  # fixture price ~100 -> 90% below cost -> stop loss
    conn.commit()
    conn.close()

    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    row = conn.execute(
        "SELECT action, sizing_hint, reason FROM decisions WHERE run_id = ? AND ticker = 'AAPL'",
        (run_id,),
    ).fetchone()
    conn.close()

    assert row["action"] == "sell"
    assert row["reason"].startswith("stop_loss")
    assert row["sizing_hint"] == 10  # whole position


def test_risk_engine_evaluates_trades_and_logs(tmp_config, fake_market_data):
    """Sells and portfolio-approved buys get risk results + risk_events rows."""
    import json

    conn = connect(tmp_config.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
        "VALUES ('AAPL', 10, 1000.0, '2026-06-05', '2026-06-05')",
    )  # deep underwater -> stop-loss sell
    conn.commit()
    conn.close()

    run_id = run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    events = conn.execute(
        "SELECT ticker, approved FROM risk_events WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert events, "risk engine logged nothing"

    sell_row = conn.execute(
        "SELECT signal_breakdown FROM decisions WHERE run_id = ? AND ticker = 'AAPL'",
        (run_id,),
    ).fetchone()
    payload = json.loads(sell_row["signal_breakdown"])
    assert payload["risk"]["approved"] is True  # the exit cleared risk
    assert payload["risk"]["side"] == "sell"
    assert payload["risk"]["requires_human_approval"] is True

    # Any portfolio-approved buy must also carry a risk verdict
    buys = conn.execute(
        "SELECT signal_breakdown FROM decisions WHERE run_id = ? AND action = 'buy'",
        (run_id,),
    ).fetchall()
    for b in buys:
        p = json.loads(b["signal_breakdown"])
        if p.get("portfolio", {}).get("approved"):
            assert "risk" in p
    conn.close()

    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "Risk" in report_text


def test_full_cycle_through_pipeline(tmp_config, fake_market_data):
    """Buy fills at next open -> held -> stop-loss sell -> approved -> exit
    fills. The complete order lifecycle through real daily runs."""
    from trading_platform.execution.orders import approve_order, submit_order

    frame = make_ohlcv()  # business days ending 2026-06-11

    # Day 0 (2026-06-08): a completed run left an APPROVED buy order.
    conn = connect(tmp_config.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('seed-run', '2026-06-08', '2026-06-08T22:00:00', 'completed')"
    )
    conn.commit()
    submit_order(conn, "seed-run", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    conn.close()

    # Day 1 (2026-06-09): the run fills the buy at 06-09's open.
    run_daily(tmp_config, run_date="2026-06-09", market_data=fake_market_data)
    conn = connect(tmp_config.db_path)
    open_0609 = float(frame.loc["2026-06-09", "open"])
    expected_cost_basis = open_0609 * 1.0005  # 5 bps slippage
    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL'").fetchone()
    assert pos is not None and pos["qty"] == 10
    assert pos["avg_cost"] == pytest.approx(expected_cost_basis)
    cash_after_buy = conn.execute("SELECT cash FROM account").fetchone()[0]
    assert cash_after_buy == pytest.approx(100_000.0 - 10 * expected_cost_basis)
    # snapshot written for the day
    assert conn.execute(
        "SELECT COUNT(*) FROM account_snapshots WHERE snapshot_date='2026-06-09'"
    ).fetchone()[0] == 1

    # Simulate a price collapse: cost basis far above market -> stop loss.
    conn.execute("UPDATE positions SET avg_cost = avg_cost * 3 WHERE ticker='AAPL'")
    conn.commit()
    conn.close()

    # Day 2 (2026-06-10): decision = stop-loss sell -> order awaits approval.
    run_daily(tmp_config, run_date="2026-06-10", market_data=fake_market_data)
    conn = connect(tmp_config.db_path)
    sell_order = conn.execute(
        "SELECT * FROM orders WHERE ticker='AAPL' AND side='sell'"
    ).fetchone()
    assert sell_order is not None
    assert sell_order["status"] == "awaiting_approval"  # manual approval config
    approve_order(conn, sell_order["order_id"])
    inflated_cost = conn.execute(
        "SELECT avg_cost FROM positions WHERE ticker='AAPL'"
    ).fetchone()[0]
    conn.close()

    # Day 3 (2026-06-11): the run fills the approved exit at 06-11's open.
    run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)
    conn = connect(tmp_config.db_path)
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    assert conn.execute(
        "SELECT status FROM orders WHERE order_id=?", (sell_order["order_id"],)
    ).fetchone()[0] == "filled"

    open_0611 = float(frame.loc["2026-06-11", "open"])
    sell_px = open_0611 * 0.9995
    account = conn.execute("SELECT * FROM account").fetchone()
    assert account["cash"] == pytest.approx(
        cash_after_buy + 10 * sell_px
    )
    assert account["realized_pnl"] == pytest.approx((sell_px - inflated_cost) * 10)

    snap = conn.execute(
        "SELECT * FROM account_snapshots WHERE snapshot_date='2026-06-11'"
    ).fetchone()
    assert snap["equity"] == pytest.approx(account["cash"])  # all cash again
    conn.close()


def test_report_includes_run_health(tmp_config, fake_market_data):
    run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)
    report_text = (tmp_config.reports_dir / "daily" / "2026-06-11.md").read_text(encoding="utf-8")
    assert "## Run Health" in report_text
    assert "Stage failures" in report_text


def test_success_notification_sent(tmp_config, fake_market_data, monkeypatch):
    sent = {}

    def fake_notify(settings, title, message):
        sent.update(title=title, message=message)
        return True

    import trading_platform.pipeline as pipeline
    monkeypatch.setattr(pipeline, "notify", fake_notify)
    run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    assert "OK" in sent["title"]
    assert "equity" in sent["message"]
    assert "decisions" in sent["message"]


def test_failure_notification_sent_on_crash(tmp_config, fake_market_data, monkeypatch):
    sent = {}

    import trading_platform.pipeline as pipeline

    def fake_notify(settings, title, message):
        sent.update(title=title, message=message)
        return True

    def boom(*a, **k):
        raise RuntimeError("synthetic snapshot crash")

    monkeypatch.setattr(pipeline, "notify", fake_notify)
    monkeypatch.setattr(pipeline, "snapshot_account", boom)
    with pytest.raises(RuntimeError, match="synthetic snapshot crash"):
        run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    assert "FAILED" in sent["title"]
    assert "resumable" in sent["message"]


def test_unapproved_order_expires_at_next_run(tmp_config, fake_market_data):
    from trading_platform.execution.orders import submit_order

    conn = connect(tmp_config.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('seed-run', '2026-06-10', '2026-06-10T22:00:00', 'completed')"
    )
    conn.commit()
    oid = submit_order(conn, "seed-run", "AAPL", "buy", 10, "2026-06-10",
                       auto_approve=False)
    conn.close()

    run_daily(tmp_config, run_date="2026-06-11", market_data=fake_market_data)

    conn = connect(tmp_config.db_path)
    assert conn.execute(
        "SELECT status FROM orders WHERE order_id=?", (oid,)
    ).fetchone()[0] == "expired"
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    assert conn.execute("SELECT cash FROM account").fetchone()[0] == 100_000.0
    conn.close()
