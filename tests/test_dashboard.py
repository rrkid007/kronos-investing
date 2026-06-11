"""Dashboard — routes, rendering, and the approval workflow in the browser."""

import pytest
from fastapi.testclient import TestClient

from trading_platform.core.db import connect, init_db
from trading_platform.dashboard.app import create_app, equity_svg
from trading_platform.execution.orders import submit_order
from trading_platform.execution.paper_broker import ensure_account, snapshot_account


@pytest.fixture
def seeded(tmp_config):
    """A populated database: run, scores, decision, position, order, snapshots."""
    conn = connect(tmp_config.db_path)
    init_db(conn)
    ensure_account(conn, 100_000.0)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('r1', '2026-06-10', '2026-06-10T22:00:00', 'completed')"
    )
    conn.execute(
        "INSERT INTO agent_scores (run_id, agent, ticker, score, confidence, direction, "
        "details, created_at) VALUES ('r1', 'technical', 'AAPL', 81.0, 0.9, 'bullish', '{}', '')"
    )
    conn.execute(
        "INSERT INTO decisions (run_id, ticker, action, final_score, signal_breakdown, "
        "sizing_hint, reason, created_at) VALUES "
        "('r1', 'AAPL', 'buy', 78.5, '{\"risk\": {\"approved\": true, \"checks\": []}}', "
        "9500, 'final score 78.5 >= buy threshold', '')"
    )
    conn.execute(
        "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
        "VALUES ('MSFT', 20, 400.0, '2026-06-01', '2026-06-01')"
    )
    conn.execute(
        "INSERT INTO price_cache (ticker, date, close, adj_close) "
        "VALUES ('MSFT', '2026-06-10', 420.0, 420.0)"
    )
    conn.commit()
    submit_order(conn, "r1", "AAPL", "buy", 25, "2026-06-10", auto_approve=False,
                 context={"final_score": 78.5, "est_value": 9500.0, "reason": "buy signal"})
    snapshot_account(conn, "2026-06-09", closes={})
    snapshot_account(conn, "2026-06-10", closes={"MSFT": 420.0})
    conn.close()
    return tmp_config


@pytest.fixture
def client(seeded):
    return TestClient(create_app(seeded))


def get_order_status(config, order_id=None):
    conn = connect(config.db_path)
    row = conn.execute("SELECT order_id, status FROM orders").fetchone()
    conn.close()
    return row["order_id"], row["status"]


def test_index_renders_all_sections(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    for fragment in ("Account", "Pending Approvals", "Decisions", "Agent Leaderboard",
                     "Run Health", "Open Positions", "AAPL", "MSFT"):
        assert fragment in html, fragment
    assert "100,0" in html  # equity rendered
    assert "<svg" in html   # equity curve (2 snapshots)


def test_approve_in_browser(seeded, client):
    order_id, status = get_order_status(seeded)
    assert status == "awaiting_approval"

    resp = client.post(f"/orders/{order_id}/approve")
    assert resp.status_code == 200
    assert "no orders awaiting approval" in resp.text  # partial re-rendered empty
    assert "approved" in resp.text  # shows in recent activity

    _, status = get_order_status(seeded)
    assert status == "approved"


def test_reject_in_browser(seeded, client):
    order_id, _ = get_order_status(seeded)
    resp = client.post(f"/orders/{order_id}/reject")
    assert resp.status_code == 200
    _, status = get_order_status(seeded)
    assert status == "rejected"


def test_double_approve_shows_flash_not_crash(seeded, client):
    order_id, _ = get_order_status(seeded)
    client.post(f"/orders/{order_id}/approve")
    resp = client.post(f"/orders/{order_id}/approve")
    assert resp.status_code == 200
    assert "not awaiting_approval" in resp.text  # flash message

    _, status = get_order_status(seeded)
    assert status == "approved"  # unchanged


def test_unknown_order_flash(client):
    resp = client.post("/orders/doesnotexist/approve")
    assert resp.status_code == 200
    assert "not found" in resp.text


def test_reports_browser(seeded, client):
    report_dir = seeded.reports_dir / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "2026-06-10.md").write_text("# Daily Research Report — test",
                                              encoding="utf-8")
    resp = client.get("/reports")
    assert resp.status_code == 200
    assert "2026-06-10" in resp.text

    resp = client.get("/reports/2026-06-10")
    assert resp.status_code == 200
    assert "Daily Research Report" in resp.text


def test_report_path_traversal_blocked(seeded, client):
    resp = client.get("/reports/..%2F..%2Fsettings")
    assert resp.status_code == 404


def test_partials_endpoint(client):
    resp = client.get("/partials/approvals")
    assert resp.status_code == 200
    assert "Pending Approvals" in resp.text


def test_equity_svg_rendering():
    assert equity_svg([]) == ""
    assert equity_svg([("2026-06-10", 100.0)]) == ""
    svg = equity_svg([("2026-06-09", 100.0), ("2026-06-10", 110.0), ("2026-06-11", 105.0)])
    assert svg.startswith("<svg")
    assert "polyline" in svg
    assert "var(--up)" in svg  # ended above start
    down = equity_svg([("2026-06-09", 100.0), ("2026-06-10", 90.0)])
    assert "var(--down)" in down
