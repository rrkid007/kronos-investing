"""Order de-duplication: re-running a day must not stack duplicate orders.

Covers the two guards in execution.orders:
- submit_order blocks a cross-run duplicate while an order for the same
  ticker+side is still open (awaiting_approval or approved).
- expire_stale_orders, given the current run_id, supersedes other runs'
  pending orders for the same date (and all earlier dates).
"""

import pytest

from trading_platform.core.db import connect, init_db
from trading_platform.execution.orders import expire_stale_orders, submit_order


@pytest.fixture
def conn(tmp_config):
    c = connect(tmp_config.db_path)
    init_db(c)
    return c


def _run(conn, run_id, run_date):
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES (?, ?, ?, 'completed')",
        (run_id, run_date, run_date + "T22:00"),
    )
    conn.commit()


def _open(conn, ticker="AAPL", side="buy"):
    return conn.execute(
        "SELECT COUNT(*) c FROM orders WHERE ticker=? AND side=? "
        "AND status IN ('awaiting_approval','approved')",
        (ticker, side),
    ).fetchone()["c"]


def test_cross_run_open_order_blocks_duplicate(conn):
    _run(conn, "A", "2026-06-18")
    _run(conn, "B", "2026-06-18")
    a = submit_order(conn, "A", "AAPL", "buy", 10, "2026-06-18", auto_approve=False)
    b = submit_order(conn, "B", "AAPL", "buy", 10, "2026-06-18", auto_approve=False)
    assert a == b
    assert _open(conn) == 1


def test_approved_order_blocks_new_buy(conn):
    _run(conn, "A", "2026-06-18")
    _run(conn, "B", "2026-06-19")
    a = submit_order(conn, "A", "AAPL", "buy", 10, "2026-06-18", auto_approve=False)
    conn.execute("UPDATE orders SET status='approved' WHERE order_id=?", (a,))
    conn.commit()
    b = submit_order(conn, "B", "AAPL", "buy", 10, "2026-06-19", auto_approve=False)
    assert b == a
    assert _open(conn) == 1


def test_filled_order_allows_reentry(conn):
    _run(conn, "A", "2026-06-18")
    _run(conn, "B", "2026-06-25")
    a = submit_order(conn, "A", "AAPL", "buy", 10, "2026-06-18", auto_approve=False)
    conn.execute("UPDATE orders SET status='filled' WHERE order_id=?", (a,))
    conn.commit()
    b = submit_order(conn, "B", "AAPL", "buy", 10, "2026-06-25", auto_approve=False)
    assert b != a  # a closed/filled position may be re-bought later


def test_same_date_supersede_expires_other_runs(conn):
    _run(conn, "A", "2026-06-18")
    _run(conn, "B", "2026-06-18")
    submit_order(conn, "A", "MSFT", "buy", 5, "2026-06-18", auto_approve=False)
    submit_order(conn, "B", "NVDA", "buy", 5, "2026-06-18", auto_approve=False)
    n = expire_stale_orders(conn, "2026-06-18", current_run_id="B")
    assert n == 1
    assert conn.execute("SELECT status FROM orders WHERE run_id='A'").fetchone()[0] == "expired"
    assert conn.execute("SELECT status FROM orders WHERE run_id='B'").fetchone()[0] == "awaiting_approval"


def test_approved_orders_never_expired(conn):
    _run(conn, "A", "2026-06-18")
    _run(conn, "B", "2026-06-18")
    a = submit_order(conn, "A", "MSFT", "buy", 5, "2026-06-18", auto_approve=False)
    conn.execute("UPDATE orders SET status='approved' WHERE order_id=?", (a,))
    conn.commit()
    expire_stale_orders(conn, "2026-06-18", current_run_id="B")
    assert conn.execute("SELECT status FROM orders WHERE order_id=?", (a,)).fetchone()[0] == "approved"


def test_legacy_two_arg_expiry_still_earlier_only(conn):
    _run(conn, "A", "2026-06-17")
    _run(conn, "B", "2026-06-18")
    submit_order(conn, "A", "MSFT", "buy", 5, "2026-06-17", auto_approve=False)
    submit_order(conn, "B", "NVDA", "buy", 5, "2026-06-18", auto_approve=False)
    n = expire_stale_orders(conn, "2026-06-18")  # no run_id → earlier dates only
    assert n == 1
    assert conn.execute("SELECT status FROM orders WHERE run_id='A'").fetchone()[0] == "expired"
    assert conn.execute("SELECT status FROM orders WHERE run_id='B'").fetchone()[0] == "awaiting_approval"
