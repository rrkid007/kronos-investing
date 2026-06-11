"""Paper broker + order lifecycle — exact accounting math, every transition."""

from datetime import date

import pytest

from trading_platform.core.config import FillModel
from trading_platform.core.db import connect, init_db
from trading_platform.execution.orders import (
    approve_order,
    expire_stale_orders,
    fillable_orders,
    list_pending,
    reject_order,
    submit_order,
)
from trading_platform.execution.paper_broker import (
    FillError,
    ensure_account,
    fill_order,
    get_account,
    snapshot_account,
)

FILL = FillModel(slippage_bps=5.0, commission_per_trade=0.0)
FILL_WITH_COMMISSION = FillModel(slippage_bps=5.0, commission_per_trade=1.0)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    ensure_account(c, 100_000.0)
    for i, d in enumerate(["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11"]):
        c.execute(
            "INSERT INTO runs (run_id, run_date, started_at, status) "
            "VALUES (?, ?, ?, 'completed')",
            (f"run-{i}", d, f"{d}T22:00:00"),
        )
    c.commit()
    yield c
    c.close()


def get_order(conn, order_id):
    return conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()


# --- account -----------------------------------------------------------------

def test_account_init_idempotent(conn):
    ensure_account(conn, 50_000.0)  # second call must not reset
    assert get_account(conn)["cash"] == 100_000.0


# --- order lifecycle ---------------------------------------------------------

def test_submit_creates_awaiting_order(conn):
    oid = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=False)
    order = get_order(conn, oid)
    assert order["status"] == "awaiting_approval"
    assert order["decided_at"] is None
    assert order["expires_at"] == "2026-06-09"  # next business day


def test_auto_approve_creates_approved_order(conn):
    oid = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    order = get_order(conn, oid)
    assert order["status"] == "approved"
    assert order["decided_at"] is not None


def test_duplicate_submit_returns_existing(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=False)
    b = submit_order(conn, "run-0", "AAPL", "buy", 99, "2026-06-08", auto_approve=False)
    assert a == b
    assert get_order(conn, a)["qty"] == 10  # original wins
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_approve_and_reject_transitions(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=False)
    b = submit_order(conn, "run-0", "MSFT", "buy", 5, "2026-06-08", auto_approve=False)
    approve_order(conn, a)
    reject_order(conn, b)
    assert get_order(conn, a)["status"] == "approved"
    assert get_order(conn, b)["status"] == "rejected"
    assert list_pending(conn) == []


def test_double_decide_raises(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=False)
    approve_order(conn, a)
    with pytest.raises(ValueError, match="not awaiting_approval"):
        approve_order(conn, a)
    with pytest.raises(ValueError, match="not awaiting_approval"):
        reject_order(conn, a)


def test_unknown_order_raises(conn):
    with pytest.raises(ValueError, match="not found"):
        approve_order(conn, "nope")


def test_expiry_only_hits_stale_awaiting(conn):
    stale = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=False)
    approved = submit_order(conn, "run-0", "MSFT", "buy", 5, "2026-06-08", auto_approve=True)
    fresh = submit_order(conn, "run-1", "NVDA", "buy", 3, "2026-06-09", auto_approve=False)

    n = expire_stale_orders(conn, "2026-06-09")
    assert n == 1
    assert get_order(conn, stale)["status"] == "expired"
    assert get_order(conn, approved)["status"] == "approved"  # approval survives
    assert get_order(conn, fresh)["status"] == "awaiting_approval"  # same-day safe


def test_fillable_orders_are_approved_and_older(conn):
    submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    submit_order(conn, "run-1", "MSFT", "buy", 5, "2026-06-09", auto_approve=True)
    due = fillable_orders(conn, "2026-06-09")
    assert [o["ticker"] for o in due] == ["AAPL"]  # MSFT's run is today, not due


# --- fills: exact math -------------------------------------------------------

def test_buy_fill_exact_math(conn):
    oid = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    result = fill_order(conn, get_order(conn, oid), date(2026, 6, 9), 100.0, FILL)

    exec_price = 100.0 * 1.0005  # 5 bps
    assert result["exec_price"] == pytest.approx(exec_price)
    assert get_account(conn)["cash"] == pytest.approx(100_000.0 - 10 * exec_price)

    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL'").fetchone()
    assert pos["qty"] == 10
    assert pos["avg_cost"] == pytest.approx(exec_price)
    assert pos["opened_at"] == "2026-06-09"
    assert get_order(conn, oid)["status"] == "filled"

    fill = conn.execute("SELECT * FROM fills").fetchone()
    assert fill["price"] == pytest.approx(exec_price)
    assert fill["filled_at"] == "2026-06-09"


def test_sell_fill_exact_math_and_realized_pnl(conn):
    buy_id = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, buy_id), date(2026, 6, 9), 100.0, FILL)
    cash_after_buy = get_account(conn)["cash"]

    sell_id = submit_order(conn, "run-1", "AAPL", "sell", 10, "2026-06-09", auto_approve=True)
    result = fill_order(conn, get_order(conn, sell_id), date(2026, 6, 10), 110.0, FILL)

    buy_px = 100.0 * 1.0005
    sell_px = 110.0 * 0.9995
    expected_realized = (sell_px - buy_px) * 10
    assert result["exec_price"] == pytest.approx(sell_px)
    assert result["realized_pnl"] == pytest.approx(expected_realized)

    account = get_account(conn)
    assert account["cash"] == pytest.approx(cash_after_buy + 10 * sell_px)
    assert account["realized_pnl"] == pytest.approx(expected_realized)
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0


def test_commission_charged_both_ways(conn):
    buy_id = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, buy_id), date(2026, 6, 9), 100.0, FILL_WITH_COMMISSION)
    buy_px = 100.0 * 1.0005
    assert get_account(conn)["cash"] == pytest.approx(100_000.0 - 10 * buy_px - 1.0)

    sell_id = submit_order(conn, "run-1", "AAPL", "sell", 10, "2026-06-09", auto_approve=True)
    result = fill_order(conn, get_order(conn, sell_id), date(2026, 6, 10), 110.0,
                        FILL_WITH_COMMISSION)
    sell_px = 110.0 * 0.9995
    assert result["realized_pnl"] == pytest.approx((sell_px - buy_px) * 10 - 1.0)


def test_partial_sell_keeps_remainder(conn):
    buy_id = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, buy_id), date(2026, 6, 9), 100.0, FILL)

    sell_id = submit_order(conn, "run-1", "AAPL", "sell", 4, "2026-06-09", auto_approve=True)
    fill_order(conn, get_order(conn, sell_id), date(2026, 6, 10), 110.0, FILL)

    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL'").fetchone()
    assert pos["qty"] == pytest.approx(6)
    assert pos["avg_cost"] == pytest.approx(100.0 * 1.0005)  # cost basis unchanged


def test_addon_buy_weighted_average_cost(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, a), date(2026, 6, 9), 100.0, FILL)
    b = submit_order(conn, "run-1", "AAPL", "buy", 10, "2026-06-09", auto_approve=True)
    fill_order(conn, get_order(conn, b), date(2026, 6, 10), 120.0, FILL)

    pos = conn.execute("SELECT * FROM positions WHERE ticker='AAPL'").fetchone()
    assert pos["qty"] == 20
    expected_avg = (10 * 100.0 * 1.0005 + 10 * 120.0 * 1.0005) / 20
    assert pos["avg_cost"] == pytest.approx(expected_avg)


def test_oversell_refused(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, a), date(2026, 6, 9), 100.0, FILL)
    s = submit_order(conn, "run-1", "AAPL", "sell", 11, "2026-06-09", auto_approve=True)
    with pytest.raises(FillError, match="cannot sell"):
        fill_order(conn, get_order(conn, s), date(2026, 6, 10), 110.0, FILL)


def test_sell_without_position_refused(conn):
    s = submit_order(conn, "run-0", "MSFT", "sell", 5, "2026-06-08", auto_approve=True)
    with pytest.raises(FillError, match="cannot sell"):
        fill_order(conn, get_order(conn, s), date(2026, 6, 9), 110.0, FILL)


def test_buy_exceeding_cash_refused(conn):
    o = submit_order(conn, "run-0", "AAPL", "buy", 2000, "2026-06-08", auto_approve=True)
    with pytest.raises(FillError, match="insufficient cash"):
        fill_order(conn, get_order(conn, o), date(2026, 6, 9), 100.0, FILL)
    assert get_account(conn)["cash"] == 100_000.0  # untouched


# --- snapshots ---------------------------------------------------------------

def test_snapshot_marks_to_market(conn):
    a = submit_order(conn, "run-0", "AAPL", "buy", 10, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, a), date(2026, 6, 9), 100.0, FILL)

    summary = snapshot_account(conn, "2026-06-09", closes={"AAPL": 105.0})
    buy_px = 100.0 * 1.0005
    cash = 100_000.0 - 10 * buy_px
    assert summary["cash"] == pytest.approx(round(cash, 2))
    assert summary["equity"] == pytest.approx(round(cash + 10 * 105.0, 2))
    assert summary["unrealized_pnl"] == pytest.approx(round((105.0 - buy_px) * 10, 2))
    assert summary["n_positions"] == 1

    # Upsert: re-snapshot same date with a new close overwrites
    snapshot_account(conn, "2026-06-09", closes={"AAPL": 90.0})
    row = conn.execute(
        "SELECT * FROM account_snapshots WHERE snapshot_date='2026-06-09'"
    ).fetchone()
    assert row["equity"] == pytest.approx(round(cash + 900.0, 6))
    assert conn.execute("SELECT COUNT(*) FROM account_snapshots").fetchone()[0] == 1


# --- the done criterion: full cycle, exact P&L -------------------------------

def test_full_buy_hold_exit_cycle_exact_pnl(conn):
    """Buy day 1, mark day 2, sell day 3 — every number verified."""
    start = 100_000.0

    buy = submit_order(conn, "run-0", "AAPL", "buy", 50, "2026-06-08", auto_approve=True)
    fill_order(conn, get_order(conn, buy), date(2026, 6, 9), 200.0, FILL)
    buy_px = 200.0 * 1.0005          # 200.10
    cash = start - 50 * buy_px       # 89,995.00

    day1 = snapshot_account(conn, "2026-06-09", closes={"AAPL": 204.0})
    assert day1["equity"] == pytest.approx(round(cash + 50 * 204.0, 2))
    assert day1["unrealized_pnl"] == pytest.approx(round((204.0 - buy_px) * 50, 2))

    sell = submit_order(conn, "run-2", "AAPL", "sell", 50, "2026-06-10", auto_approve=True)
    fill_order(conn, get_order(conn, sell), date(2026, 6, 11), 210.0, FILL)
    sell_px = 210.0 * 0.9995         # 209.895
    realized = (sell_px - buy_px) * 50

    final = snapshot_account(conn, "2026-06-11", closes={})
    assert final["n_positions"] == 0
    assert final["cash"] == pytest.approx(round(cash + 50 * sell_px, 2))
    assert final["equity"] == pytest.approx(final["cash"])  # all cash again
    assert final["realized_pnl"] == pytest.approx(round(realized, 2))
    assert final["unrealized_pnl"] == 0.0
    # Round-trip sanity: equity = start + realized
    assert final["equity"] == pytest.approx(round(start + realized, 2))
