"""Paper broker: fills at next open with slippage/commission, exact accounting.

Fill model (PLAN.md E3):
    buy  exec price = open x (1 + slippage_bps/10000); cash -= qty x px + commission
    sell exec price = open x (1 - slippage_bps/10000); cash += qty x px - commission
    realized pnl on sell = (exec px - avg cost) x qty - commission

Positions use weighted-average cost. The same fill conventions will drive the
backtester (phase 12) so paper and backtest results stay comparable.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date

from trading_platform.core.config import FillModel
from trading_platform.core.models import utcnow


def ensure_account(conn: sqlite3.Connection, starting_cash: float) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO account (id, cash, realized_pnl, updated_at) "
        "VALUES (1, ?, 0, ?)",
        (starting_cash, utcnow().isoformat()),
    )
    conn.commit()


def get_account(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM account WHERE id = 1").fetchone()


class FillError(RuntimeError):
    """A fill that must not proceed (e.g. selling an absent position)."""


def fill_order(
    conn: sqlite3.Connection,
    order: sqlite3.Row,
    fill_date: date,
    open_price: float,
    fill_model: FillModel,
) -> dict:
    """Execute one approved order at the given open price (local simulation)."""
    slip = fill_model.slippage_bps / 10_000.0
    exec_price = (open_price * (1 + slip) if order["side"] == "buy"
                  else open_price * (1 - slip))
    return apply_fill(conn, order, fill_date, exec_price,
                      fill_model.commission_per_trade,
                      raw_price=open_price)


def apply_fill(
    conn: sqlite3.Connection,
    order: sqlite3.Row,
    fill_date: date,
    exec_price: float,
    commission: float = 0.0,
    raw_price: float | None = None,
    enforce_cash: bool = True,
) -> dict:
    """Record a fill at an explicit execution price — the single accounting
    path for both simulated fills and real broker (Alpaca paper) fills.

    enforce_cash=False is for external fills: the broker already executed, so
    the local ledger must record reality; any resulting negative cash is
    surfaced by reconciliation, not hidden by a refusal."""
    qty = float(order["qty"])
    ticker = order["ticker"]
    now = utcnow().isoformat()
    raw_price = raw_price if raw_price is not None else exec_price

    if order["side"] == "buy":
        cost = qty * exec_price + commission
        account = get_account(conn)
        if enforce_cash and cost > account["cash"] + 1e-9:
            raise FillError(
                f"insufficient cash: need {cost:.2f}, have {account['cash']:.2f}"
            )
        existing = conn.execute(
            "SELECT qty, avg_cost FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()
        if existing:
            new_qty = existing["qty"] + qty
            new_avg = (existing["qty"] * existing["avg_cost"] + qty * exec_price) / new_qty
            conn.execute(
                "UPDATE positions SET qty = ?, avg_cost = ?, updated_at = ? WHERE ticker = ?",
                (new_qty, new_avg, now, ticker),
            )
        else:
            conn.execute(
                "INSERT INTO positions (ticker, qty, avg_cost, opened_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, qty, exec_price, fill_date.isoformat(), now),
            )
        conn.execute(
            "UPDATE account SET cash = cash - ?, updated_at = ? WHERE id = 1",
            (cost, now),
        )
        realized = 0.0
    else:  # sell
        position = conn.execute(
            "SELECT qty, avg_cost FROM positions WHERE ticker = ?", (ticker,)
        ).fetchone()
        if position is None or position["qty"] + 1e-9 < qty:
            raise FillError(
                f"cannot sell {qty} {ticker}: held "
                f"{position['qty'] if position else 0}"
            )
        proceeds = qty * exec_price - commission
        realized = (exec_price - position["avg_cost"]) * qty - commission
        remaining = position["qty"] - qty
        if remaining <= 1e-9:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        else:
            conn.execute(
                "UPDATE positions SET qty = ?, updated_at = ? WHERE ticker = ?",
                (remaining, now, ticker),
            )
        conn.execute(
            "UPDATE account SET cash = cash + ?, realized_pnl = realized_pnl + ?, "
            "updated_at = ? WHERE id = 1",
            (proceeds, realized, now),
        )

    fill_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO fills (fill_id, order_id, ticker, side, qty, price, slippage, "
        "commission, filled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fill_id, order["order_id"], ticker, order["side"], qty,
            round(exec_price, 6), round(abs(exec_price - raw_price) * qty, 6),
            commission, fill_date.isoformat(),
        ),
    )
    conn.execute(
        "UPDATE orders SET status = 'filled' WHERE order_id = ?", (order["order_id"],)
    )
    conn.commit()
    return {
        "fill_id": fill_id, "ticker": ticker, "side": order["side"], "qty": qty,
        "exec_price": round(exec_price, 6), "realized_pnl": round(realized, 6),
    }


def snapshot_account(
    conn: sqlite3.Connection, snapshot_date: str, closes: dict[str, float]
) -> dict:
    """Mark-to-market snapshot. Positions without a close fall back to avg cost."""
    account = get_account(conn)
    positions = conn.execute("SELECT ticker, qty, avg_cost FROM positions").fetchall()
    market_value = 0.0
    unrealized = 0.0
    for p in positions:
        price = closes.get(p["ticker"], p["avg_cost"])
        market_value += p["qty"] * price
        unrealized += (price - p["avg_cost"]) * p["qty"]

    equity = account["cash"] + market_value
    conn.execute(
        """
        INSERT INTO account_snapshots
            (snapshot_date, cash, equity, unrealized_pnl, realized_pnl, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (snapshot_date) DO UPDATE SET
            cash = excluded.cash, equity = excluded.equity,
            unrealized_pnl = excluded.unrealized_pnl,
            realized_pnl = excluded.realized_pnl, created_at = excluded.created_at
        """,
        (
            snapshot_date, round(account["cash"], 6), round(equity, 6),
            round(unrealized, 6), round(account["realized_pnl"], 6),
            utcnow().isoformat(),
        ),
    )
    conn.commit()
    return {
        "cash": round(account["cash"], 2), "equity": round(equity, 2),
        "unrealized_pnl": round(unrealized, 2),
        "realized_pnl": round(account["realized_pnl"], 2),
        "n_positions": len(positions),
    }
