"""Order lifecycle: awaiting_approval -> approved/rejected/expired -> filled.

Semantics (PLAN.md A3/E3):
- A run on date T creates orders from T's risk-cleared decisions.
- The approval window lasts until the next run processes fills: an order
  still awaiting approval when a later run starts is expired — unapproved
  trades never fill late.
- Approved orders fill at the open of the next run's date (T+1 open when
  runs are daily; the next run's open if a day was missed — documented,
  conservative).
- auto_approve (config require_human_approval: false) creates orders
  pre-approved.

Idempotency: UNIQUE(run_id, ticker, side) — re-running a date never
duplicates an order; the original (and its approval state) wins.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np

from trading_platform.core.models import utcnow


def _next_business_day(run_date: str) -> str:
    return str(np.busday_offset(run_date, 1, roll="forward"))


def submit_order(
    conn: sqlite3.Connection,
    run_id: str,
    ticker: str,
    side: str,
    qty: float,
    run_date: str,
    auto_approve: bool,
    context: dict | None = None,
) -> str | None:
    """Create an order; returns its id (existing id if already submitted)."""
    order_id = uuid.uuid4().hex[:12]
    status = "approved" if auto_approve else "awaiting_approval"
    now = utcnow().isoformat()
    cursor = conn.execute(
        """
        INSERT INTO orders
            (order_id, run_id, ticker, side, qty, status, created_at, decided_at,
             expires_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, ticker, side) DO NOTHING
        """,
        (
            order_id, run_id, ticker, side, qty, status, now,
            now if auto_approve else None,
            _next_business_day(run_date),
            json.dumps(context or {}),
        ),
    )
    conn.commit()
    if cursor.rowcount == 0:  # already submitted by an earlier invocation
        row = conn.execute(
            "SELECT order_id FROM orders WHERE run_id = ? AND ticker = ? AND side = ?",
            (run_id, ticker, side),
        ).fetchone()
        return row["order_id"]
    return order_id


def list_pending(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT o.*, r.run_date FROM orders o
        JOIN runs r ON o.run_id = r.run_id
        WHERE o.status = 'awaiting_approval'
        ORDER BY r.run_date, o.ticker
        """
    ).fetchall()


def _transition(conn: sqlite3.Connection, order_id: str, new_status: str) -> None:
    row = conn.execute(
        "SELECT status FROM orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"order {order_id} not found")
    if row["status"] != "awaiting_approval":
        raise ValueError(
            f"order {order_id} is '{row['status']}', not awaiting_approval"
        )
    conn.execute(
        "UPDATE orders SET status = ?, decided_at = ? WHERE order_id = ?",
        (new_status, utcnow().isoformat(), order_id),
    )
    conn.commit()


def approve_order(conn: sqlite3.Connection, order_id: str) -> None:
    _transition(conn, order_id, "approved")


def reject_order(conn: sqlite3.Connection, order_id: str) -> None:
    _transition(conn, order_id, "rejected")


def expire_stale_orders(conn: sqlite3.Connection, current_run_date: str) -> int:
    """Expire orders still awaiting approval from runs before current_run_date."""
    cursor = conn.execute(
        """
        UPDATE orders SET status = 'expired', decided_at = ?
        WHERE status = 'awaiting_approval'
          AND run_id IN (SELECT run_id FROM runs WHERE run_date < ?)
        """,
        (utcnow().isoformat(), current_run_date),
    )
    conn.commit()
    return cursor.rowcount


def fillable_orders(conn: sqlite3.Connection, current_run_date: str) -> list[sqlite3.Row]:
    """Approved orders from earlier runs — due to fill at today's open."""
    return conn.execute(
        """
        SELECT o.*, r.run_date FROM orders o
        JOIN runs r ON o.run_id = r.run_id
        WHERE o.status = 'approved' AND r.run_date < ?
        ORDER BY r.run_date, o.ticker
        """,
        (current_run_date,),
    ).fetchall()
