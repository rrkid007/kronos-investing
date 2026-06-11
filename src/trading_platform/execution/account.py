"""Account and position state readers.

The Paper Trading Engine (phase 8) owns writes to these tables; this module
provides the read view that the decision/portfolio layer needs. Until the
account row exists, the account is all cash at the configured starting
balance.
"""

from __future__ import annotations

import sqlite3
from datetime import date

from trading_platform.core.config import AppConfig
from trading_platform.core.models import Position


def load_positions(conn: sqlite3.Connection) -> list[Position]:
    rows = conn.execute(
        "SELECT ticker, qty, avg_cost, opened_at FROM positions WHERE qty > 0"
    ).fetchall()
    return [
        Position(
            ticker=r["ticker"],
            qty=r["qty"],
            avg_cost=r["avg_cost"],
            opened_at=date.fromisoformat(r["opened_at"][:10]),
        )
        for r in rows
    ]


def load_cash(conn: sqlite3.Connection, config: AppConfig) -> float:
    row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
    return row["cash"] if row else config.risk.paper_account.starting_cash
