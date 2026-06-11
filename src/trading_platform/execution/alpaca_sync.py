"""Alpaca paper sync — runs at the start of each daily run when the broker
is alpaca_paper, replacing the local simulated-fill pass.

Three jobs:
1. Retry submissions for approved orders that never reached Alpaca
   (client_order_id keeps retries idempotent at the broker).
2. Poll submitted orders: record real fills through the SAME accounting path
   as simulated fills (apply_fill, at Alpaca's actual average price, cash
   check disabled — the broker already executed, the ledger records reality).
   Terminal broker states (canceled/expired/rejected) cancel locally.
3. Reconcile: compare local positions/cash against Alpaca's view and report
   drift. Drift is surfaced, never auto-corrected — a disagreement between
   ledgers is a bug to investigate, not paper over.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime

from trading_platform.core.config import AppConfig
from trading_platform.execution.alpaca import (
    AlpacaError,
    AlpacaPaperBroker,
    alpaca_symbol,
)
from trading_platform.execution.approval import submit_to_alpaca
from trading_platform.execution.paper_broker import apply_fill, get_account

logger = logging.getLogger(__name__)

TERMINAL_BROKER_STATES = {"canceled", "expired", "rejected", "suspended"}


def sync_alpaca(conn: sqlite3.Connection, config: AppConfig, as_of: date) -> dict:
    """Returns {'filled': n, 'cancelled': n, 'resubmitted': n, 'drift': [...]}"""
    result = {"filled": 0, "cancelled": 0, "resubmitted": 0, "drift": []}
    try:
        broker = AlpacaPaperBroker.from_env(config.settings.execution)
    except AlpacaError as exc:
        logger.warning("alpaca sync skipped: %s", exc)
        result["drift"].append(f"sync skipped: {exc}")
        return result

    # 1. retry never-submitted approved orders from earlier runs
    unsubmitted = conn.execute(
        """SELECT o.* FROM orders o JOIN runs r ON o.run_id = r.run_id
           WHERE o.status = 'approved' AND o.broker_order_id IS NULL
             AND r.run_date < ?""",
        (as_of.isoformat(),),
    ).fetchall()
    for order in unsubmitted:
        message = submit_to_alpaca(conn, config, order)
        if "broker id" in message:
            result["resubmitted"] += 1

    # 2. poll submitted orders for fills / terminal states
    submitted = conn.execute(
        "SELECT * FROM orders WHERE status = 'approved' AND broker_order_id IS NOT NULL"
    ).fetchall()
    for order in submitted:
        try:
            info = broker.get_order(order["broker_order_id"])
        except AlpacaError as exc:
            logger.warning("poll failed for %s: %s", order["order_id"], exc)
            continue
        status = info.get("status")
        if status == "filled":
            fill_date = _parse_date(info.get("filled_at")) or as_of
            price = float(info["filled_avg_price"])
            apply_fill(conn, order, fill_date, price, commission=0.0,
                       enforce_cash=False)
            logger.info("alpaca fill: %s %s x%s @ %s",
                        order["side"], order["ticker"], order["qty"], price)
            result["filled"] += 1
        elif status in TERMINAL_BROKER_STATES:
            conn.execute(
                "UPDATE orders SET status = 'cancelled', notes = ? WHERE order_id = ?",
                (f"alpaca terminal state: {status}", order["order_id"]),
            )
            conn.commit()
            result["cancelled"] += 1

    # 3. reconcile positions and cash
    result["drift"] = _reconcile(conn, broker)
    for line in result["drift"]:
        logger.warning("alpaca drift: %s", line)
    return result


def _reconcile(conn: sqlite3.Connection, broker: AlpacaPaperBroker) -> list[str]:
    drift: list[str] = []
    try:
        remote_positions = {p["symbol"]: float(p["qty"]) for p in broker.get_positions()}
        remote_cash = float(broker.get_account()["cash"])
    except AlpacaError as exc:
        return [f"reconciliation unavailable: {exc}"]

    local = {
        alpaca_symbol(r["ticker"]): float(r["qty"])
        for r in conn.execute("SELECT ticker, qty FROM positions").fetchall()
    }
    for symbol in sorted(set(local) | set(remote_positions)):
        lq, rq = local.get(symbol, 0.0), remote_positions.get(symbol, 0.0)
        if abs(lq - rq) > 1e-6:
            drift.append(f"{symbol}: local qty {lq:g} vs alpaca {rq:g}")

    local_cash = get_account(conn)["cash"]
    if abs(local_cash - remote_cash) > 1.0:  # ignore sub-dollar rounding
        drift.append(f"cash: local {local_cash:,.2f} vs alpaca {remote_cash:,.2f}")
    return drift


def _parse_date(timestamp: str | None) -> date | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
    except ValueError:
        return None
