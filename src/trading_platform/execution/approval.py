"""Order approval with broker routing — shared by the CLI and the dashboard.

Local broker: approval just flips status; the next run fills internally.
Alpaca paper: approval also submits a market-on-open order, so a trade
approved on evening T fills at T+1's open — the same convention as local.
A failed submission leaves the order approved with the error noted; the next
run's sync pass retries (client_order_id makes retries idempotent at Alpaca).
"""

from __future__ import annotations

import logging
import sqlite3

from trading_platform.core.config import AppConfig
from trading_platform.execution.alpaca import AlpacaError, AlpacaPaperBroker
from trading_platform.execution.orders import approve_order

logger = logging.getLogger(__name__)


def approve_and_submit(conn: sqlite3.Connection, config: AppConfig, order_id: str) -> str:
    """Approve an order and route it to the configured broker. Returns a
    human-readable status message. Raises ValueError for invalid approvals."""
    approve_order(conn, order_id)
    if config.settings.execution.broker != "alpaca_paper":
        return f"approved {order_id} (local fill at next open)"

    order = conn.execute(
        "SELECT * FROM orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    return submit_to_alpaca(conn, config, order)


def submit_to_alpaca(conn: sqlite3.Connection, config: AppConfig, order) -> str:
    try:
        broker = AlpacaPaperBroker.from_env(config.settings.execution)
        resp = broker.submit_market_on_open(
            order["ticker"], order["side"], order["qty"],
            client_order_id=order["order_id"],
        )
    except AlpacaError as exc:
        logger.warning("alpaca submission failed for %s: %s", order["order_id"], exc)
        conn.execute(
            "UPDATE orders SET notes = ? WHERE order_id = ?",
            (f"alpaca submission failed, will retry next run: {exc}",
             order["order_id"]),
        )
        conn.commit()
        return f"approved {order['order_id']} but alpaca submission failed: {exc}"

    conn.execute(
        "UPDATE orders SET broker = 'alpaca_paper', broker_order_id = ? "
        "WHERE order_id = ?",
        (resp["id"], order["order_id"]),
    )
    conn.commit()
    return (f"approved {order['order_id']} -> alpaca paper "
            f"(market-on-open, broker id {resp['id'][:8]}…)")
