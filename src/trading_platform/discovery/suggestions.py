"""Approve or dismiss watchlist discovery suggestions.

Mirrors the ``scripts/discover_stocks.py --add/--dismiss`` flow, but writes the
watchlist through ``core.config_writer`` (validated, comment-preserving) instead
of a raw text append, so the dashboard and the CLI share the same guarantees.

Approving a suggestion:
  1. rejects tickers already on the watchlist,
  2. enforces ``discovery.max_watchlist_size`` (with a weakest-incumbent hint),
  3. resolves the sector from the suggestion row (falling back to the universe),
  4. writes the validated watchlist.yaml entry, and
  5. marks the suggestion ``added`` in the database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trading_platform.core import config_writer
from trading_platform.core.config import AppConfig


def _resolve_sector(conn: sqlite3.Connection, ticker: str) -> str | None:
    row = conn.execute(
        "SELECT sector FROM watchlist_suggestions WHERE ticker = ? "
        "AND sector IS NOT NULL ORDER BY created_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if row and row["sector"]:
        return row["sector"]
    row = conn.execute(
        "SELECT sector FROM universe WHERE ticker = ? LIMIT 1", (ticker,)
    ).fetchone()
    return row["sector"] if row and row["sector"] else None


def approve_suggestion(
    conn: sqlite3.Connection,
    config: AppConfig,
    config_dir: Path | str,
    ticker: str,
) -> tuple[AppConfig, str]:
    """Add a suggested ticker to the watchlist. Returns (new_config, sector).

    Raises ValueError with a human-readable reason if it can't be added.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    if ticker in config.watchlist.symbols:
        raise ValueError(f"{ticker} is already on the watchlist")

    max_size = config.settings.discovery.max_watchlist_size
    if len(config.watchlist.symbols) >= max_size:
        # Lazy import: the screener pulls the heavy data stack; only needed here.
        from trading_platform.discovery.screener import weakest_incumbent

        weakest = weakest_incumbent(conn, config)
        hint = f" — weakest incumbent: {weakest['ticker']}" if weakest else ""
        raise ValueError(
            f"watchlist at capacity ({max_size}); remove a member first{hint}"
        )

    sector = _resolve_sector(conn, ticker)
    if not sector:
        raise ValueError(
            f"no sector on record for {ticker}; add it manually in the Watchlist panel"
        )

    new_config = config_writer.add_ticker(config_dir, ticker, sector)
    conn.execute(
        "UPDATE watchlist_suggestions SET status = 'added' WHERE ticker = ?", (ticker,)
    )
    conn.commit()
    return new_config, sector


def dismiss_suggestion(conn: sqlite3.Connection, ticker: str) -> str:
    """Mark a suggestion dismissed. Returns the ticker. Idempotent."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    conn.execute(
        "UPDATE watchlist_suggestions SET status = 'dismissed' WHERE ticker = ?",
        (ticker,),
    )
    conn.commit()
    return ticker
