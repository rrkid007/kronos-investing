"""Candidate universe: S&P 500 constituents with GICS sectors.

Source: Wikipedia's constituents table (no API key, stable schema, includes
sectors). Cached in the `universe` table and refreshed monthly — index
membership churns slowly. Symbols are normalized to the platform's dash
convention (Wikipedia's BRK.B -> BRK-B, matching yfinance).
"""

from __future__ import annotations

import io
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "trading-platform-research/0.1 (contact: info@peaklogic.ai)"


def _fetch_constituents() -> list[dict]:
    """Download and parse the constituents table. Isolated for tests."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]  # Symbol | Security | GICS Sector | ...
    out = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip().replace(".", "-")
        sector = str(row.get("GICS Sector", "")).strip()
        if symbol and sector:
            out.append({
                "ticker": symbol,
                "name": str(row.get("Security", "")).strip(),
                "sector": sector,
            })
    if len(out) < 400:  # sanity: a broken parse must not poison the cache
        raise ValueError(f"unexpected constituents count: {len(out)}")
    return out


def refresh_universe(
    conn: sqlite3.Connection, max_age_days: int = 30, force: bool = False
) -> int:
    """Refresh if stale; returns the number of members in the table."""
    newest = conn.execute("SELECT MAX(refreshed_at) FROM universe").fetchone()[0]
    if newest and not force:
        age = datetime.now(tz=timezone.utc) - datetime.fromisoformat(newest)
        if age < timedelta(days=max_age_days):
            return conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]

    members = _fetch_constituents()
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO universe (ticker, name, sector, refreshed_at)
        VALUES (:ticker, :name, :sector, :refreshed_at)
        ON CONFLICT (ticker) DO UPDATE SET
            name = excluded.name, sector = excluded.sector,
            refreshed_at = excluded.refreshed_at
        """,
        [{**m, "refreshed_at": now} for m in members],
    )
    # Drop departed members so suggestions never point at ex-constituents.
    # (Explicit ticker set, not a timestamp comparison — Windows clock
    # granularity can give two refreshes the same isoformat timestamp.)
    placeholders = ",".join("?" * len(members))
    conn.execute(
        f"DELETE FROM universe WHERE ticker NOT IN ({placeholders})",
        [m["ticker"] for m in members],
    )
    conn.commit()
    logger.info("universe refreshed: %d members", len(members))
    return len(members)


def load_universe(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM universe ORDER BY ticker"
    ).fetchall()


def mark_screened(conn: sqlite3.Connection, tickers: list[str]) -> None:
    now = datetime.now(tz=timezone.utc).isoformat()
    conn.executemany(
        "UPDATE universe SET last_screened_at = ? WHERE ticker = ?",
        [(now, t) for t in tickers],
    )
    conn.commit()
