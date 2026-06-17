"""Macro indicators from FRED's keyless CSV endpoints.

Three stress dials: the 10Y-2Y treasury spread (T10Y2Y), the VIX (VIXCLS),
and high-yield credit spreads (BAMLH0A0HYM2). Values cache in
macro_indicators, so a failed fetch falls back to the most recent cached
value within the staleness window — macro context degrades, it never crashes
a run.
"""

from __future__ import annotations

import io
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SERIES = {
    "yield_curve": "T10Y2Y",
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class MacroSnapshot(BaseModel):
    yield_curve: float | None = None
    vix: float | None = None
    hy_oas: float | None = None
    as_of: dict[str, str] = {}  # indicator -> observation date

    @property
    def n_available(self) -> int:
        return sum(v is not None for v in (self.yield_curve, self.vix, self.hy_oas))


def _fetch_csv(series_id: str, start: date | None = None) -> pd.DataFrame:
    """Download one series. Isolated for tests. FRED encodes missing as '.'

    `cosd` bounds the window — without it FRED streams the series' full
    multi-decade history, which is slow enough to time out.
    """
    params = {"id": series_id}
    if start is not None:
        params["cosd"] = start.isoformat()
    resp = requests.get(
        FRED_CSV_URL, params=params, timeout=30,
        headers={"User-Agent": "trading-platform-research/0.1"},
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def refresh_macro(
    conn: sqlite3.Connection, as_of: date | None = None, max_staleness_days: int = 7
) -> MacroSnapshot:
    """Fetch all series (cache on success), return the latest fresh values."""
    as_of = as_of or date.today()
    now = datetime.now(tz=timezone.utc).isoformat()
    values: dict[str, float | None] = {}
    dates: dict[str, str] = {}

    for name, series_id in SERIES.items():
        try:
            df = _fetch_csv(series_id, start=as_of - timedelta(days=60)).tail(30)
            conn.executemany(
                "INSERT INTO macro_indicators (series, date, value, fetched_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (series, date) DO UPDATE SET "
                "value = excluded.value, fetched_at = excluded.fetched_at",
                [(series_id, str(r["date"])[:10], float(r["value"]), now)
                 for _, r in df.iterrows()],
            )
            conn.commit()
        except Exception as exc:
            logger.warning("macro fetch failed for %s (%s): %s — using cache",
                           name, series_id, exc)

        cached = conn.execute(
            "SELECT date, value FROM macro_indicators "
            "WHERE series = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (series_id, as_of.isoformat()),
        ).fetchone()
        if cached and (as_of - date.fromisoformat(cached["date"])).days <= max_staleness_days:
            values[name] = cached["value"]
            dates[name] = cached["date"]
        else:
            values[name] = None

    return MacroSnapshot(**values, as_of=dates)
