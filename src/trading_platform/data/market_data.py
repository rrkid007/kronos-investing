"""Market data ingestion: yfinance -> SQLite price_cache.

Refresh strategy: re-download the full lookback window every time and upsert.
Splits and dividends retroactively change *adjusted* prices for the entire
history, so append-only incremental updates silently leave stale adj_close
values behind. At watchlist scale (tens of tickers x ~2.5 years of daily
bars) a full-window refresh is sub-second per ticker and always correct.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from trading_platform.data.quality import OHLCV_COLUMNS, QualityReport, validate_ohlcv

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 900  # calendar days; ~600 trading bars for the 200DMA + history


class RefreshResult(BaseModel):
    ticker: str
    rows_upserted: int = 0
    last_date: date | None = None
    error: str | None = None


class TickerData(BaseModel):
    """Outcome of refresh + quality gate for one ticker."""

    model_config = {"arbitrary_types_allowed": True}

    ticker: str
    refresh: RefreshResult
    quality: QualityReport | None = None
    df: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return self.refresh.error is None and self.quality is not None and self.quality.passed

    def detail(self) -> str:
        if self.refresh.error:
            return f"fetch failed: {self.refresh.error}"
        if self.quality and not self.quality.passed:
            return f"quality gate failed: {self.quality.summary()}"
        return f"ok ({self.refresh.rows_upserted} rows upserted, last {self.refresh.last_date})"


def _download(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch raw OHLCV from yfinance. Isolated so tests can patch it."""
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,  # keep raw close AND adjusted close
        progress=False,
        threads=False,
    )
    if df is None:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance >= 0.2.40 single-ticker quirk
        df.columns = df.columns.droplevel(1)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    return df[[c for c in OHLCV_COLUMNS if c in df.columns]]


class MarketDataService:
    def __init__(self, conn: sqlite3.Connection, lookback_days: int = DEFAULT_LOOKBACK_DAYS):
        self.conn = conn
        self.lookback_days = lookback_days

    def refresh(self, ticker: str, as_of: date | None = None) -> RefreshResult:
        """Download the lookback window and upsert into price_cache."""
        as_of = as_of or date.today()
        start = as_of - timedelta(days=self.lookback_days)
        try:
            df = _download(ticker, start, as_of + timedelta(days=1))
        except Exception as exc:  # network/API failures must never crash the run
            logger.warning("download failed for %s: %s", ticker, exc)
            return RefreshResult(ticker=ticker, error=str(exc))

        if df.empty:
            return RefreshResult(ticker=ticker, error="no data returned")

        df = _drop_partial_last_bar(df, as_of)
        if df.empty:
            return RefreshResult(ticker=ticker, error="only a partial bar returned")

        rows = [
            (
                ticker,
                idx.date().isoformat(),
                _f(row.get("open")),
                _f(row.get("high")),
                _f(row.get("low")),
                _f(row.get("close")),
                _f(row.get("adj_close")),
                int(row["volume"]) if pd.notna(row.get("volume")) else None,
            )
            for idx, row in df.iterrows()
        ]
        self.conn.executemany(
            """
            INSERT INTO price_cache (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low,
                close = excluded.close, adj_close = excluded.adj_close,
                volume = excluded.volume
            """,
            rows,
        )
        # Purge anything newer than the last good bar (e.g. a partial bar
        # cached by an earlier intraday refresh) — upserts never delete.
        last_date = df.index[-1].date()
        self.conn.execute(
            "DELETE FROM price_cache WHERE ticker = ? AND date > ?",
            (ticker, last_date.isoformat()),
        )
        self.conn.commit()
        return RefreshResult(ticker=ticker, rows_upserted=len(rows), last_date=last_date)

    def load(self, ticker: str) -> pd.DataFrame:
        """Load a ticker's cached history as a DatetimeIndex OHLCV frame."""
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM price_cache WHERE ticker = ? ORDER BY date",
            self.conn,
            params=(ticker,),
            parse_dates=["date"],
            index_col="date",
        )
        return df

    def refresh_and_validate(self, ticker: str, as_of: date | None = None) -> TickerData:
        """Refresh, reload from cache, and run the quality gate."""
        refresh = self.refresh(ticker, as_of=as_of)
        if refresh.error:
            return TickerData(ticker=ticker, refresh=refresh)
        df = self.load(ticker)
        quality = validate_ohlcv(df, ticker, as_of=as_of)
        return TickerData(ticker=ticker, refresh=refresh, quality=quality, df=df)


def _drop_partial_last_bar(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Drop today's in-progress bar when it is internally inconsistent.

    During market hours Yahoo's provisional daily bar can carry the prior
    session's open alongside the current session's high/low/close (observed
    live: open > high). Finalized bars always satisfy low <= open/close <=
    high, so an as_of-dated bar that violates it is partial — daily analysis
    must only ever see completed bars. After the close the bar is final,
    passes, and the next full-window refresh upserts it normally.
    """
    last = df.iloc[-1]
    if df.index[-1].date() != as_of:
        return df
    consistent = (
        last["low"] <= min(last["open"], last["close"]) + 1e-9
        and last["high"] >= max(last["open"], last["close"]) - 1e-9
    )
    if consistent:
        return df
    logger.info("dropping partial bar %s for %s", df.index[-1].date(), df.columns.name or "")
    return df.iloc[:-1]


def _f(value) -> float | None:
    return float(value) if pd.notna(value) else None
