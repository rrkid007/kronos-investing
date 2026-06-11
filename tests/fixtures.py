"""Synthetic OHLCV fixtures — tests never touch the network."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_platform.data.market_data import MarketDataService, RefreshResult, TickerData
from trading_platform.data.quality import validate_ohlcv


def make_ohlcv(
    n_rows: int = 300,
    end: date = date(2026, 6, 11),
    start_price: float = 100.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Clean synthetic daily bars ending at `end`, indexed by business day."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n_rows)
    rets = rng.normal(0.0005, 0.012, n_rows)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.004, n_rows))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_rows)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_rows)))
    volume = rng.integers(1_000_000, 50_000_000, n_rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=idx,
    )


class FakeMarketDataService(MarketDataService):
    """Serves canned frames; optionally fails specific tickers."""

    def __init__(self, conn=None, frames: dict[str, pd.DataFrame] | None = None,
                 default_frame: pd.DataFrame | None = None,
                 fetch_errors: dict[str, str] | None = None):
        self.conn = conn
        self.frames = frames or {}
        self.default_frame = default_frame
        self.fetch_errors = fetch_errors or {}

    def refresh_and_validate(self, ticker: str, as_of: date | None = None) -> TickerData:
        if ticker in self.fetch_errors:
            return TickerData(
                ticker=ticker,
                refresh=RefreshResult(ticker=ticker, error=self.fetch_errors[ticker]),
            )
        df = self.frames.get(ticker, self.default_frame)
        if df is None:
            return TickerData(
                ticker=ticker, refresh=RefreshResult(ticker=ticker, error="no fixture")
            )
        refresh = RefreshResult(
            ticker=ticker, rows_upserted=len(df), last_date=df.index[-1].date()
        )
        quality = validate_ohlcv(df, ticker, as_of=df.index[-1].date())
        return TickerData(ticker=ticker, refresh=refresh, quality=quality, df=df)
