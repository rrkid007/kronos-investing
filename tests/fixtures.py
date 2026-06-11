"""Synthetic OHLCV fixtures — tests never touch the network."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_platform.data.fundamentals import FundamentalsSnapshot
from trading_platform.data.market_data import MarketDataService, RefreshResult, TickerData
from trading_platform.data.quality import validate_ohlcv


def make_snapshot(ticker: str = "TEST", **overrides) -> FundamentalsSnapshot:
    """Full-coverage snapshot of a strong, reasonably-priced business."""
    from datetime import datetime, timezone

    base = dict(
        ticker=ticker,
        fetched_at=datetime(2026, 6, 11, tzinfo=timezone.utc),
        data_as_of=date(2026, 3, 31),
        revenue_growth=0.12,
        earnings_growth=0.15,
        revenue_cagr_3y=0.10,
        gross_margin=0.45,
        operating_margin=0.28,
        profit_margin=0.22,
        return_on_equity=0.30,
        debt_to_equity=0.6,
        current_ratio=1.5,
        net_cash_to_market_cap=0.02,
        fcf_margin=0.20,
        ocf_margin=0.25,
        trailing_pe=24.0,
        forward_pe=21.0,
        ev_to_ebitda=16.0,
        price_to_fcf=28.0,
    )
    base.update(overrides)
    return FundamentalsSnapshot(**base)


def make_ohlcv(
    n_rows: int = 300,
    end: date = date(2026, 6, 11),
    start_price: float = 100.0,
    seed: int = 7,
    drift: float = 0.0005,
    daily_vol: float = 0.012,
) -> pd.DataFrame:
    """Clean synthetic daily bars ending at `end`, indexed by business day."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n_rows)
    rets = rng.normal(drift, daily_vol, n_rows)
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


def make_ohlcv_piecewise(
    segments: list[tuple[int, float]],
    end: date = date(2026, 6, 11),
    start_price: float = 100.0,
    seed: int = 7,
    daily_vol: float = 0.004,
) -> pd.DataFrame:
    """Bars with piecewise drift, e.g. [(300, 0.002), (30, -0.012)] = rally then crash."""
    rng = np.random.default_rng(seed)
    rets = np.concatenate([rng.normal(d, daily_vol, n) for n, d in segments])
    n_rows = len(rets)
    idx = pd.bdate_range(end=end, periods=n_rows)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.002, n_rows))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_rows)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_rows)))
    volume = rng.integers(1_000_000, 50_000_000, n_rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": volume},
        index=idx,
    )


class FakeLLM:
    """Returns a canned response object; optionally raises instead."""

    def __init__(self, response=None, fail: Exception | None = None):
        self.response = response
        self.fail = fail
        self.calls = 0

    def generate(self, prompt, response_model, system=None, retries=2):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return self.response


def make_news_items(ticker: str = "TEST", n: int = 5):
    from datetime import datetime, timedelta, timezone

    from trading_platform.data.news import NewsItem, _hash_headline

    base = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
    return [
        NewsItem(
            ticker=ticker,
            source="testwire",
            headline=f"{ticker} headline number {i}",
            summary=f"summary {i}",
            url=f"https://example.com/{i}",
            published_at=base - timedelta(hours=i),
            content_hash=_hash_headline(f"{ticker} headline number {i}"),
        )
        for i in range(n)
    ]


class FakeForecaster:
    """Deterministic Kronos stand-in: linear paths to a configurable return."""

    def __init__(self, final_return: float = 0.03, spread: float = 0.01,
                 fail: Exception | None = None):
        self.final_return = final_return
        self.spread = spread
        self.fail = fail

    def predict_paths(self, df, horizon: int, sample_count: int) -> np.ndarray:
        if self.fail is not None:
            raise self.fail
        last = float(df["close"].iloc[-1])
        final_returns = np.linspace(
            self.final_return - self.spread, self.final_return + self.spread, sample_count
        )
        return np.array([
            np.linspace(last, last * (1.0 + r), horizon) for r in final_returns
        ])


class FakeMarketDataService(MarketDataService):
    """Serves canned frames; optionally fails specific tickers."""

    def __init__(self, conn=None, frames: dict[str, pd.DataFrame] | None = None,
                 default_frame: pd.DataFrame | None = None,
                 fetch_errors: dict[str, str] | None = None):
        self.conn = conn
        self.frames = frames or {}
        self.default_frame = default_frame
        self.fetch_errors = fetch_errors or {}

    def load(self, ticker: str) -> pd.DataFrame:
        df = self.frames.get(ticker, self.default_frame)
        return df if df is not None else pd.DataFrame()

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
