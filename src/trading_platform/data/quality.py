"""Data-quality gate for OHLCV history.

Every ticker's data must pass this gate before any agent scores it. A ticker
that fails is skipped and flagged for the run report — scoring on bad data is
worse than not scoring at all.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class QualityCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class QualityReport(BaseModel):
    ticker: str
    passed: bool
    checks: list[QualityCheck]

    @property
    def failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.passed:
            return "all checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in self.failures)


def validate_ohlcv(
    df: pd.DataFrame,
    ticker: str,
    *,
    min_rows: int = 250,
    max_staleness_bdays: int = 5,
    max_gap_bdays: int = 5,
    max_daily_move_pct: float = 45.0,
    max_recent_zero_volume: int = 2,
    as_of: date | None = None,
) -> QualityReport:
    """Validate an OHLCV frame (DatetimeIndex, OHLCV_COLUMNS).

    min_rows defaults to 250 because the technical agent needs a 200-day
    moving average plus buffer. as_of anchors the staleness check (defaults to
    today; tests pass a fixed date).
    """
    as_of = as_of or date.today()
    checks: list[QualityCheck] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append(QualityCheck(name=name, passed=passed, detail=detail))

    n = len(df)
    check("sufficient_history", n >= min_rows, f"{n} rows, need {min_rows}")

    if n == 0:
        return QualityReport(ticker=ticker, passed=False, checks=checks)

    nan_counts = df[OHLCV_COLUMNS].isna().sum()
    bad_nan = {col: int(c) for col, c in nan_counts.items() if c > 0}
    check("no_nan", not bad_nan, f"NaN values: {bad_nan}")

    price_cols = ["open", "high", "low", "close", "adj_close"]
    prices = df[price_cols].dropna()
    nonpositive = int((prices <= 0).any(axis=1).sum())
    check("positive_prices", nonpositive == 0, f"{nonpositive} rows with price <= 0")

    ohlc = df[["open", "high", "low", "close"]].dropna()
    eps = 1e-9
    hl_bad = int(
        (
            (ohlc["low"] > ohlc[["open", "close"]].min(axis=1) + eps)
            | (ohlc["high"] < ohlc[["open", "close"]].max(axis=1) - eps)
        ).sum()
    )
    check("high_low_consistency", hl_bad == 0, f"{hl_bad} rows violate low<=open/close<=high")

    # Gaps: consecutive bars more than max_gap_bdays business days apart
    # (weekends/holidays are fine; a week-long hole is not).
    dates64 = df.index.values.astype("datetime64[D]")
    if n >= 2:
        gaps = np.busday_count(dates64[:-1], dates64[1:])
        worst = int(gaps.max())
        check("no_date_gaps", worst <= max_gap_bdays,
              f"largest gap {worst} business days (max {max_gap_bdays})")
    else:
        check("no_date_gaps", True)

    last_date = df.index[-1].date()
    staleness = int(np.busday_count(dates64[-1], np.datetime64(as_of, "D")))
    check("fresh", staleness <= max_staleness_bdays,
          f"last bar {last_date} is {staleness} business days before {as_of}")

    # A huge single-day move in *adjusted* prices usually means a missed
    # split/dividend adjustment, not a real move.
    moves = df["adj_close"].pct_change().abs() * 100
    worst_move = float(moves.max()) if n >= 2 else 0.0
    check("no_split_artifacts", worst_move <= max_daily_move_pct,
          f"max daily move {worst_move:.1f}% (limit {max_daily_move_pct}%)")

    recent = df.tail(30)
    zero_vol = int((recent["volume"].fillna(0) <= 0).sum())
    check("recent_volume", zero_vol <= max_recent_zero_volume,
          f"{zero_vol} zero-volume days in last 30 (max {max_recent_zero_volume})")

    return QualityReport(ticker=ticker, passed=all(c.passed for c in checks), checks=checks)
