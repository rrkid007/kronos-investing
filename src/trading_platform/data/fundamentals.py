"""Fundamentals snapshot: yfinance -> typed, scoreable fields.

Every field is optional — yfinance coverage varies by ticker — and the agent
scores only what's present, degrading confidence with coverage. The snapshot
is a current view, NOT point-in-time: per PLAN.md S2 it must never feed the
backtester.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FundamentalsSnapshot(BaseModel):
    ticker: str
    fetched_at: datetime
    data_as_of: date | None = None  # most recent reported quarter

    # growth
    revenue_growth: float | None = None       # latest quarter YoY
    earnings_growth: float | None = None      # latest quarter YoY
    revenue_cagr_3y: float | None = None      # from annual income statements

    # profitability
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    return_on_equity: float | None = None

    # balance sheet
    debt_to_equity: float | None = None       # ratio (yfinance % normalized /100)
    current_ratio: float | None = None
    net_cash_to_market_cap: float | None = None

    # cash flow
    fcf_margin: float | None = None
    ocf_margin: float | None = None

    # valuation
    trailing_pe: float | None = None
    forward_pe: float | None = None
    ev_to_ebitda: float | None = None
    price_to_fcf: float | None = None


def _fetch_raw(ticker: str) -> tuple[dict, pd.DataFrame | None]:
    """Pull info dict + annual income statement. Isolated so tests can patch."""
    t = yf.Ticker(ticker)
    info = t.get_info() or {}
    try:
        income = t.income_stmt
    except Exception:  # statements endpoint is flakier than info; degrade
        income = None
    return info, income


def _num(info: dict, key: str) -> float | None:
    value = info.get(key)
    if value is None or isinstance(value, str):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # NaN guard


def _revenue_cagr(income: pd.DataFrame | None) -> float | None:
    if income is None or income.empty or "Total Revenue" not in income.index:
        return None
    revenues = income.loc["Total Revenue"].dropna()
    if len(revenues) < 2:
        return None
    latest, oldest = float(revenues.iloc[0]), float(revenues.iloc[-1])
    years = len(revenues) - 1
    if oldest <= 0 or latest <= 0:
        return None
    return (latest / oldest) ** (1.0 / years) - 1.0


def fetch_fundamentals(ticker: str) -> FundamentalsSnapshot:
    info, income = _fetch_raw(ticker)

    mrq = info.get("mostRecentQuarter")
    data_as_of = (
        datetime.fromtimestamp(mrq, tz=timezone.utc).date()
        if isinstance(mrq, (int, float)) and mrq > 0
        else None
    )

    revenue = _num(info, "totalRevenue")
    fcf = _num(info, "freeCashflow")
    ocf = _num(info, "operatingCashflow")
    cash = _num(info, "totalCash")
    debt = _num(info, "totalDebt")
    mcap = _num(info, "marketCap")
    de = _num(info, "debtToEquity")

    return FundamentalsSnapshot(
        ticker=ticker,
        fetched_at=datetime.now(timezone.utc),
        data_as_of=data_as_of,
        revenue_growth=_num(info, "revenueGrowth"),
        earnings_growth=_num(info, "earningsGrowth"),
        revenue_cagr_3y=_revenue_cagr(income),
        gross_margin=_num(info, "grossMargins"),
        operating_margin=_num(info, "operatingMargins"),
        profit_margin=_num(info, "profitMargins"),
        return_on_equity=_num(info, "returnOnEquity"),
        debt_to_equity=de / 100.0 if de is not None else None,  # yfinance reports %
        current_ratio=_num(info, "currentRatio"),
        net_cash_to_market_cap=(
            (cash - debt) / mcap
            if cash is not None and debt is not None and mcap and mcap > 0
            else None
        ),
        fcf_margin=fcf / revenue if fcf is not None and revenue and revenue > 0 else None,
        ocf_margin=ocf / revenue if ocf is not None and revenue and revenue > 0 else None,
        trailing_pe=_num(info, "trailingPE"),
        forward_pe=_num(info, "forwardPE"),
        ev_to_ebitda=_num(info, "enterpriseToEbitda"),
        price_to_fcf=mcap / fcf if mcap is not None and fcf and fcf > 0 else None,
    )
