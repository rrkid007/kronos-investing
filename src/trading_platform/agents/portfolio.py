"""Portfolio Agent — deterministic position sizing and portfolio-fit checks.

Sizing (PLAN.md S5):
    target = equity x base_position_pct
             x score_scalar   (final 70 -> 0.75, 100 -> 1.25, clamped 0.5-1.25)
             x vol_scalar     (target_vol / realized_vol, clamped 0.5-1.5)
then trimmed by, in order: max single-position cap, sector room, cash above
the reserve floor. A buy that trims below min_position_value (or to zero
whole shares) is rejected.

Hard rejections: ticker already held (no pyramiding), portfolio at
max_positions, sector at its cap, cash at the reserve floor.

fit_score starts at 100 and loses points for trims and high volatility —
it measures how comfortably the trade fits, not whether it's allowed
(that's the Risk Engine's job, phase 7).

Buy candidates are assessed in descending final-score order against a
running state, so earlier (stronger) candidates consume cash and sector
room before weaker ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from trading_platform.core.config import RiskLimits, Watchlist
from trading_platform.core.models import Position, TradeDecision

TRADING_DAYS = 252


def realized_vol(df: pd.DataFrame, window: int = 63) -> float:
    returns = df["adj_close"].pct_change().tail(window).dropna()
    if len(returns) < 10:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PortfolioState(BaseModel):
    """Running account view the agent debits as it approves buys."""

    cash: float
    equity: float
    positions: list[Position] = Field(default_factory=list)
    sector_values: dict[str, float] = Field(default_factory=dict)
    pending_buys: list[str] = Field(default_factory=list)

    @property
    def held_tickers(self) -> set[str]:
        return {p.ticker for p in self.positions} | set(self.pending_buys)

    @property
    def position_count(self) -> int:
        return len(self.positions) + len(self.pending_buys)

    def apply_buy(self, ticker: str, sector: str, cost: float) -> None:
        self.cash -= cost
        self.sector_values[sector] = self.sector_values.get(sector, 0.0) + cost
        self.pending_buys.append(ticker)


class PortfolioAssessment(BaseModel):
    ticker: str
    approved: bool
    qty: int = 0
    target_value: float = 0.0
    fit_score: float = 0.0
    checks: list[str] = Field(default_factory=list)
    rejection: str | None = None


class PortfolioAgent:
    def __init__(self, risk: RiskLimits, watchlist: Watchlist):
        self.risk = risk
        self.watchlist = watchlist

    def assess_buy(
        self,
        decision: TradeDecision,
        price: float,
        df: pd.DataFrame,
        state: PortfolioState,
        regime_scalar: float = 1.0,
    ) -> PortfolioAssessment:
        ticker = decision.ticker
        sizing = self.risk.sizing
        checks: list[str] = []

        def reject(reason: str) -> PortfolioAssessment:
            return PortfolioAssessment(
                ticker=ticker, approved=False, checks=checks, rejection=reason
            )

        if ticker in state.held_tickers:
            return reject("already held (no pyramiding)")
        if state.position_count >= sizing.max_positions:
            return reject(f"portfolio at max positions ({sizing.max_positions})")
        if price is None or price <= 0:
            return reject("no valid price")

        fit = 100.0

        # --- base size, scaled by score and volatility
        base = state.equity * sizing.base_position_pct / 100
        score_scalar = _clamp(0.75 + (decision.final_score - 70.0) / 30.0 * 0.5, 0.5, 1.25)
        vol = realized_vol(df)
        vol_scalar = _clamp(sizing.target_vol / vol, 0.5, 1.5) if vol > 0 else 1.0
        target = base * score_scalar * vol_scalar * regime_scalar
        checks.append(
            f"base {base:.0f} x score_scalar {score_scalar:.2f} "
            f"x vol_scalar {vol_scalar:.2f} (vol {vol:.2f}) "
            f"x regime_scalar {regime_scalar:.2f} = {target:.0f}"
        )
        if vol_scalar < 0.75:
            fit -= 10  # high-volatility name, size already cut
        if regime_scalar < 1.0:
            checks.append(f"macro regime scaling new positions by {regime_scalar:.2f}")

        # --- single-position cap
        cap = state.equity * self.risk.max_position_pct / 100
        if target > cap:
            target = cap
            checks.append(f"trimmed to max position cap {cap:.0f}")

        # --- sector exposure
        sector = self.watchlist.sector_of(ticker) or "Unknown"
        sector_cap = state.equity * self.risk.max_sector_pct / 100
        sector_room = sector_cap - state.sector_values.get(sector, 0.0)
        if sector_room <= 0:
            return reject(f"sector {sector} at cap ({self.risk.max_sector_pct}%)")
        if target > sector_room:
            target = sector_room
            fit -= 15
            checks.append(f"trimmed to sector room {sector_room:.0f} ({sector})")

        # --- cash above the reserve floor
        reserve = state.equity * self.risk.min_cash_reserve_pct / 100
        available = state.cash - reserve
        if available < sizing.min_position_value:
            return reject(
                f"available cash above reserve ({available:.0f}) below minimum position"
            )
        if target > available:
            target = available
            fit -= 15
            checks.append(f"trimmed to available cash {available:.0f}")

        # --- whole shares
        qty = int(target // price)
        value = qty * price
        if qty < 1 or value < sizing.min_position_value:
            return reject(f"sized below minimum ({value:.0f} < {sizing.min_position_value:.0f})")

        return PortfolioAssessment(
            ticker=ticker,
            approved=True,
            qty=qty,
            target_value=round(value, 2),
            fit_score=_clamp(fit, 0.0, 100.0),
            checks=checks,
        )
