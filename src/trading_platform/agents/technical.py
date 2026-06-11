"""Technical Analysis Agent — deterministic, pure pandas/numpy. No AI.

All metrics are computed on adjusted close (split/dividend correct).

Score rubric (sums to 0-100):

  Trend structure, 30 pts — four binary signals, 7.5 each:
      price > 50DMA, price > 200DMA, 50DMA > 200DMA (golden cross),
      50DMA rising (vs 10 bars ago)
  Momentum, 30 pts — 21/63/126-day returns, each mapped linearly
      from -15% (0 pts) to +15% (10 pts), clamped
  Trend quality, 20 pts — R^2 of a 126-day log-price regression,
      signed by slope: up + tight fit -> 20, down + tight fit -> 0,
      noisy/flat -> ~10
  Risk, 20 pts — drawdown from 252-day high (0% -> 10 pts, -30% -> 0)
      plus annualized volatility (<=15% -> 10 pts, >=60% -> 0)

Direction: bullish >= 60, bearish <= 40, else neutral.
Confidence: 0.9 with a full 252-bar year of history, 0.7 below that
(the data gate guarantees at least 250 bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.models import AgentResult, Direction

NAME = "technical"
TRADING_DAYS = 252


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _momentum_points(window_return: float) -> float:
    """Map a window return to 0-10 pts: -15% -> 0, 0% -> 5, +15% -> 10."""
    return _clamp((window_return + 0.15) / 0.30, 0.0, 1.0) * 10.0


def _trend_regression(prices: pd.Series) -> tuple[float, float]:
    """Linear regression on log price. Returns (annualized_slope, r_squared)."""
    y = np.log(prices.to_numpy())
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope) * TRADING_DAYS, r2


class TechnicalAgent:
    name = NAME

    def analyze(self, ticker: str, run_id: str, df: pd.DataFrame) -> AgentResult:
        if len(df) < 200:  # defensive; the data gate requires 250+
            return AgentResult.neutral(self.name, ticker, run_id,
                                       f"only {len(df)} bars; need 200")

        px = df["adj_close"]
        price = float(px.iloc[-1])

        sma50 = float(px.rolling(50).mean().iloc[-1])
        sma200 = float(px.rolling(200).mean().iloc[-1])
        sma50_prev = float(px.rolling(50).mean().iloc[-11])

        structure_signals = {
            "price_above_sma50": price > sma50,
            "price_above_sma200": price > sma200,
            "golden_cross": sma50 > sma200,
            "sma50_rising": sma50 > sma50_prev,
        }
        structure_pts = 7.5 * sum(structure_signals.values())

        momentum = {
            f"momentum_{w}d": float(px.iloc[-1] / px.iloc[-(w + 1)] - 1.0)
            for w in (21, 63, 126)
        }
        momentum_pts = sum(_momentum_points(r) for r in momentum.values())

        slope_annual, r2 = _trend_regression(px.tail(126))
        quality_pts = 10.0 + 10.0 * r2 if slope_annual > 0 else 10.0 - 10.0 * r2

        year = px.tail(TRADING_DAYS)
        current_drawdown = float(price / year.max() - 1.0)
        max_drawdown = float((year / year.cummax() - 1.0).min())
        drawdown_pts = _clamp(1.0 + current_drawdown / 0.30, 0.0, 1.0) * 10.0

        daily_returns = px.pct_change().tail(63).dropna()
        vol_annual = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
        vol_pts = _clamp((0.60 - vol_annual) / 0.45, 0.0, 1.0) * 10.0
        vol_bucket = "low" if vol_annual <= 0.20 else "moderate" if vol_annual <= 0.35 else "high"

        score = round(structure_pts + momentum_pts + quality_pts + drawdown_pts + vol_pts, 2)
        n_up = sum(structure_signals.values())
        trend = "uptrend" if n_up >= 3 else "downtrend" if n_up <= 1 else "sideways"
        direction = (
            Direction.BULLISH if score >= 60
            else Direction.BEARISH if score <= 40
            else Direction.NEUTRAL
        )

        return AgentResult(
            agent=self.name,
            ticker=ticker,
            run_id=run_id,
            score=score,
            confidence=0.9 if len(df) >= TRADING_DAYS else 0.7,
            direction=direction,
            data_as_of=df.index[-1].to_pydatetime(),
            details={
                "trend": trend,
                "price": round(price, 4),
                "sma50": round(sma50, 4),
                "sma200": round(sma200, 4),
                "structure_signals": structure_signals,
                **{k: round(v, 6) for k, v in momentum.items()},
                "trend_slope_annual": round(slope_annual, 6),
                "trend_r2": round(r2, 6),
                "volatility_annual": round(vol_annual, 6),
                "volatility_bucket": vol_bucket,
                "current_drawdown": round(current_drawdown, 6),
                "max_drawdown_252d": round(max_drawdown, 6),
                "component_points": {
                    "trend_structure": round(structure_pts, 2),
                    "momentum": round(momentum_pts, 2),
                    "trend_quality": round(quality_pts, 2),
                    "drawdown": round(drawdown_pts, 2),
                    "volatility": round(vol_pts, 2),
                },
            },
        )
