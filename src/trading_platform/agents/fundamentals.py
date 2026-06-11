"""Fundamentals Agent — deterministic rubric scoring of a yfinance snapshot.

Each sub-score (per the spec: growth, profitability, balance sheet, cash
flow, valuation) is the mean of its available components, each mapped
linearly from a "worst" to a "best" anchor (clamped 0-100). A sub-score with
no data is neutral 50 and lowers confidence — never a guess.

Final score = weighted sub-scores:
    growth 25% | profitability 25% | balance sheet 15% | cash flow 15% | valuation 20%

Confidence = 0.3 + 0.6 x (fraction of the 16 components with data); a
snapshot with zero coverage returns the standard neutral fallback.

Anchors (worst -> best); valuation anchors run high -> low because cheaper
is better:
    revenue_growth        0% -> 25%      gross_margin    20% -> 60%
    earnings_growth       0% -> 30%      operating_margin 5% -> 35%
    revenue_cagr_3y       0% -> 20%      profit_margin    3% -> 30%
                                         return_on_equity 5% -> 35%
    debt_to_equity      2.5 -> 0.3       fcf_margin       0% -> 25%
    current_ratio       0.8 -> 2.0       ocf_margin       5% -> 30%
    net_cash/mcap      -10% -> +10%
    trailing_pe          45 -> 12        ev_to_ebitda     30 -> 8
    forward_pe           40 -> 10        price_to_fcf     60 -> 15
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import pandas as pd

from trading_platform.core.models import AgentResult, Direction
from trading_platform.data.fundamentals import FundamentalsSnapshot, fetch_fundamentals

NAME = "fundamentals"

# field -> (worst, best) anchors, grouped by sub-score
RUBRIC: dict[str, dict[str, tuple[float, float]]] = {
    "growth": {
        "revenue_growth": (0.0, 0.25),
        "earnings_growth": (0.0, 0.30),
        "revenue_cagr_3y": (0.0, 0.20),
    },
    "profitability": {
        "gross_margin": (0.20, 0.60),
        "operating_margin": (0.05, 0.35),
        "profit_margin": (0.03, 0.30),
        "return_on_equity": (0.05, 0.35),
    },
    "balance_sheet": {
        "debt_to_equity": (2.5, 0.3),
        "current_ratio": (0.8, 2.0),
        "net_cash_to_market_cap": (-0.10, 0.10),
    },
    "cash_flow": {
        "fcf_margin": (0.0, 0.25),
        "ocf_margin": (0.05, 0.30),
    },
    "valuation": {
        "trailing_pe": (45.0, 12.0),
        "forward_pe": (40.0, 10.0),
        "ev_to_ebitda": (30.0, 8.0),
        "price_to_fcf": (60.0, 15.0),
    },
}

SUBSCORE_WEIGHTS = {
    "growth": 0.25,
    "profitability": 0.25,
    "balance_sheet": 0.15,
    "cash_flow": 0.15,
    "valuation": 0.20,
}

N_COMPONENTS = sum(len(fields) for fields in RUBRIC.values())


def linear_score(value: float, worst: float, best: float) -> float:
    """Map value onto 0-100 between the anchors, clamped. Works both directions."""
    t = (value - worst) / (best - worst)
    return max(0.0, min(1.0, t)) * 100.0


def score_snapshot(snapshot: FundamentalsSnapshot) -> dict:
    """Pure scoring: snapshot -> sub-scores, final score, coverage, components."""
    sub_scores: dict[str, float] = {}
    components: dict[str, float] = {}
    missing: list[str] = []
    available = 0

    for group, fields in RUBRIC.items():
        group_scores = []
        for field, (worst, best) in fields.items():
            value = getattr(snapshot, field)
            if value is None:
                missing.append(field)
                continue
            pts = linear_score(value, worst, best)
            components[field] = round(pts, 2)
            group_scores.append(pts)
            available += 1
        sub_scores[group] = (
            round(sum(group_scores) / len(group_scores), 2) if group_scores else 50.0
        )

    final = round(sum(sub_scores[g] * w for g, w in SUBSCORE_WEIGHTS.items()), 2)
    return {
        "final": final,
        "sub_scores": sub_scores,
        "components": components,
        "missing": missing,
        "coverage": available / N_COMPONENTS,
    }


class FundamentalsAgent:
    name = NAME

    def analyze(self, ticker: str, run_id: str, df: pd.DataFrame) -> AgentResult:
        """df (OHLCV) is unused — fundamentals come from their own snapshot."""
        try:
            snapshot = fetch_fundamentals(ticker)
        except Exception as exc:
            return AgentResult.neutral(self.name, ticker, run_id, f"fetch failed: {exc}")

        scored = score_snapshot(snapshot)
        if scored["coverage"] == 0:
            return AgentResult.neutral(self.name, ticker, run_id, "no fundamental data")

        score = scored["final"]
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
            confidence=round(0.3 + 0.6 * scored["coverage"], 2),
            direction=direction,
            data_as_of=(
                datetime.combine(snapshot.data_as_of, time(), tzinfo=timezone.utc)
                if snapshot.data_as_of
                else snapshot.fetched_at
            ),
            details={
                "growth_score": scored["sub_scores"]["growth"],
                "profitability_score": scored["sub_scores"]["profitability"],
                "balance_sheet_score": scored["sub_scores"]["balance_sheet"],
                "cash_flow_score": scored["sub_scores"]["cash_flow"],
                "valuation_score": scored["sub_scores"]["valuation"],
                "components": scored["components"],
                "missing_fields": scored["missing"],
                "coverage": round(scored["coverage"], 3),
                "raw": {
                    k: v
                    for k, v in snapshot.model_dump(
                        mode="json", exclude={"ticker", "fetched_at"}
                    ).items()
                    if v is not None
                },
            },
        )
