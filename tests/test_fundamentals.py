"""Fundamentals Agent — rubric scoring on synthetic snapshots, no network."""

import pytest

from tests.fixtures import make_snapshot
from trading_platform.agents.fundamentals import (
    N_COMPONENTS,
    RUBRIC,
    SUBSCORE_WEIGHTS,
    FundamentalsAgent,
    linear_score,
    score_snapshot,
)
from trading_platform.core.models import Direction
from trading_platform.data.fundamentals import FundamentalsSnapshot

agent = FundamentalsAgent()


def empty_snapshot(ticker="TEST"):
    from datetime import datetime, timezone

    return FundamentalsSnapshot(ticker=ticker, fetched_at=datetime(2026, 6, 11, tzinfo=timezone.utc))


# --- linear_score -----------------------------------------------------------

def test_linear_score_ascending_and_clamped():
    assert linear_score(0.25, 0.0, 0.25) == 100.0
    assert linear_score(0.0, 0.0, 0.25) == 0.0
    assert linear_score(0.125, 0.0, 0.25) == 50.0
    assert linear_score(5.0, 0.0, 0.25) == 100.0   # clamped high
    assert linear_score(-5.0, 0.0, 0.25) == 0.0    # clamped low


def test_linear_score_inverted_anchors():
    # PE: 45 worst -> 12 best
    assert linear_score(12.0, 45.0, 12.0) == 100.0
    assert linear_score(45.0, 45.0, 12.0) == 0.0
    assert linear_score(8.0, 45.0, 12.0) == 100.0  # cheaper than best -> clamped


# --- score_snapshot ---------------------------------------------------------

def test_full_snapshot_scores_all_components():
    scored = score_snapshot(make_snapshot())
    assert scored["coverage"] == 1.0
    assert scored["missing"] == []
    assert len(scored["components"]) == N_COMPONENTS
    assert 0.0 <= scored["final"] <= 100.0


def test_weighted_sum_consistency():
    scored = score_snapshot(make_snapshot())
    expected = sum(scored["sub_scores"][g] * w for g, w in SUBSCORE_WEIGHTS.items())
    assert scored["final"] == pytest.approx(expected, abs=0.01)


def test_great_cheap_business_beats_weak_expensive_one():
    great = make_snapshot(
        revenue_growth=0.30, earnings_growth=0.35, revenue_cagr_3y=0.25,
        gross_margin=0.65, operating_margin=0.40, profit_margin=0.32,
        return_on_equity=0.40, debt_to_equity=0.1, current_ratio=2.5,
        net_cash_to_market_cap=0.15, fcf_margin=0.30, ocf_margin=0.35,
        trailing_pe=10.0, forward_pe=9.0, ev_to_ebitda=7.0, price_to_fcf=12.0,
    )
    weak = make_snapshot(
        revenue_growth=-0.05, earnings_growth=-0.10, revenue_cagr_3y=-0.02,
        gross_margin=0.15, operating_margin=0.02, profit_margin=0.01,
        return_on_equity=0.03, debt_to_equity=3.0, current_ratio=0.7,
        net_cash_to_market_cap=-0.20, fcf_margin=-0.05, ocf_margin=0.02,
        trailing_pe=55.0, forward_pe=48.0, ev_to_ebitda=35.0, price_to_fcf=80.0,
    )
    great_scored, weak_scored = score_snapshot(great), score_snapshot(weak)
    assert great_scored["final"] >= 95.0
    assert weak_scored["final"] <= 5.0


def test_missing_group_is_neutral_50():
    snap = empty_snapshot()
    snap = snap.model_copy(update={"gross_margin": 0.60, "operating_margin": 0.35})
    scored = score_snapshot(snap)
    assert scored["sub_scores"]["growth"] == 50.0       # no data -> neutral
    assert scored["sub_scores"]["valuation"] == 50.0
    assert scored["sub_scores"]["profitability"] == 100.0
    assert scored["coverage"] == pytest.approx(2 / N_COMPONENTS)


def test_rubric_groups_cover_disjoint_fields():
    seen = set()
    for fields in RUBRIC.values():
        for f in fields:
            assert f not in seen, f"{f} appears in two groups"
            seen.add(f)
    assert sum(SUBSCORE_WEIGHTS.values()) == pytest.approx(1.0)


# --- agent ------------------------------------------------------------------

def test_agent_scores_full_snapshot(monkeypatch):
    monkeypatch.setattr(
        "trading_platform.agents.fundamentals.fetch_fundamentals",
        lambda t: make_snapshot(ticker=t),
    )
    result = agent.analyze("AAPL", "r1", None)
    assert 0.0 <= result.score <= 100.0
    assert result.confidence == 0.9  # full coverage
    assert result.details["coverage"] == 1.0
    for key in ("growth_score", "profitability_score", "balance_sheet_score",
                "cash_flow_score", "valuation_score"):
        assert key in result.details
    assert result.data_as_of is not None


def test_agent_zero_coverage_returns_neutral(monkeypatch):
    monkeypatch.setattr(
        "trading_platform.agents.fundamentals.fetch_fundamentals",
        lambda t: empty_snapshot(t),
    )
    result = agent.analyze("AAPL", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert result.details["fallback_reason"] == "no fundamental data"


def test_agent_fetch_failure_returns_neutral(monkeypatch):
    def boom(ticker):
        raise ConnectionError("yahoo down")

    monkeypatch.setattr("trading_platform.agents.fundamentals.fetch_fundamentals", boom)
    result = agent.analyze("AAPL", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert "yahoo down" in result.details["fallback_reason"]


def test_partial_coverage_lowers_confidence(monkeypatch):
    # Mid-anchor values: gross_margin 0.40 -> 50 pts, trailing_pe 28.5 -> 50 pts,
    # so the final score stays neutral while coverage is 2/16.
    snap = empty_snapshot().model_copy(update={"gross_margin": 0.40, "trailing_pe": 28.5})
    monkeypatch.setattr(
        "trading_platform.agents.fundamentals.fetch_fundamentals", lambda t: snap
    )
    result = agent.analyze("AAPL", "r1", None)
    assert 0.3 < result.confidence < 0.5  # 0.3 + 0.6 * (2/16)
    assert result.direction == Direction.NEUTRAL


def test_direction_thresholds(monkeypatch):
    strong = make_snapshot(
        revenue_growth=0.30, earnings_growth=0.35, revenue_cagr_3y=0.25,
        gross_margin=0.65, operating_margin=0.40, profit_margin=0.32,
        return_on_equity=0.40, debt_to_equity=0.1, current_ratio=2.5,
        net_cash_to_market_cap=0.15, fcf_margin=0.30, ocf_margin=0.35,
        trailing_pe=10.0, forward_pe=9.0, ev_to_ebitda=7.0, price_to_fcf=12.0,
    )
    monkeypatch.setattr(
        "trading_platform.agents.fundamentals.fetch_fundamentals", lambda t: strong
    )
    assert agent.analyze("X", "r1", None).direction == Direction.BULLISH
