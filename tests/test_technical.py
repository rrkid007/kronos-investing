"""Technical Agent — deterministic scoring against fixed-seed golden fixtures."""

import pytest

from tests.fixtures import make_ohlcv
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.core.models import Direction

agent = TechnicalAgent()


def test_golden_score_default_fixture():
    """Pinned output for the seed-7 fixture — any rubric change must be deliberate."""
    result = agent.analyze("TEST", "r1", make_ohlcv())
    assert result.score == pytest.approx(38.95, abs=0.01)
    assert result.direction == Direction.BEARISH
    assert result.confidence == 0.9


def test_strong_uptrend_scores_high_and_bullish():
    df = make_ohlcv(n_rows=320, drift=0.003, daily_vol=0.008)
    result = agent.analyze("TEST", "r1", df)
    assert result.score == pytest.approx(95.51, abs=0.01)
    assert result.score >= 85
    assert result.direction == Direction.BULLISH
    assert result.details["trend"] == "uptrend"
    assert all(result.details["structure_signals"].values())


def test_strong_downtrend_scores_low_and_bearish():
    df = make_ohlcv(n_rows=320, drift=-0.003, daily_vol=0.008)
    result = agent.analyze("TEST", "r1", df)
    assert result.score == pytest.approx(13.37, abs=0.01)
    assert result.score <= 25
    assert result.direction == Direction.BEARISH
    assert result.details["trend"] == "downtrend"
    assert not any(result.details["structure_signals"].values())


def test_scoring_is_deterministic():
    df = make_ohlcv()
    a = agent.analyze("TEST", "r1", df)
    b = agent.analyze("TEST", "r1", df)
    assert a.score == b.score
    assert a.details == b.details


def test_score_always_within_bounds():
    for seed in range(10):
        for drift in (-0.005, 0.0, 0.005):
            df = make_ohlcv(seed=seed, drift=drift, daily_vol=0.02)
            result = agent.analyze("TEST", "r1", df)
            assert 0.0 <= result.score <= 100.0, f"seed={seed} drift={drift}"


def test_component_points_sum_to_score():
    result = agent.analyze("TEST", "r1", make_ohlcv())
    components = result.details["component_points"]
    assert sum(components.values()) == pytest.approx(result.score, abs=0.05)


def test_insufficient_history_returns_neutral_fallback():
    result = agent.analyze("TEST", "r1", make_ohlcv(n_rows=150))
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert "fallback_reason" in result.details


def test_details_carry_all_metrics():
    result = agent.analyze("TEST", "r1", make_ohlcv())
    for key in ("sma50", "sma200", "momentum_21d", "momentum_63d", "momentum_126d",
                "trend_r2", "volatility_annual", "volatility_bucket",
                "current_drawdown", "max_drawdown_252d", "component_points"):
        assert key in result.details, key
    assert result.data_as_of is not None
