"""Kronos Agent — score/confidence derivation on fake forecast paths, no torch."""

import numpy as np
import pytest

from tests.fixtures import FakeForecaster, make_ohlcv
from trading_platform.agents.kronos import KronosAgent, score_paths
from trading_platform.core.config import KronosSettings
from trading_platform.core.models import Direction

SETTINGS = KronosSettings()  # defaults: 10-day horizon, 8 samples


def make_agent(forecaster) -> KronosAgent:
    return KronosAgent(SETTINGS, forecaster=forecaster)


# --- score_paths (pure) -----------------------------------------------------

def test_flat_paths_score_neutral_50():
    paths = np.full((8, 10), 100.0)
    scored = score_paths(100.0, paths)
    assert scored["score"] == 50.0
    assert scored["expected_return_pct"] == 0.0


def test_anchor_mapping():
    up8 = np.full((8, 10), 108.0)   # +8% -> score 100
    down8 = np.full((8, 10), 92.0)  # -8% -> score 0
    assert score_paths(100.0, up8)["score"] == 100.0
    assert score_paths(100.0, down8)["score"] == 0.0
    beyond = np.full((8, 10), 130.0)  # +30% clamps at 100
    assert score_paths(100.0, beyond)["score"] == 100.0


def test_unanimous_paths_give_high_confidence():
    paths = 100.0 * (1 + np.linspace(0.02, 0.05, 8))[:, None] * np.ones((8, 10))
    scored = score_paths(100.0, paths)
    assert scored["p_up"] == 1.0
    assert scored["confidence"] == 0.9


def test_split_paths_give_low_confidence():
    finals = np.array([104.0, 103.0, 102.0, 101.0, 99.0, 98.0, 97.0, 96.0])
    paths = np.repeat(finals[:, None], 10, axis=1)
    scored = score_paths(100.0, paths)
    assert scored["p_up"] == 0.5
    assert scored["confidence"] == pytest.approx(0.3)


def test_few_samples_get_flat_default_confidence():
    paths = np.full((2, 10), 105.0)
    assert score_paths(100.0, paths)["confidence"] == 0.5


def test_percentiles_ordered():
    rng = np.random.default_rng(3)
    paths = 100.0 * (1 + rng.normal(0.01, 0.03, size=(50, 10)))
    scored = score_paths(100.0, paths)
    assert scored["forecast_p10_pct"] <= scored["expected_return_pct"] <= scored["forecast_p90_pct"]


# --- agent ------------------------------------------------------------------

def test_bullish_forecast():
    result = make_agent(FakeForecaster(final_return=0.05, spread=0.01)).analyze(
        "AAPL", "r1", make_ohlcv()
    )
    assert result.score > 60
    assert result.direction == Direction.BULLISH
    assert result.confidence == 0.9  # all paths agree upward
    assert result.details["horizon_days"] == 10
    assert result.details["n_samples"] == SETTINGS.sample_count


def test_bearish_forecast():
    result = make_agent(FakeForecaster(final_return=-0.05, spread=0.01)).analyze(
        "AAPL", "r1", make_ohlcv()
    )
    assert result.score < 40
    assert result.direction == Direction.BEARISH


def test_forecast_failure_returns_neutral():
    result = make_agent(FakeForecaster(fail=RuntimeError("CUDA OOM"))).analyze(
        "AAPL", "r1", make_ohlcv()
    )
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert "CUDA OOM" in result.details["fallback_reason"]


def test_missing_torch_returns_neutral_with_install_hint():
    result = make_agent(FakeForecaster(fail=ImportError("No module named 'torch'"))).analyze(
        "AAPL", "r1", make_ohlcv()
    )
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert "uv sync --extra kronos" in result.details["fallback_reason"]


def test_agent_is_deterministic_given_paths():
    df = make_ohlcv()
    a = make_agent(FakeForecaster(final_return=0.02)).analyze("AAPL", "r1", df)
    b = make_agent(FakeForecaster(final_return=0.02)).analyze("AAPL", "r1", df)
    assert a.score == b.score
    assert a.details == b.details
