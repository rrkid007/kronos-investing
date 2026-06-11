"""Kronos Forecast Agent — score and confidence from forecast sample paths.

The forecaster produces `sample_count` stochastic 10-day close paths. From
the per-path final returns:

    expected_return = mean(path final close / last close - 1)
    score           = expected_return mapped linearly, -8% -> 0, +8% -> 100
    p_up            = fraction of paths ending positive
    confidence      = 0.3 + 0.6 x |2 x p_up - 1|   (path agreement; >= 4 paths)
                      0.5 flat when fewer than 4 paths (no dispersion signal)

Direction: bullish score >= 60 (~ +1.6% expected), bearish <= 40, else neutral.

If the kronos extra is not installed or inference fails, the agent returns
the standard neutral zero-confidence fallback — never a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.config import KronosSettings
from trading_platform.core.models import AgentResult, Direction

NAME = "kronos"
RETURN_ANCHOR = 0.08  # +/- expected-return range mapped onto the score scale


def score_paths(last_close: float, paths: np.ndarray) -> dict:
    """Pure derivation: (sample_count, horizon) close paths -> score fields."""
    finals = paths[:, -1]
    returns = finals / last_close - 1.0
    expected = float(returns.mean())
    p_up = float((returns > 0).mean())
    n = len(returns)

    score = round(max(0.0, min(1.0, (expected + RETURN_ANCHOR) / (2 * RETURN_ANCHOR))) * 100, 2)
    confidence = round(0.3 + 0.6 * abs(2 * p_up - 1.0), 2) if n >= 4 else 0.5

    return {
        "score": score,
        "confidence": confidence,
        "expected_return_pct": round(expected * 100, 3),
        "return_std_pct": round(float(returns.std()) * 100, 3),
        "p_up": round(p_up, 3),
        "forecast_p10_pct": round(float(np.percentile(returns, 10)) * 100, 3),
        "forecast_p90_pct": round(float(np.percentile(returns, 90)) * 100, 3),
        "n_samples": n,
    }


class KronosAgent:
    name = NAME

    def __init__(self, settings: KronosSettings, forecaster=None):
        self.settings = settings
        self._forecaster = forecaster  # tests inject a fake

    def _get_forecaster(self):
        if self._forecaster is None:
            from trading_platform.forecast.kronos_forecaster import KronosForecaster

            self._forecaster = KronosForecaster(self.settings)
        return self._forecaster

    def analyze(self, ticker: str, run_id: str, df: pd.DataFrame) -> AgentResult:
        try:
            forecaster = self._get_forecaster()
            paths = forecaster.predict_paths(
                df, self.settings.horizon_days, self.settings.sample_count
            )
        except ImportError as exc:
            return AgentResult.neutral(
                self.name, ticker, run_id,
                f"kronos extra not installed (uv sync --extra kronos): {exc}",
            )
        except Exception as exc:
            return AgentResult.neutral(self.name, ticker, run_id, f"forecast failed: {exc}")

        last_close = float(df["close"].iloc[-1])
        scored = score_paths(last_close, np.asarray(paths))

        score = scored.pop("score")
        confidence = scored.pop("confidence")
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
            confidence=confidence,
            direction=direction,
            data_as_of=df.index[-1].to_pydatetime(),
            details={
                "forecast_direction": direction.value,
                "horizon_days": self.settings.horizon_days,
                "last_close": round(last_close, 4),
                "model_id": self.settings.model_id,
                **scored,
            },
        )
