"""Serialised GPU execution for the paid endpoints.

`KronosForecaster` loads one model per process and `KronosPredictor.predict` is
GPU-bound and not thread-safe. FastAPI runs sync handlers in a threadpool, so
without this two paying callers would enter the same predictor concurrently.

Everything here exists to protect one invariant:

    the caller is charged only if a forecast was actually produced

x402 settles after the handler returns 2xx, so every failure path below must
raise rather than degrade. This is the one place the service deliberately
diverges from `agents.kronos.KronosAgent`, which swallows failures into a
neutral zero-confidence result — correct inside the pipeline (a dead signal
should be ignorable), a billing bug in a shop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

from trading_platform.agents.kronos import score_paths
from trading_platform.core.config import KronosSettings
from trading_platform.service.tiers import Tier

logger = logging.getLogger(__name__)


class ForecastUnavailable(RuntimeError):
    """The model cannot run at all (kronos extra missing, weights absent)."""


class ForecastFailed(RuntimeError):
    """Inference started and blew up (CUDA OOM, bad shapes)."""


class CapacityExceeded(RuntimeError):
    """Too many callers already queued; shed load rather than time out."""


class ForecastTimeout(RuntimeError):
    """Queued or ran past the deadline. Must stay under the x402 payment
    authorisation's maxTimeoutSeconds, or we accept payments we cannot honour."""


def _seed_torch(seed: int) -> None:
    """Best-effort reproducibility. Not bit-exact across devices or driver
    versions — the receipt records the seed so a buyer knows what was asked
    for, not that two machines will agree to the last decimal."""
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ForecastRunner:
    """Single-flight forecast execution with admission control and an LRU cache.

    forecaster: injected in tests (see tests/fixtures.FakeForecaster); built
    lazily from settings in production so importing this module never pulls torch.
    """

    def __init__(
        self,
        settings: KronosSettings,
        forecaster=None,
        *,
        max_queue: int = 8,
        timeout_seconds: float = 60.0,
        cache_size: int = 256,
    ):
        self.settings = settings
        self._forecaster = forecaster
        self.max_queue = max_queue
        self.timeout_seconds = timeout_seconds
        self.cache_size = cache_size

        self._gpu = asyncio.Semaphore(1)
        self._waiting = 0
        self._cache: OrderedDict[tuple, dict] = OrderedDict()

    # --- model -------------------------------------------------------------

    def _get_forecaster(self):
        if self._forecaster is None:
            try:
                from trading_platform.forecast.kronos_forecaster import KronosForecaster
            except ImportError as exc:  # pragma: no cover - needs torch absent
                raise ForecastUnavailable(
                    f"kronos extra not installed on this host: {exc}"
                ) from exc
            self._forecaster = KronosForecaster(self.settings)
        return self._forecaster

    # --- cache -------------------------------------------------------------

    def _cache_get(self, key: tuple) -> dict | None:
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
        return hit

    def _cache_put(self, key: tuple, value: dict) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # --- execution ---------------------------------------------------------

    def _predict_sync(self, df: pd.DataFrame, tier: Tier, seed: int | None) -> np.ndarray:
        """Blocking. Runs on a worker thread, one at a time."""
        forecaster = self._get_forecaster()
        if seed is not None:
            _seed_torch(seed)
        try:
            paths = forecaster.predict_paths(
                df, self.settings.horizon_days, tier.sample_count
            )
        except ImportError as exc:
            raise ForecastUnavailable(str(exc)) from exc
        except Exception as exc:
            raise ForecastFailed(f"{type(exc).__name__}: {exc}") from exc
        return np.asarray(paths)

    async def score(
        self,
        df: pd.DataFrame,
        tier: Tier,
        *,
        input_hash: str,
        seed: int | None = None,
    ) -> tuple[dict, bool, float]:
        """Return (scored fields, cache_hit, compute_seconds).

        Only seeded requests are cached: an unseeded call is a promise of a
        fresh stochastic sample, and serving a stored one would be selling a
        different product than advertised.
        """
        cacheable = seed is not None
        key = (input_hash, tier.name, seed)

        if cacheable:
            hit = self._cache_get(key)
            if hit is not None:
                return dict(hit), True, 0.0

        if self._waiting >= self.max_queue:
            raise CapacityExceeded(
                f"{self._waiting} requests already queued (limit {self.max_queue})"
            )

        started = time.perf_counter()
        self._waiting += 1
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with self._gpu:
                    paths = await asyncio.to_thread(self._predict_sync, df, tier, seed)
        except TimeoutError as exc:
            raise ForecastTimeout(
                f"exceeded {self.timeout_seconds:.0f}s while queued or running"
            ) from exc
        finally:
            self._waiting -= 1

        elapsed = time.perf_counter() - started

        last_close = float(df["close"].iloc[-1])
        scored = score_paths(last_close, paths)
        if cacheable:
            self._cache_put(key, dict(scored))

        logger.info(
            "forecast served tier=%s samples=%d hash=%s in %.2fs",
            tier.name, tier.sample_count, input_hash[:12], elapsed,
        )
        return scored, False, round(elapsed, 3)
