"""Kronos inference wrapper.

Loads the vendored Kronos model + tokenizer lazily and exactly once per
process (the scanner reuses one forecaster across all tickers in a run).
Device auto-detects cuda -> mps -> cpu, so the same code runs on the DGX
Spark and a CPU-only dev box.

Heavy imports (torch) happen inside _load(), never at module import — the
rest of the platform must work without the forecast extra installed
(`uv sync --extra kronos` to enable).
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from trading_platform.core.config import KronosSettings

logger = logging.getLogger(__name__)

CLOSE_DIM = 3  # column order inside Kronos: open, high, low, close, volume, amount


class KronosForecaster:
    def __init__(self, settings: KronosSettings):
        self.settings = settings
        self._predictor = None
        self.device: str | None = None
        self.load_seconds: float | None = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        started = time.perf_counter()
        from trading_platform.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(self.settings.tokenizer_id)
        model = Kronos.from_pretrained(self.settings.model_id)
        self._predictor = KronosPredictor(
            model, tokenizer, max_context=self.settings.max_context
        )
        self.device = self._predictor.device
        self.load_seconds = round(time.perf_counter() - started, 2)
        logger.info(
            "kronos loaded: %s on %s in %.1fs",
            self.settings.model_id, self.device, self.load_seconds,
        )

    def predict_paths(self, df: pd.DataFrame, horizon: int, sample_count: int) -> np.ndarray:
        """Forecast close-price sample paths.

        df: gate-passed OHLCV frame (DatetimeIndex). Raw (unadjusted) candles
        are used — Kronos models the traded series, and a 10-day relative
        return is insensitive to historical adjustment basis.

        Returns ndarray (sample_count, horizon) of predicted closes.
        """
        self._load()

        window = df.tail(self.settings.context_candles).copy()
        x_df = window[["open", "high", "low", "close", "volume"]].copy()
        x_df["amount"] = x_df["close"] * x_df["volume"]

        x_timestamp = pd.Series(window.index)
        y_timestamp = pd.Series(
            pd.bdate_range(start=window.index[-1] + timedelta(days=1), periods=horizon)
        )

        preds = self._predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=self.settings.temperature,
            top_p=self.settings.top_p,
            sample_count=sample_count,
            verbose=False,
            return_samples=True,  # vendored patch: per-sample paths, not the mean
        )
        return preds[:, :, CLOSE_DIM]
