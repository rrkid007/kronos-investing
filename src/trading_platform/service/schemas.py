"""Public request/response contract for the paid service.

This is the only part of the platform that is a *published API*: once a paying
caller depends on it, field names and semantics are frozen. Keep it decoupled
from `core.models.AgentResult` — that is an internal contract free to churn.
The one thing deliberately shared is `Direction`, so a buyer's bullish/bearish
means exactly what yours does.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_platform.core.models import Direction

# Upper bound on accepted rows. The forecaster only ever reads the last
# `context_candles` (400) and the model's max_context is 512, so anything
# beyond this is parse cost with no signal benefit.
MAX_CANDLES = 2_000

# Below this the forecast is not worth selling, regardless of what the model
# will technically accept.
MIN_CANDLES = 128


class Candle(BaseModel):
    """One OHLCV bar. Asset-agnostic: no adj_close, no calendar assumptions."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class ScoreRequest(BaseModel):
    candles: list[Candle] = Field(min_length=MIN_CANDLES, max_length=MAX_CANDLES)
    symbol: str | None = Field(
        default=None,
        max_length=32,
        description="Free-text label echoed back in the receipt. Never used to "
                    "fetch data — the service scores only what you send.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**31 - 1,
        description="Set for a reproducible (and cacheable) forecast. Omit for a "
                    "fresh stochastic sample.",
    )

    @field_validator("candles")
    @classmethod
    def _strictly_increasing(cls, candles: list[Candle]) -> list[Candle]:
        for prev, nxt in zip(candles, candles[1:]):
            if nxt.timestamp <= prev.timestamp:
                raise ValueError(
                    f"candles must be strictly increasing in time; "
                    f"{nxt.timestamp.isoformat()} follows {prev.timestamp.isoformat()}"
                )
        return candles

    def to_frame(self) -> pd.DataFrame:
        """OHLCV frame in the shape KronosForecaster.predict_paths expects."""
        return pd.DataFrame(
            {
                "open": [c.open for c in self.candles],
                "high": [c.high for c in self.candles],
                "low": [c.low for c in self.candles],
                "close": [c.close for c in self.candles],
                "volume": [c.volume for c in self.candles],
            },
            index=pd.DatetimeIndex([c.timestamp for c in self.candles]),
        )


class ModelInfo(BaseModel):
    """Exactly which model produced the number, so results are auditable."""

    # `model_id` collides with pydantic's reserved `model_` prefix; we keep the
    # name because it matches KronosSettings and the published schema.
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    tokenizer_id: str
    horizon_days: int
    sample_count: int
    temperature: float
    top_p: float
    context_candles_used: int


class Receipt(BaseModel):
    """Dispute record. Join `input_hash` against the settlement tx hash from
    the x402 `X-PAYMENT-RESPONSE` header to prove what was paid for."""

    input_hash: str
    tier: str
    symbol: str | None = None
    seed: int | None = None
    cached: bool = False
    served_at: datetime
    compute_seconds: float


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Same rubric as the in-platform Kronos agent: 0-100, -8%/+8% anchored.
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    direction: Direction

    expected_return_pct: float
    return_std_pct: float
    p_up: float
    forecast_p10_pct: float
    forecast_p90_pct: float
    n_samples: int

    last_close: float
    data_as_of: datetime

    model_info: ModelInfo
    receipt: Receipt


class ErrorResponse(BaseModel):
    """Every non-2xx body. A non-2xx means no settlement occurred — the caller
    was not charged."""

    error: str
    detail: str
    retry_after_seconds: int | None = None
