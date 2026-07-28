"""FastAPI app for the paid Kronos forecast service.

Separate app from `dashboard.app` on purpose. The dashboard is documented as
localhost-only with no auth, and it writes config YAML, stores API keys and
approves orders — it must never be the thing facing the open internet. This app
reads nothing, writes nothing, and touches no database.

Free:  GET /health, GET /v1/kronos/schema  (discovery — an agent must be able
       to learn the price before deciding to pay it)
Paid:  POST /v1/kronos/score, POST /v1/kronos/score/deep
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from trading_platform.core.config import KronosSettings
from trading_platform.core.models import Direction
from trading_platform.service import validation
from trading_platform.service.config import ServiceConfig
from trading_platform.service.payments import attach_payments
from trading_platform.service.runner import (
    CapacityExceeded,
    ForecastFailed,
    ForecastRunner,
    ForecastTimeout,
    ForecastUnavailable,
)
from trading_platform.service.schemas import (
    MAX_CANDLES,
    MIN_CANDLES,
    ErrorResponse,
    ModelInfo,
    Receipt,
    ScoreRequest,
    ScoreResponse,
)
from trading_platform.service.tiers import DEEP, STANDARD, TIERS, Tier

logger = logging.getLogger(__name__)


def _direction(score: float) -> Direction:
    """Mirrors agents.kronos.KronosAgent.analyze — buyers and the internal
    pipeline must read the same score the same way. Pinned by test_service.py."""
    if score >= 60:
        return Direction.BULLISH
    if score <= 40:
        return Direction.BEARISH
    return Direction.NEUTRAL


def _error(status: int, error: str, detail: str, retry_after: int | None = None) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, retry_after_seconds=retry_after)
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def create_service_app(
    kronos_settings: KronosSettings | None = None,
    config: ServiceConfig | None = None,
    forecaster=None,
    runner: ForecastRunner | None = None,
) -> FastAPI:
    settings = kronos_settings or KronosSettings()
    cfg = config or ServiceConfig.from_env()

    runner = runner or ForecastRunner(
        settings,
        forecaster=forecaster,
        max_queue=cfg.max_queue,
        timeout_seconds=cfg.timeout_seconds,
        cache_size=cfg.cache_size,
    )

    app = FastAPI(
        title="Kronos Forecast Service",
        version="0.1.0",
        description="Pay-per-call Kronos price-forecast scoring over x402.",
    )
    app.state.config = cfg
    app.state.runner = runner
    app.state.kronos_settings = settings

    # Raises if payments are demanded but unconfigured — better a dead service
    # than one quietly giving away GPU time.
    app.state.payments_active = attach_payments(app, cfg, [STANDARD, DEEP])

    # --- free endpoints ----------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "payments_active": app.state.payments_active,
            "network": cfg.network if app.state.payments_active else None,
            "queue_depth": runner._waiting,
            "queue_limit": cfg.max_queue,
        }

    @app.get("/v1/kronos/schema")
    def schema() -> dict:
        """Machine-readable discovery: what this sells, at what price, on what
        chain, and what shape the input must take."""
        return {
            "service": "kronos-forecast",
            "description": (
                "Kronos time-series forecast scoring. You supply OHLCV candles; "
                "the service returns a 0-100 score, a confidence, and the "
                "distribution of simulated forward returns. It never fetches "
                "market data on your behalf."
            ),
            "payment": {
                "protocol": "x402",
                "active": app.state.payments_active,
                "network": cfg.network if app.state.payments_active else None,
                "asset": "USDC",
                "pay_to": cfg.pay_to if app.state.payments_active else None,
            },
            "tiers": [
                {
                    "name": t.name,
                    "path": t.path,
                    "method": "POST",
                    "price": t.price_usd,
                    "sample_paths": t.sample_count,
                    "description": t.description,
                }
                for t in TIERS.values()
            ],
            "input": {
                "candles": {
                    "min": MIN_CANDLES,
                    "max": MAX_CANDLES,
                    "fields": ["timestamp", "open", "high", "low", "close", "volume"],
                    "ordering": "strictly increasing by timestamp",
                    "note": (
                        "Asset-agnostic — daily equity bars, 24/7 crypto bars and "
                        "intraday bars are all accepted. No adjusted-close column "
                        "is required or used."
                    ),
                },
                "symbol": "optional label, echoed on the receipt, never used to fetch",
                "seed": "optional int; makes the forecast reproducible and cacheable",
            },
            "output": {
                "score": "0-100, linear in expected return with -8% -> 0 and +8% -> 100",
                "confidence": "0.3-0.9 from path agreement; 0.5 when fewer than 4 paths",
                "direction": "bullish >= 60, bearish <= 40, else neutral",
                "receipt": (
                    "input_hash covers the candles and priced parameters; join it "
                    "against the settlement tx hash in X-PAYMENT-RESPONSE"
                ),
            },
            "model": {
                "model_id": settings.model_id,
                "tokenizer_id": settings.tokenizer_id,
                "horizon_days": settings.horizon_days,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "stochastic": (
                    "Sampling is stochastic. Identical input without a seed returns "
                    "a different draw each call; pass a seed for reproducibility."
                ),
            },
        }

    # --- paid endpoints ----------------------------------------------------

    async def _score(request: ScoreRequest, tier: Tier):
        df = request.to_frame()

        # Input gate first: malformed candles cost the caller nothing, because
        # a 4xx here means the x402 middleware never settles.
        problems = validation.validate_candles(df)
        if problems:
            return _error(422, "invalid_candles", "; ".join(problems))

        digest = validation.input_hash(
            df,
            tier=tier.name,
            model_id=settings.model_id,
            horizon=settings.horizon_days,
        )

        try:
            scored, cached, elapsed = await runner.score(
                df, tier, input_hash=digest, seed=request.seed
            )
        except CapacityExceeded as exc:
            return _error(503, "capacity_exceeded", str(exc), retry_after=5)
        except ForecastTimeout as exc:
            return _error(504, "forecast_timeout", str(exc), retry_after=30)
        except ForecastUnavailable as exc:
            return _error(503, "model_unavailable", str(exc), retry_after=60)
        except ForecastFailed as exc:
            logger.exception("forecast failed for hash=%s", digest[:12])
            return _error(502, "forecast_failed", str(exc))

        score = scored.pop("score")
        confidence = scored.pop("confidence")

        return ScoreResponse(
            score=score,
            confidence=confidence,
            direction=_direction(score),
            last_close=round(float(df["close"].iloc[-1]), 6),
            data_as_of=df.index[-1].to_pydatetime(),
            model_info=ModelInfo(
                model_id=settings.model_id,
                tokenizer_id=settings.tokenizer_id,
                horizon_days=settings.horizon_days,
                sample_count=tier.sample_count,
                temperature=settings.temperature,
                top_p=settings.top_p,
                context_candles_used=min(len(df), settings.context_candles),
            ),
            receipt=Receipt(
                input_hash=digest,
                tier=tier.name,
                symbol=request.symbol,
                seed=request.seed,
                cached=cached,
                served_at=datetime.now(timezone.utc),
                compute_seconds=elapsed,
            ),
            **scored,
        )

    @app.post(STANDARD.path, response_model=ScoreResponse)
    async def score_standard(request: ScoreRequest):
        return await _score(request, STANDARD)

    @app.post(DEEP.path, response_model=ScoreResponse)
    async def score_deep(request: ScoreRequest):
        return await _score(request, DEEP)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        """Last resort. Non-2xx means no settlement, so an unexpected crash
        must not be allowed to surface as a 200."""
        logger.exception("unhandled service error")
        return _error(500, "internal_error", type(exc).__name__)

    return app
