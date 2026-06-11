"""Core data contracts shared by every agent and pipeline stage.

Every research agent returns exactly an AgentResult; the decision layer
consumes AgentResults and emits TradeDecisions; the execution layer consumes
approved Orders. These shapes are the seams between phases.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WATCHLIST = "watchlist"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FILLED = "filled"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    """Uniform output contract for every research agent.

    score: 0-100 per the agent's documented rubric.
    confidence: 0-1. A confidence of 0 means "ignore this signal" — the
    neutral fallback agents must return when their data source is unavailable,
    rather than guessing.
    """

    agent: str
    ticker: str
    run_id: str
    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    direction: Direction = Direction.NEUTRAL
    details: dict = Field(default_factory=dict)
    data_as_of: datetime | None = None

    @classmethod
    def neutral(cls, agent: str, ticker: str, run_id: str, reason: str) -> "AgentResult":
        """Neutral, zero-confidence result for when a data source is unavailable."""
        return cls(
            agent=agent,
            ticker=ticker,
            run_id=run_id,
            score=50.0,
            confidence=0.0,
            direction=Direction.NEUTRAL,
            details={"fallback_reason": reason},
        )


class Position(BaseModel):
    ticker: str
    qty: float = Field(gt=0.0)
    avg_cost: float = Field(gt=0.0)
    opened_at: date


class TradeDecision(BaseModel):
    run_id: str
    ticker: str
    action: Action
    final_score: float = Field(ge=0.0, le=100.0)
    signal_breakdown: dict = Field(default_factory=dict)
    sizing_hint: float | None = None  # target position value in account currency
    reason: str = ""


class Order(BaseModel):
    order_id: str
    run_id: str
    ticker: str
    side: OrderSide
    qty: float = Field(gt=0.0)
    status: OrderStatus = OrderStatus.AWAITING_APPROVAL
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    notes: str = ""
