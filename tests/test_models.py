import pytest
from pydantic import ValidationError

from trading_platform.core.models import AgentResult, Direction, Order, OrderSide


def test_score_bounds_enforced():
    with pytest.raises(ValidationError):
        AgentResult(agent="technical", ticker="AAPL", run_id="r1", score=101, confidence=0.5)
    with pytest.raises(ValidationError):
        AgentResult(agent="technical", ticker="AAPL", run_id="r1", score=-1, confidence=0.5)


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        AgentResult(agent="technical", ticker="AAPL", run_id="r1", score=50, confidence=1.5)


def test_neutral_fallback_is_ignorable():
    result = AgentResult.neutral("news", "AAPL", "r1", reason="finnhub unreachable")
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert result.direction == Direction.NEUTRAL
    assert result.details["fallback_reason"] == "finnhub unreachable"


def test_order_requires_positive_qty():
    with pytest.raises(ValidationError):
        Order(order_id="o1", run_id="r1", ticker="AAPL", side=OrderSide.BUY, qty=0)
