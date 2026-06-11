"""News Agent — grounding enforcement and fallback behavior, no network/LLM."""

from pathlib import Path

import pytest

from tests.fixtures import FakeLLM, make_news_items
from trading_platform.agents.news import (
    HeadlineClassification,
    KeyDriver,
    NewsAgent,
    NewsAnalysis,
)
from trading_platform.core.config import load_config
from trading_platform.core.llm import LLMError
from trading_platform.core.models import Direction

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config")


def make_analysis(score=75.0, drivers=None, sentiment="positive"):
    return NewsAnalysis(
        news_score=score,
        overall_sentiment=sentiment,
        classifications=[
            HeadlineClassification(index=0, event_type="earnings", sentiment=0.8),
            HeadlineClassification(index=1, event_type="product", sentiment=0.5),
        ],
        key_drivers=drivers if drivers is not None else [
            KeyDriver(claim="strong earnings beat", headline_index=0),
        ],
    )


def make_agent(llm, items=None, fetch_fail=None):
    def fetcher(ticker):
        if fetch_fail is not None:
            raise fetch_fail
        return items if items is not None else make_news_items(ticker)

    return NewsAgent(CONFIG, llm=llm, fetcher=fetcher)


def test_positive_news_scores_bullish():
    agent = make_agent(FakeLLM(response=make_analysis(score=82.0)))
    result = agent.analyze("TEST", "r1", None)
    assert result.score == 82.0
    assert result.direction == Direction.BULLISH
    assert result.details["sentiment"] == "positive"
    assert result.details["n_items"] == 5
    assert result.confidence == 0.55  # 0.3 + 0.05 * 5


def test_grounded_driver_carries_actual_headline():
    agent = make_agent(FakeLLM(response=make_analysis()))
    result = agent.analyze("TEST", "r1", None)
    assert result.details["key_drivers"][0]["headline"] == "TEST headline number 0"


def test_ungrounded_drivers_are_dropped_and_cap_confidence():
    bad_drivers = [
        KeyDriver(claim="invented event", headline_index=99),
        KeyDriver(claim="another invention", headline_index=-3),
    ]
    agent = make_agent(FakeLLM(response=make_analysis(drivers=bad_drivers)))
    result = agent.analyze("TEST", "r1", None)
    assert result.details["key_drivers"] == []
    assert result.details["ungrounded_drivers_dropped"] == 2
    assert result.confidence == 0.4  # no verifiable grounding


def test_mixed_drivers_keep_only_grounded():
    drivers = [
        KeyDriver(claim="real", headline_index=1),
        KeyDriver(claim="fake", headline_index=42),
    ]
    agent = make_agent(FakeLLM(response=make_analysis(drivers=drivers)))
    result = agent.analyze("TEST", "r1", None)
    assert len(result.details["key_drivers"]) == 1
    assert result.details["ungrounded_drivers_dropped"] == 1
    assert result.confidence == 0.55  # grounding survived, no cap


def test_no_news_returns_neutral():
    agent = make_agent(FakeLLM(response=make_analysis()), items=[])
    result = agent.analyze("TEST", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert result.details["fallback_reason"] == "no recent news"


def test_fetch_failure_returns_neutral():
    agent = make_agent(FakeLLM(response=make_analysis()), fetch_fail=ConnectionError("dns"))
    result = agent.analyze("TEST", "r1", None)
    assert result.confidence == 0.0
    assert "news fetch failed" in result.details["fallback_reason"]


def test_llm_failure_returns_neutral():
    agent = make_agent(FakeLLM(fail=LLMError("ollama unreachable")))
    result = agent.analyze("TEST", "r1", None)
    assert result.score == 50.0
    assert result.confidence == 0.0
    assert "unreachable" in result.details["fallback_reason"]


def test_confidence_scales_with_volume_capped():
    agent = make_agent(FakeLLM(response=make_analysis()), items=make_news_items(n=20))
    result = agent.analyze("TEST", "r1", None)
    assert result.confidence == pytest.approx(0.9)  # min(0.9, 0.3 + 0.05*20)
