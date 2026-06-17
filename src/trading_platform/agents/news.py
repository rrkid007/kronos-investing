"""News Agent — LLM classification of recent headlines, grounded and guarded.

Guardrails (PLAN.md S4):
- Output is schema-enforced JSON (NewsAnalysis) via grammar-constrained decoding.
- Every key driver must cite a headline index from the provided list; drivers
  citing nonexistent headlines are dropped, and if none survive, confidence is
  capped at 0.4 (the score has no verifiable grounding).
- No news, fetch failure, or LLM failure -> neutral zero-confidence fallback.

Confidence: min(0.9, 0.3 + 0.05 x n_items), capped at 0.4 without grounding.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from trading_platform.core.config import AppConfig
from trading_platform.core.llm import LLMError, OllamaClient
from trading_platform.core.models import AgentResult, Direction
from trading_platform.data.news import NewsItem, fetch_news, persist_news

NAME = "news"

SYSTEM_PROMPT = """You are a financial news analyst. You will receive recent news
headlines for one stock, numbered from 0. Classify each headline, then produce an
overall news score for the stock.

Two DIFFERENT numeric scales are used — do not mix them up:
- news_score: a 0-100 integer (0 = extremely negative news flow, 50 = neutral/mixed,
  100 = extremely positive).
- each classification's `sentiment`: a decimal between -1.0 and 1.0 (-1.0 = very
  negative, 0.0 = neutral, 1.0 = very positive). Never put a 0-100 value here.

Guidance:
- Weight material events (earnings surprises, lawsuits, regulatory actions,
  executive changes, product launches, analyst actions) over routine coverage.
- key_drivers: the 1-5 headlines that most drove your score. Each must reference
  a headline by its exact index number. Never invent events not in the headlines."""


class HeadlineClassification(BaseModel):
    index: int
    event_type: Literal[
        "earnings", "product", "regulatory", "lawsuit", "executive_change",
        "analyst_action", "mna", "macro", "other",
    ]
    sentiment: float = Field(
        ge=-1.0,
        le=1.0,
        description="Decimal from -1.0 (very negative) to 1.0 (very positive). "
        "NOT the 0-100 news_score scale.",
    )


class KeyDriver(BaseModel):
    claim: str
    headline_index: int


class NewsAnalysis(BaseModel):
    news_score: float = Field(ge=0.0, le=100.0)
    overall_sentiment: Literal["positive", "neutral", "negative"]
    classifications: list[HeadlineClassification]
    key_drivers: list[KeyDriver]


class NewsAgent:
    name = NAME

    def __init__(
        self,
        config: AppConfig,
        llm: OllamaClient | None = None,
        fetcher=None,
        conn: sqlite3.Connection | None = None,
    ):
        self.settings = config.settings.news
        self.llm = llm or OllamaClient(config.settings.llm)
        self._fetcher = fetcher  # tests inject
        self._conn = conn

    def _fetch(self, ticker: str) -> list[NewsItem]:
        if self._fetcher is not None:
            return self._fetcher(ticker)
        return fetch_news(
            ticker,
            lookback_days=self.settings.lookback_days,
            max_items=self.settings.max_items,
            finnhub_api_key=os.environ.get(self.settings.finnhub_api_key_env),
        )

    def analyze(self, ticker: str, run_id: str, df: pd.DataFrame) -> AgentResult:
        try:
            items = self._fetch(ticker)
        except Exception as exc:
            return AgentResult.neutral(self.name, ticker, run_id, f"news fetch failed: {exc}")

        if not items:
            return AgentResult.neutral(self.name, ticker, run_id, "no recent news")

        if self._conn is not None:
            persist_news(self._conn, items)

        numbered = "\n".join(
            f"[{i}] ({item.published_at.date()}, {item.source}) {item.headline}"
            + (f" — {item.summary}" if item.summary else "")
            for i, item in enumerate(items)
        )
        prompt = f"Stock: {ticker}\n\nRecent headlines:\n{numbered}"

        try:
            analysis = self.llm.generate(prompt, NewsAnalysis, system=SYSTEM_PROMPT)
        except LLMError as exc:
            return AgentResult.neutral(self.name, ticker, run_id, str(exc))

        # Grounding: drop drivers citing headlines that don't exist.
        valid_drivers = [
            d for d in analysis.key_drivers if 0 <= d.headline_index < len(items)
        ]
        dropped = len(analysis.key_drivers) - len(valid_drivers)

        score = round(min(100.0, max(0.0, analysis.news_score)), 2)
        confidence = min(0.9, 0.3 + 0.05 * len(items))
        if not valid_drivers:
            confidence = min(confidence, 0.4)  # score lacks verifiable grounding
        confidence = round(confidence, 2)

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
            data_as_of=items[0].published_at,
            details={
                "sentiment": analysis.overall_sentiment,
                "n_items": len(items),
                "sources": sorted({i.source for i in items}),
                "classifications": [c.model_dump() for c in analysis.classifications],
                "key_drivers": [
                    {"claim": d.claim, "headline": items[d.headline_index].headline}
                    for d in valid_drivers
                ],
                "ungrounded_drivers_dropped": dropped,
            },
        )
