"""SEC Filing Agent — LLM risk analysis of 10-K/10-Q sections, grounded and cached.

Guardrails (PLAN.md S4):
- Schema-enforced JSON output (FilingAnalysis).
- Every finding must include a verbatim quote from the provided filing text;
  quotes that don't appear in the text are dropped. Heavy dropping (over half)
  cuts confidence — the analysis wasn't well grounded.
- No filing, fetch failure, or LLM failure -> neutral zero-confidence fallback.

Caching: filings change quarterly, the pipeline runs daily. Analyses are
cached in the `filings` table keyed by accession number, so each filing is
sent through the LLM exactly once.

Score: 0 = severe long-term risk profile, 100 = clean. Higher is better,
consistent with every other agent.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, time, timezone
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from trading_platform.core.config import AppConfig
from trading_platform.core.llm import LLMError, OllamaClient
from trading_platform.core.models import AgentResult, Direction
from trading_platform.data.sec import FilingDoc, fetch_latest_filing

NAME = "sec_filing"

SYSTEM_PROMPT = """You are a securities filing analyst reviewing excerpts from a
company's latest 10-K or 10-Q (risk factors and management discussion). Identify
concrete long-term risk indicators: debt concerns, litigation, regulatory exposure,
material weaknesses, going-concern language, competitive threats.

For every finding you MUST include a short verbatim quote (under 30 words) copied
exactly from the provided text. Do not paraphrase inside quotes; do not invent
risks not present in the text.

filing_score: 100 = clean filing with routine boilerplate risks only,
50 = typical risk profile, 0 = severe red flags (going concern, material
weakness, major litigation). Judge only from the text provided."""


class RiskFinding(BaseModel):
    category: Literal[
        "debt", "litigation", "regulatory", "material_weakness",
        "going_concern", "competition", "other",
    ]
    severity: Literal["low", "medium", "high"]
    quote: str


class FilingAnalysis(BaseModel):
    filing_score: float = Field(ge=0.0, le=100.0)
    summary: str
    findings: list[RiskFinding]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def quote_is_grounded(quote: str, source_text: str) -> bool:
    return len(quote.strip()) >= 10 and _normalize(quote) in _normalize(source_text)


class SECFilingAgent:
    name = NAME

    def __init__(
        self,
        config: AppConfig,
        llm: OllamaClient | None = None,
        fetcher=None,
        conn: sqlite3.Connection | None = None,
    ):
        self.settings = config.settings.sec
        self.llm = llm or OllamaClient(config.settings.llm)
        self._fetcher = fetcher  # tests inject
        self._conn = conn

    def _fetch(self, ticker: str) -> FilingDoc | None:
        if self._fetcher is not None:
            return self._fetcher(ticker)
        return fetch_latest_filing(
            ticker,
            identity=self.settings.identity,
            max_risk_chars=self.settings.max_risk_chars,
            max_mdna_chars=self.settings.max_mdna_chars,
        )

    def analyze(self, ticker: str, run_id: str, df: pd.DataFrame) -> AgentResult:
        try:
            doc = self._fetch(ticker)
        except Exception as exc:
            return AgentResult.neutral(self.name, ticker, run_id, f"filing fetch failed: {exc}")

        if doc is None:
            return AgentResult.neutral(self.name, ticker, run_id, "no 10-K/10-Q found")

        cached = self._cache_get(doc.accession_no)
        if cached is not None:
            return self._result_from_cache(ticker, run_id, doc, cached)

        prompt = (
            f"Company: {ticker} — {doc.form_type} filed {doc.filing_date}\n\n"
            f"RISK FACTORS (excerpt):\n{doc.risk_factors or '(not available)'}\n\n"
            f"MANAGEMENT DISCUSSION (excerpt):\n{doc.mdna or '(not available)'}"
        )
        try:
            analysis = self.llm.generate(prompt, FilingAnalysis, system=SYSTEM_PROMPT)
        except LLMError as exc:
            return AgentResult.neutral(self.name, ticker, run_id, str(exc))

        grounded = [f for f in analysis.findings if quote_is_grounded(f.quote, doc.combined_text)]
        dropped = len(analysis.findings) - len(grounded)

        confidence = 0.7
        if analysis.findings and dropped > len(analysis.findings) / 2:
            confidence = 0.4  # most findings weren't actually in the text

        payload = {
            "score": round(analysis.filing_score, 2),
            "confidence": confidence,
            "details": {
                "form_type": doc.form_type,
                "filing_date": doc.filing_date.isoformat(),
                "accession_no": doc.accession_no,
                "summary": analysis.summary,
                "risk_terms": len(grounded),
                "findings": [f.model_dump() for f in grounded],
                "ungrounded_findings_dropped": dropped,
            },
        }
        self._cache_put(doc, payload)
        return self._build_result(ticker, run_id, doc, payload)

    # --- cache ---------------------------------------------------------------

    def _cache_get(self, accession_no: str) -> dict | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT summary FROM filings WHERE accession_no = ? AND summary IS NOT NULL",
            (accession_no,),
        ).fetchone()
        return json.loads(row["summary"]) if row else None

    def _cache_put(self, doc: FilingDoc, payload: dict) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            INSERT INTO filings (ticker, form_type, filing_date, accession_no, analyzed_at, summary)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (accession_no) DO UPDATE SET
                analyzed_at = excluded.analyzed_at, summary = excluded.summary
            """,
            (
                doc.ticker, doc.form_type, doc.filing_date.isoformat(), doc.accession_no,
                datetime.now(tz=timezone.utc).isoformat(), json.dumps(payload),
            ),
        )
        self._conn.commit()

    def _result_from_cache(
        self, ticker: str, run_id: str, doc: FilingDoc, payload: dict
    ) -> AgentResult:
        payload = dict(payload)
        payload["details"] = {**payload["details"], "from_cache": True}
        return self._build_result(ticker, run_id, doc, payload)

    def _build_result(
        self, ticker: str, run_id: str, doc: FilingDoc, payload: dict
    ) -> AgentResult:
        score = payload["score"]
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
            confidence=payload["confidence"],
            direction=direction,
            data_as_of=datetime.combine(doc.filing_date, time(), tzinfo=timezone.utc),
            details=payload["details"],
        )
