"""Pre-approval research memos — ADVISORY ONLY (phase 16).

Before a human approves a pending order, an LLM writes a Pershing-style
review memo: thesis, bull case, bear case, variant perception, and explicit
kill criteria — grounded ONLY in data the platform already gathered (agent
scores and details, the decision audit trail, position context, macro
regime). The memo is for the human's eyes; its recommendation gates nothing,
and a failed memo never blocks an order.

The model routes to an external OpenAI-compatible endpoint when configured
(a frontier model writes much better adversarial analysis than a local 32B),
falling back to the local Ollama otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from trading_platform.core.config import AppConfig
from trading_platform.core.llm import LLMError, OllamaClient, OpenAICompatClient
from trading_platform.core.models import utcnow

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an investment research analyst writing a pre-trade review
memo for a HUMAN who must approve or reject a proposed paper trade. You are advisory
only; you cannot place trades and your recommendation does not auto-execute anything.

Ground every claim ONLY in the data provided. Never invent facts, prices, events, or
news. Where the data is thin or a signal is dead (zero confidence), say so plainly.

Write:
- thesis: 2-3 sentences — what this trade is and why the system proposed it
- bull_case: the 3 strongest data-backed reasons it works
- bear_case: the 3 biggest risks, argued as honestly as the bull case
- variant_perception: what the signal blend sees that a casual look would miss
  (or state plainly that there is none)
- kill_criteria: 3-5 specific, MEASURABLE conditions that should break this thesis
  (price levels, score thresholds, events — not vague feelings)
- approval_recommendation: approve / reject / needs_review
- confidence_note: 1-2 sentences on the quality and coverage of the underlying signals
"""


class ResearchMemo(BaseModel):
    thesis: str
    bull_case: list[str] = Field(min_length=1, max_length=5)
    bear_case: list[str] = Field(min_length=1, max_length=5)
    variant_perception: str
    kill_criteria: list[str] = Field(min_length=1, max_length=6)
    approval_recommendation: Literal["approve", "reject", "needs_review"]
    confidence_note: str


def build_memo_client(config: AppConfig):
    """External endpoint when configured, local Ollama otherwise."""
    memo = config.settings.memo
    api_key = os.environ.get(memo.api_key_env, "")
    if memo.base_url and memo.model and api_key:
        return OpenAICompatClient(memo.base_url, api_key, memo.model,
                                  memo.timeout_seconds)
    return OllamaClient(config.settings.llm)


def _client_model_name(client) -> str:
    model = getattr(client, "model", None)  # OpenAICompatClient
    if model:
        return model
    settings = getattr(client, "settings", None)  # OllamaClient
    return getattr(settings, "model", None) or "unknown"


def gather_context(conn: sqlite3.Connection, order: sqlite3.Row) -> str:
    """Compact, factual context doc from the run's audit trail."""
    notes = json.loads(order["notes"] or "{}")
    lines = [
        f"PROPOSED TRADE: {order['side'].upper()} {order['qty']:g} {order['ticker']} "
        f"(estimated value ${notes.get('est_value') or 'n/a'})",
        f"System reason: {notes.get('reason', 'n/a')}",
        "",
    ]

    decision = conn.execute(
        "SELECT * FROM decisions WHERE run_id = ? AND ticker = ?",
        (order["run_id"], order["ticker"]),
    ).fetchone()
    if decision:
        payload = json.loads(decision["signal_breakdown"] or "{}")
        lines.append(f"FINAL SCORE: {decision['final_score']} "
                     f"(signal coverage {payload.get('coverage', 'n/a')})")
        risk = payload.get("risk")
        if risk:
            lines.append(f"Risk engine: {'cleared' if risk.get('approved') else 'BLOCKED'}")
        portfolio = payload.get("portfolio")
        if portfolio and portfolio.get("checks"):
            lines.append("Sizing: " + "; ".join(portfolio["checks"][:3]))
        lines.append("")

    scores = conn.execute(
        "SELECT agent, score, confidence, direction, details FROM agent_scores "
        "WHERE run_id = ? AND ticker = ? ORDER BY agent",
        (order["run_id"], order["ticker"]),
    ).fetchall()
    for s in scores:
        details = json.loads(s["details"] or "{}")
        lines.append(f"AGENT {s['agent']}: score {s['score']} "
                     f"confidence {s['confidence']} ({s['direction']})")
        if s["confidence"] == 0:
            lines.append(f"  dead signal: {details.get('fallback_reason', 'no data')}")
            continue
        if s["agent"] == "technical":
            lines.append(
                f"  trend {details.get('trend')}, momentum 63d "
                f"{details.get('momentum_63d')}, vol {details.get('volatility_bucket')}, "
                f"drawdown {details.get('current_drawdown')}")
        elif s["agent"] == "fundamentals":
            lines.append(
                f"  growth {details.get('growth_score')}, profitability "
                f"{details.get('profitability_score')}, balance sheet "
                f"{details.get('balance_sheet_score')}, cash flow "
                f"{details.get('cash_flow_score')}, valuation "
                f"{details.get('valuation_score')}")
        elif s["agent"] == "kronos":
            lines.append(
                f"  10d expected return {details.get('expected_return_pct')}%, "
                f"p_up {details.get('p_up')}, band "
                f"[{details.get('forecast_p10_pct')}%, {details.get('forecast_p90_pct')}%]")
        elif s["agent"] == "news":
            for driver in (details.get("key_drivers") or [])[:3]:
                lines.append(f"  headline: {driver.get('headline', '')[:90]}")
        elif s["agent"] == "sec_filing":
            for finding in (details.get("findings") or [])[:3]:
                lines.append(f"  filing risk [{finding.get('severity')}] "
                             f"{finding.get('category')}: {finding.get('quote', '')[:80]}")
    lines.append("")

    macro = conn.execute(
        "SELECT detail FROM run_stages WHERE run_id = ? AND stage = 'macro'",
        (order["run_id"],),
    ).fetchone()
    if macro:
        lines.append(f"MACRO REGIME: {macro['detail']}")

    position = conn.execute(
        "SELECT qty, avg_cost, opened_at FROM positions WHERE ticker = ?",
        (order["ticker"],),
    ).fetchone()
    if position:
        lines.append(f"CURRENT POSITION: {position['qty']:g} shares @ "
                     f"{position['avg_cost']:.2f} since {position['opened_at'][:10]}")

    return "\n".join(lines)[:6000]


def render_markdown(memo: ResearchMemo, order: sqlite3.Row, model: str) -> str:
    lines = [
        f"# Research Memo — {order['side'].upper()} {order['ticker']}",
        "",
        f"_Advisory only. Generated by {model}; grounded in the run's audit trail._",
        "",
        f"**Thesis.** {memo.thesis}",
        "",
        "## Bull case",
        *[f"- {p}" for p in memo.bull_case],
        "",
        "## Bear case",
        *[f"- {p}" for p in memo.bear_case],
        "",
        f"**Variant perception.** {memo.variant_perception}",
        "",
        "## Kill criteria",
        *[f"- {p}" for p in memo.kill_criteria],
        "",
        f"**Recommendation: {memo.approval_recommendation}** — {memo.confidence_note}",
        "",
    ]
    return "\n".join(lines)


def generate_memo(
    conn: sqlite3.Connection, config: AppConfig, order: sqlite3.Row, client=None
) -> str | None:
    """Generate + persist a memo for one pending order. Returns the markdown,
    or None on any failure — memos never block the queue."""
    existing = conn.execute(
        "SELECT memo_md FROM research_memos WHERE order_id = ?", (order["order_id"],)
    ).fetchone()
    if existing:
        return existing["memo_md"]

    try:
        client = client or build_memo_client(config)
        context = gather_context(conn, order)
        memo = client.generate(context, ResearchMemo, system=SYSTEM_PROMPT)
    except LLMError as exc:
        logger.warning("memo generation failed for %s: %s", order["order_id"], exc)
        return None

    model = _client_model_name(client)
    markdown = render_markdown(memo, order, model)
    conn.execute(
        "INSERT INTO research_memos (order_id, ticker, run_id, created_at, model, "
        "recommendation, memo_md) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (order["order_id"], order["ticker"], order["run_id"],
         utcnow().isoformat(), model, memo.approval_recommendation, markdown),
    )
    conn.commit()

    memo_dir = config.reports_dir / "memos"
    memo_dir.mkdir(parents=True, exist_ok=True)
    (memo_dir / f"{order['order_id']}-{order['ticker']}.md").write_text(
        markdown, encoding="utf-8")
    return markdown
