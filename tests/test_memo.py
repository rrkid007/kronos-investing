"""Research memos — client routing, context grounding, persistence, advisory-only."""

import json
from pathlib import Path

import pytest
import requests

from tests.fixtures import FakeLLM
from trading_platform.core.db import connect, init_db
from trading_platform.core.llm import (
    LLMError,
    OllamaClient,
    OpenAICompatClient,
    _strip_fences,
)
from trading_platform.research.memo import (
    ResearchMemo,
    build_memo_client,
    gather_context,
    generate_memo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def canned_memo(recommendation="approve"):
    return ResearchMemo(
        thesis="The system proposes buying AAPL on a strong blended signal.",
        bull_case=["technical uptrend", "fundamentals quality", "macro calm"],
        bear_case=["rich valuation", "kronos forecast neutral", "sector crowding"],
        variant_perception="The blend weighs fundamentals the market is discounting.",
        kill_criteria=["price closes below $250", "final score drops under 45",
                       "stop loss at -8% from entry"],
        approval_recommendation=recommendation,
        confidence_note="Four of five signals live; coverage 0.78.",
    )


@pytest.fixture
def seeded(tmp_config):
    """A run with one pending buy order plus its full audit trail."""
    conn = connect(tmp_config.db_path)
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('r1', '2026-06-11', '2026-06-11T22:00:00', 'completed')"
    )
    conn.execute(
        "INSERT INTO run_stages (run_id, stage, ticker, status, detail, updated_at) "
        "VALUES ('r1', 'macro', '', 'completed', 'caution (stress 2/6) — sizing x0.75', '')"
    )
    conn.execute(
        "INSERT INTO agent_scores (run_id, agent, ticker, score, confidence, direction, "
        "details, created_at) VALUES ('r1', 'technical', 'AAPL', 81.0, 0.9, 'bullish', ?, '')",
        (json.dumps({"trend": "uptrend", "momentum_63d": 0.12,
                     "volatility_bucket": "moderate", "current_drawdown": -0.02}),),
    )
    conn.execute(
        "INSERT INTO agent_scores (run_id, agent, ticker, score, confidence, direction, "
        "details, created_at) VALUES ('r1', 'news', 'AAPL', 50.0, 0.0, 'neutral', ?, '')",
        (json.dumps({"fallback_reason": "ollama unreachable"}),),
    )
    conn.execute(
        "INSERT INTO decisions (run_id, ticker, action, final_score, signal_breakdown, "
        "sizing_hint, reason, created_at) VALUES ('r1', 'AAPL', 'buy', 78.5, ?, 9500, "
        "'final score 78.5 >= buy threshold 70.0', '')",
        (json.dumps({"coverage": 0.78,
                     "risk": {"approved": True, "checks": []},
                     "portfolio": {"approved": True, "qty": 25,
                                   "checks": ["base 10000 x ..."]}}),),
    )
    conn.execute(
        "INSERT INTO orders (order_id, run_id, ticker, side, qty, status, created_at, notes) "
        "VALUES ('ord-1', 'r1', 'AAPL', 'buy', 25, 'awaiting_approval', '', ?)",
        (json.dumps({"final_score": 78.5, "est_value": 9500.0,
                     "reason": "buy signal"}),),
    )
    conn.commit()
    yield conn, tmp_config
    conn.close()


def get_order(conn):
    return conn.execute("SELECT * FROM orders WHERE order_id = 'ord-1'").fetchone()


# --- OpenAI-compatible client ---------------------------------------------------

class FakeResp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_openai_compat_payload_and_parse(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, payload=json)
        return FakeResp(canned_memo().model_dump_json())

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenAICompatClient("https://api.example.com/v1", "sk-test",
                                "frontier-model")
    result = client.generate("context here", ResearchMemo, system="be an analyst")

    assert result.approval_recommendation == "approve"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "frontier-model"
    assert "JSON schema" in captured["payload"]["messages"][0]["content"]


def test_openai_compat_strips_markdown_fences(monkeypatch):
    fenced = "```json\n" + canned_memo().model_dump_json() + "\n```"
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(fenced))
    client = OpenAICompatClient("https://x/v1", "k", "m")
    assert client.generate("p", ResearchMemo).thesis.startswith("The system")


def test_strip_fences_handles_plain_json():
    payload = '{"a": 1}'
    assert _strip_fences(payload) == payload


def test_openai_compat_requires_config():
    with pytest.raises(LLMError, match="base_url and model"):
        OpenAICompatClient("", "key", "")


def test_openai_compat_unreachable(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    client = OpenAICompatClient("https://x/v1", "k", "m")
    with pytest.raises(LLMError, match="unreachable"):
        client.generate("p", ResearchMemo)


# --- client routing ----------------------------------------------------------------

def test_factory_falls_back_to_ollama_without_key(tmp_config, monkeypatch):
    monkeypatch.delenv("MEMO_LLM_API_KEY", raising=False)
    assert isinstance(build_memo_client(tmp_config), OllamaClient)


def test_factory_uses_external_when_configured(tmp_config, monkeypatch):
    monkeypatch.setenv("MEMO_LLM_API_KEY", "sk-test")
    config = tmp_config.model_copy(deep=True)
    config.settings.memo.base_url = "https://api.example.com/v1"
    config.settings.memo.model = "frontier-model"
    client = build_memo_client(config)
    assert isinstance(client, OpenAICompatClient)
    assert client.model == "frontier-model"


# --- context grounding ----------------------------------------------------------------

def test_context_carries_audit_trail(seeded):
    conn, _ = seeded
    context = gather_context(conn, get_order(conn))
    assert "BUY 25 AAPL" in context
    assert "FINAL SCORE: 78.5" in context
    assert "coverage 0.78" in context
    assert "trend uptrend" in context
    assert "dead signal: ollama unreachable" in context  # honesty about dead agents
    assert "MACRO REGIME: caution" in context
    assert "Risk engine: cleared" in context


# --- generation + persistence -----------------------------------------------------------

def test_memo_generated_persisted_and_rendered(seeded):
    conn, config = seeded
    markdown = generate_memo(conn, config, get_order(conn),
                             client=FakeLLM(response=canned_memo()))
    assert markdown is not None
    for fragment in ("# Research Memo — BUY AAPL", "## Bull case", "## Bear case",
                     "## Kill criteria", "Advisory only",
                     "Recommendation: approve"):
        assert fragment in markdown, fragment

    row = conn.execute("SELECT * FROM research_memos WHERE order_id = 'ord-1'").fetchone()
    assert row["recommendation"] == "approve"
    assert (config.reports_dir / "memos" / "ord-1-AAPL.md").exists()


def test_memo_idempotent_one_llm_call(seeded):
    conn, config = seeded
    llm = FakeLLM(response=canned_memo())
    first = generate_memo(conn, config, get_order(conn), client=llm)
    second = generate_memo(conn, config, get_order(conn), client=llm)
    assert llm.calls == 1  # cached on second call
    assert first == second


def test_memo_failure_returns_none_never_blocks(seeded):
    conn, config = seeded
    result = generate_memo(conn, config, get_order(conn),
                           client=FakeLLM(fail=LLMError("no model")))
    assert result is None
    assert get_order(conn)["status"] == "awaiting_approval"  # queue untouched
    assert conn.execute("SELECT COUNT(*) FROM research_memos").fetchone()[0] == 0
