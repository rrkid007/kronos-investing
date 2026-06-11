"""OllamaClient — schema validation, retry, and unreachable-server behavior."""

import json

import pytest
import requests
from pydantic import BaseModel

from trading_platform.core.config import LLMSettings
from trading_platform.core.llm import LLMError, OllamaClient


class Verdict(BaseModel):
    score: float
    label: str


class FakeResponse:
    def __init__(self, content: str, status: int = 200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return {"message": {"content": self._content}}


@pytest.fixture
def client():
    return OllamaClient(LLMSettings())


def test_valid_response_first_try(client, monkeypatch):
    payload = json.dumps({"score": 72.5, "label": "positive"})
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload))
    result = client.generate("prompt", Verdict)
    assert result.score == 72.5
    assert result.label == "positive"


def test_schema_passed_as_format(client, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["format"] = json["format"]
        return FakeResponse('{"score": 1, "label": "x"}')

    monkeypatch.setattr(requests, "post", fake_post)
    client.generate("prompt", Verdict)
    assert captured["format"]["properties"].keys() == {"score", "label"}


def test_invalid_then_valid_retries(client, monkeypatch):
    responses = iter([
        FakeResponse('{"score": "not a number and missing label"}'),
        FakeResponse('{"score": 50, "label": "ok"}'),
    ])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(responses))
    result = client.generate("prompt", Verdict)
    assert result.label == "ok"


def test_exhausted_retries_raise_llm_error(client, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse('{"bad": true}'))
    with pytest.raises(LLMError, match="after 3 attempts"):
        client.generate("prompt", Verdict, retries=2)


def test_unreachable_server_raises_immediately(client, monkeypatch):
    calls = {"n": 0}

    def fail(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(LLMError, match="unreachable"):
        client.generate("prompt", Verdict)
    assert calls["n"] == 1  # no pointless retries against a dead server
