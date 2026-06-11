"""Notification hook — delivery, opt-out, and failure swallowing."""

import requests

from trading_platform.core.config import NotificationSettings
from trading_platform.core.notify import notify


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass


def test_sends_ntfy_style_post(monkeypatch):
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured.update(url=url, data=data, headers=headers)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    settings = NotificationSettings(enabled=True, webhook_url="https://ntfy.sh/topic")
    assert notify(settings, "Run OK", "equity $100,000") is True
    assert captured["url"] == "https://ntfy.sh/topic"
    assert captured["data"] == b"equity $100,000"
    assert captured["headers"]["Title"] == "Run OK"


def test_disabled_sends_nothing(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    assert notify(NotificationSettings(enabled=False, webhook_url="https://x"), "t", "m") is False
    assert notify(NotificationSettings(enabled=True, webhook_url=""), "t", "m") is False
    assert calls["n"] == 0


def test_delivery_failure_swallowed(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("ntfy down")

    monkeypatch.setattr(requests, "post", boom)
    settings = NotificationSettings(enabled=True, webhook_url="https://ntfy.sh/topic")
    assert notify(settings, "t", "m") is False  # returns False, never raises
