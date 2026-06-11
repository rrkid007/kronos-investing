"""Run notifications (PLAN.md A5) — a silent cron failure means days of
missing data, so every run reports success or failure.

ntfy-compatible: plain POST with a Title header works with ntfy.sh, a
self-hosted ntfy, or any webhook that accepts raw text. Notification failures
are logged and swallowed — alerting must never break the run it reports on.
"""

from __future__ import annotations

import logging

import requests

from trading_platform.core.config import NotificationSettings

logger = logging.getLogger(__name__)


def notify(settings: NotificationSettings, title: str, message: str) -> bool:
    """Send a notification. Returns True only if actually delivered."""
    if not settings.enabled or not settings.webhook_url:
        return False
    try:
        resp = requests.post(
            settings.webhook_url,
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("notification failed: %s", exc)
        return False
