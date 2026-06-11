"""Alpaca PAPER trading client. Live trading is structurally impossible here:
the base URL is hard-coded to the paper endpoint and not configurable.

Only four endpoints are used, so this is a thin requests wrapper rather than
the full alpaca-py SDK. Orders are submitted as market-on-open (tif=opg),
matching the platform's decide-at-close, fill-at-next-open convention, with
our order_id as client_order_id so resubmission is idempotent at the broker.
"""

from __future__ import annotations

import logging
import os

import requests

from trading_platform.core.config import ExecutionSettings

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"  # never configurable


class AlpacaError(RuntimeError):
    pass


def alpaca_symbol(ticker: str) -> str:
    """Class shares: our 'BRK-B' is Alpaca's 'BRK.B'."""
    return ticker.replace("-", ".")


class AlpacaPaperBroker:
    def __init__(self, key_id: str, secret: str):
        if not key_id or not secret:
            raise AlpacaError("Alpaca paper API keys are not set")
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
        }

    @classmethod
    def from_env(cls, settings: ExecutionSettings) -> "AlpacaPaperBroker":
        return cls(
            os.environ.get(settings.alpaca_key_env, ""),
            os.environ.get(settings.alpaca_secret_env, ""),
        )

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict | list:
        try:
            resp = requests.request(
                method, f"{PAPER_BASE_URL}{path}",
                headers=self._headers, json=payload, timeout=20,
            )
        except requests.RequestException as exc:
            raise AlpacaError(f"alpaca unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise AlpacaError(f"alpaca {method} {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def submit_market_on_open(
        self, ticker: str, side: str, qty: float, client_order_id: str
    ) -> dict:
        return self._request("POST", "/v2/orders", {
            "symbol": alpaca_symbol(ticker),
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "opg",  # fills at next market open
            "client_order_id": client_order_id,
        })

    def get_order(self, broker_order_id: str) -> dict:
        return self._request("GET", f"/v2/orders/{broker_order_id}")

    def get_positions(self) -> list[dict]:
        return self._request("GET", "/v2/positions")

    def get_account(self) -> dict:
        return self._request("GET", "/v2/account")
