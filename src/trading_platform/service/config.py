"""Service configuration — environment only, never the platform YAML.

Deliberately separate from `core.config.AppConfig`: the paid service is a
different deployment with a different blast radius, and the dashboard's
Settings page (which writes config YAML) must never be able to reach the
payment address. Changing where money lands requires shell access to the box.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

# CAIP-2 chain ids. Base is the x402 default; the others are supported by the
# CDP facilitator.
BASE_MAINNET = "eip155:8453"
BASE_SEPOLIA = "eip155:84532"

TESTNET_FACILITATOR = "https://x402.org/facilitator"
CDP_FACILITATOR = "https://api.cdp.coinbase.com/platform/v2/x402"

TESTNETS = {BASE_SEPOLIA}


class ServiceConfig(BaseModel):
    # Payments are opt-in so the service runs unpaid in dev and in tests.
    payments_enabled: bool = False
    pay_to: str | None = None
    network: str = BASE_SEPOLIA
    facilitator_url: str = TESTNET_FACILITATOR

    max_queue: int = 8
    # Must stay below the payment authorisation's maxTimeoutSeconds, or we
    # accept authorisations that expire before we can settle them.
    timeout_seconds: float = 60.0
    cache_size: int = 256

    host: str = "127.0.0.1"
    port: int = 8402

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ServiceConfig":
        e = env if env is not None else os.environ

        def flag(name: str, default: bool) -> bool:
            raw = e.get(name)
            return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            payments_enabled=flag("X402_ENABLED", False),
            pay_to=e.get("X402_PAY_TO") or None,
            network=e.get("X402_NETWORK") or BASE_SEPOLIA,
            facilitator_url=e.get("X402_FACILITATOR_URL") or TESTNET_FACILITATOR,
            max_queue=int(e.get("KRONOS_SERVICE_MAX_QUEUE") or 8),
            timeout_seconds=float(e.get("KRONOS_SERVICE_TIMEOUT") or 60.0),
            cache_size=int(e.get("KRONOS_SERVICE_CACHE_SIZE") or 256),
            host=e.get("KRONOS_SERVICE_HOST") or "127.0.0.1",
            port=int(e.get("KRONOS_SERVICE_PORT") or 8402),
        )

    @property
    def is_testnet(self) -> bool:
        return self.network in TESTNETS

    @property
    def payment_timeout_seconds(self) -> int:
        """Validity window advertised on the payment requirements.

        Must exceed our own compute deadline, or we hand out authorisations
        that expire before the forecast finishes and settlement fails after the
        GPU work is already spent. The margin covers queueing and settlement
        round-trips.
        """
        return int(self.timeout_seconds) + 30

    def problems(self) -> list[str]:
        """Misconfigurations that should stop startup rather than silently
        serve forecasts nobody pays for (or pays to the wrong chain)."""
        if not self.payments_enabled:
            return []

        found: list[str] = []
        if not self.pay_to:
            found.append("X402_ENABLED is set but X402_PAY_TO is empty")
        elif self.network.startswith("eip155:") and not (
            self.pay_to.startswith("0x") and len(self.pay_to) == 42
        ):
            found.append(f"X402_PAY_TO {self.pay_to!r} is not a valid EVM address")

        if not self.is_testnet and self.facilitator_url == TESTNET_FACILITATOR:
            found.append(
                f"network {self.network} is mainnet but facilitator is the testnet "
                f"one ({TESTNET_FACILITATOR}); use {CDP_FACILITATOR}"
            )
        return found
