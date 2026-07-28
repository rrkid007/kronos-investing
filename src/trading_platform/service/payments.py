"""x402 binding — the only module that imports the x402 SDK.

Isolated on purpose. The SDK is young and its surface has already moved once
(flat network names -> CAIP-2 ids, `require_payment` decorator -> ASGI
middleware). When it moves again, this file is the blast radius; the rest of
the service is plain FastAPI and stays testable without payments installed.

Protocol flow, and why the handler ordering elsewhere matters:

    1. caller GETs/POSTs with no X-PAYMENT header
    2. middleware returns 402 + payment requirements (price, chain, pay_to)
    3. caller signs an EIP-3009 transferWithAuthorization, retries with the
       X-PAYMENT header
    4. middleware *verifies* with the facilitator  <- cheap, pre-compute
    5. our handler runs the forecast              <- the expensive part
    6. middleware *settles* only on a 2xx         <- so failures cost nothing
       and returns the tx hash in X-PAYMENT-RESPONSE

We never hold a private key: settlement pays `pay_to` directly. The server
holds an address, not a wallet.
"""

from __future__ import annotations

import logging

from trading_platform.service.config import ServiceConfig
from trading_platform.service.tiers import Tier

logger = logging.getLogger(__name__)

INSTALL_HINT = 'x402 not installed — run: uv sync --extra service'


class PaymentsUnavailable(RuntimeError):
    """Payments were demanded by config but the SDK/middleware could not load."""


def attach_payments(app, config: ServiceConfig, tiers: list[Tier]) -> bool:
    """Wire the x402 paywall onto `app`. Returns True if the paywall is active.

    No-op returning False when payments are disabled, so `create_service_app()`
    yields a fully functional free service in dev and in tests.
    """
    if not config.payments_enabled:
        logger.warning(
            "x402 payments DISABLED — every endpoint is free. "
            "Set X402_ENABLED=1 and X402_PAY_TO=0x... to meter."
        )
        return False

    problems = config.problems()
    if problems:
        raise PaymentsUnavailable("; ".join(problems))

    # Verified against x402 2.17.0. Note `x402[fastapi]` alone is not enough —
    # the EVM mechanism lives behind the `evm` marker and raises on import
    # without it, hence "x402[fastapi,evm]" in pyproject.
    try:
        from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
        from x402.http.middleware.fastapi import PaymentMiddlewareASGI
        from x402.http.types import RouteConfig
        from x402.mechanisms.evm.exact import ExactEvmServerScheme
        from x402.server import x402ResourceServer
    except ImportError as exc:
        raise PaymentsUnavailable(f"{INSTALL_HINT} ({exc})") from exc

    facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=config.facilitator_url))
    server = x402ResourceServer(facilitator_clients=facilitator)
    server.register(config.network, ExactEvmServerScheme())
    server.initialize()

    # One PaymentOption per tier: same scheme and payee, different price.
    routes = {
        f"POST {tier.path}": RouteConfig(
            accepts=PaymentOption(
                scheme="exact",
                pay_to=config.pay_to,
                price=tier.price_usd,
                network=config.network,
                max_timeout_seconds=config.payment_timeout_seconds,
            ),
            mime_type="application/json",
            description=tier.description[:500],  # protocol caps description length
        )
        for tier in tiers
    }

    app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)

    logger.info(
        "x402 paywall active: %s on %s via %s -> %s",
        ", ".join(f"{t.path} {t.price_usd}" for t in tiers),
        config.network,
        config.facilitator_url,
        config.pay_to,
    )
    if config.is_testnet:
        logger.warning("x402 network %s is a TESTNET — payments are not real", config.network)
    return True
