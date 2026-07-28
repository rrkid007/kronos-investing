"""Paid Kronos forecast service — x402-metered HTTP API.

Additive subpackage: nothing here is imported by the pipeline, the dashboard,
or any agent. Deleting this directory leaves the platform unchanged.

The service sells one thing: a Kronos forecast score over candles the caller
supplies. It deliberately does NOT fetch market data (see README.md).
"""

from trading_platform.service.app import create_service_app

__all__ = ["create_service_app"]
