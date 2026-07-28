"""Price tiers.

x402's `exact` scheme prices a *route*, not a request, so each tier is its own
path. The only thing a tier varies is `sample_count` — the number of stochastic
Kronos paths, which is the dominant GPU cost driver and the direct input to the
confidence calculation (path agreement needs >= 4 samples to mean anything).

Callers cannot set sample_count directly; if they could they would control the
server's GPU bill.
"""

from __future__ import annotations

from pydantic import BaseModel


class Tier(BaseModel):
    name: str
    path: str
    price_usd: str  # x402 wants the "$0.01" string form
    sample_count: int
    description: str


STANDARD = Tier(
    name="standard",
    path="/v1/kronos/score",
    price_usd="$0.01",
    sample_count=8,
    description=(
        "Kronos 10-day forecast score over caller-supplied OHLCV candles. "
        "8 stochastic sample paths."
    ),
)

DEEP = Tier(
    name="deep",
    path="/v1/kronos/score/deep",
    price_usd="$0.05",
    sample_count=32,
    description=(
        "Kronos 10-day forecast score over caller-supplied OHLCV candles. "
        "32 stochastic sample paths — tighter percentile bands and a more "
        "stable confidence estimate."
    ),
)

TIERS: dict[str, Tier] = {t.name: t for t in (STANDARD, DEEP)}
