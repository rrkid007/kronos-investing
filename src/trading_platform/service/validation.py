"""Input gate for caller-supplied candles, plus the receipt hash.

Why not reuse `data.quality.validate_ohlcv`? That gate is correct for the
internal equity pipeline and wrong for a public API on four counts:

  - it requires an `adj_close` column (crypto, FX and futures callers have none)
  - it fails bars older than `max_staleness_bdays` against *today* — but a
    backtesting customer scoring a 2021 window is doing so deliberately
  - its gap check counts business days, which rejects 24/7 crypto bars
  - `min_rows=250` is sized for the technical agent's 200-day MA, not Kronos

So this gate keeps only the checks that mean "these candles are internally
incoherent" — the ones where a forecast would be garbage no matter the asset.
Ordering matters: this runs *before* payment verification, so malformed input
costs the caller nothing and costs us no GPU.
"""

from __future__ import annotations

import hashlib
import math

import pandas as pd

# Rounding applied before hashing so float repr drift between clients doesn't
# produce a different receipt for identical candles.
HASH_PRECISION = 6


def validate_candles(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty means the frame is scoreable.

    Positivity, row count and time-ordering are already enforced by the
    pydantic schema — these are the cross-field checks it cannot express.
    """
    problems: list[str] = []

    values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not all(math.isfinite(v) for v in values.ravel()):
        problems.append("candles contain NaN or infinite values")
        return problems  # every check below would be meaningless

    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    eps = 1e-9

    bad_low = int((df["low"] > body_low + eps).sum())
    if bad_low:
        problems.append(f"{bad_low} candle(s) have low above open/close")

    bad_high = int((df["high"] < body_high - eps).sum())
    if bad_high:
        problems.append(f"{bad_high} candle(s) have high below open/close")

    inverted = int((df["low"] > df["high"] + eps).sum())
    if inverted:
        problems.append(f"{inverted} candle(s) have low above high")

    # A dead series has no forecastable structure and would burn GPU for nothing.
    if float(df["close"].std()) == 0.0:
        problems.append("close price is constant across the whole window")

    return problems


def input_hash(df: pd.DataFrame, *, tier: str, model_id: str, horizon: int) -> str:
    """Stable digest of everything that determines the answer.

    Covers the candles and the priced parameters, so a receipt can be checked
    independently by the buyer. Excludes the seed — that is recorded separately
    on the receipt, letting a buyer see that two differing answers came from the
    same input under different sampling.
    """
    h = hashlib.sha256()
    h.update(f"{tier}|{model_id}|{horizon}|".encode())
    for ts, row in zip(df.index, df.itertuples(index=False)):
        h.update(pd.Timestamp(ts).isoformat().encode())
        for value in row:
            h.update(f"|{float(value):.{HASH_PRECISION}f}".encode())
        h.update(b"\n")
    return h.hexdigest()
