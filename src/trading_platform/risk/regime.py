"""Macro regime classification — deterministic, no AI (it sits on the
control side of the house, with the Risk Engine).

Each available indicator contributes 0 (calm) / 1 (elevated) / 2 (stress)
points against config thresholds. The stress ratio (points / max possible
for the indicators actually available) maps to a regime:

    ratio <= 0.25  calm      x1.00 sizing
    ratio <= 0.60  caution   x0.75
    else           stress    x0.50

Fewer than two live indicators -> "unknown", neutral x1.0 scaling — a blind
macro read must not move position sizes. Only NEW position sizing is scaled;
exits are never blocked or shrunk by regime.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from trading_platform.core.config import MacroSettings
from trading_platform.data.macro import MacroSnapshot


class RegimeComponent(BaseModel):
    indicator: str
    value: float
    points: int
    detail: str


class RegimeAssessment(BaseModel):
    regime: Literal["calm", "caution", "stress", "unknown"]
    stress_score: int = 0
    max_score: int = 0
    sizing_scalar: float = 1.0
    components: list[RegimeComponent] = Field(default_factory=list)
    as_of: dict[str, str] = Field(default_factory=dict)

    def summary(self) -> str:
        if self.regime == "unknown":
            return "unknown (insufficient macro data) — sizing unscaled"
        return (f"{self.regime} (stress {self.stress_score}/{self.max_score}) "
                f"— sizing x{self.sizing_scalar}")


def classify_regime(snapshot: MacroSnapshot, settings: MacroSettings) -> RegimeAssessment:
    components: list[RegimeComponent] = []

    if snapshot.vix is not None:
        points = (2 if snapshot.vix >= settings.vix_stress
                  else 1 if snapshot.vix >= settings.vix_elevated else 0)
        components.append(RegimeComponent(
            indicator="vix", value=snapshot.vix, points=points,
            detail=f"VIX {snapshot.vix:.1f} "
                   f"(elevated >= {settings.vix_elevated}, stress >= {settings.vix_stress})",
        ))
    if snapshot.hy_oas is not None:
        points = (2 if snapshot.hy_oas >= settings.hy_oas_stress
                  else 1 if snapshot.hy_oas >= settings.hy_oas_elevated else 0)
        components.append(RegimeComponent(
            indicator="hy_oas", value=snapshot.hy_oas, points=points,
            detail=f"HY OAS {snapshot.hy_oas:.2f}% "
                   f"(elevated >= {settings.hy_oas_elevated}, stress >= {settings.hy_oas_stress})",
        ))
    if snapshot.yield_curve is not None:
        points = (2 if snapshot.yield_curve < settings.curve_inverted
                  else 1 if snapshot.yield_curve < settings.curve_flat else 0)
        components.append(RegimeComponent(
            indicator="yield_curve", value=snapshot.yield_curve, points=points,
            detail=f"10Y-2Y {snapshot.yield_curve:+.2f}% "
                   f"(flat < {settings.curve_flat}, inverted < {settings.curve_inverted})",
        ))

    if len(components) < 2:
        return RegimeAssessment(
            regime="unknown",
            sizing_scalar=settings.sizing_scalars.get("unknown", 1.0),
            components=components, as_of=snapshot.as_of,
        )

    score = sum(c.points for c in components)
    max_score = 2 * len(components)
    ratio = score / max_score
    regime = "calm" if ratio <= 0.25 else "caution" if ratio <= 0.60 else "stress"
    return RegimeAssessment(
        regime=regime,
        stress_score=score,
        max_score=max_score,
        sizing_scalar=settings.sizing_scalars.get(regime, 1.0),
        components=components,
        as_of=snapshot.as_of,
    )
