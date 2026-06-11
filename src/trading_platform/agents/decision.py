"""Trade Decision Engine — confidence-weighted aggregation + exit policy.

Aggregation (PLAN.md S1):
    final_score = sum(w_i x c_i x s_i) / sum(w_i x c_i)

Zero-confidence signals (the neutral fallbacks agents emit when their data
source is down) drop out of the average entirely instead of dragging it
toward 50. Coverage = sum(w x c) / sum(w) measures how much weighted signal
actually backs the score; below thresholds.min_signal_coverage no new
position is opened and score-decay exits are suspended (price-based exits
still fire — they need no signals).

Exit policy for held positions (PLAN.md A2), first match wins:
    1. stop_loss    price <= avg_cost x (1 - stop_loss_pct/100)
    2. take_profit  price >= avg_cost x (1 + take_profit_pct/100)
    3. max_hold     calendar days held >= max_hold_days
    4. score_decay  final_score < exit_score (requires sufficient coverage)

No pyramiding: a held ticker that still scores buy-level stays 'hold'.
"""

from __future__ import annotations

from datetime import date

from trading_platform.core.config import Weights
from trading_platform.core.models import Action, Position, TradeDecision


def aggregate_signals(
    signals: dict[str, tuple[float, float]], weights: dict[str, float]
) -> tuple[float | None, float, dict]:
    """Confidence-weighted score. Returns (final_score|None, coverage, breakdown)."""
    numerator = 0.0
    denominator = 0.0
    breakdown: dict[str, dict] = {}
    for agent, weight in weights.items():
        score, confidence = signals.get(agent, (None, 0.0))
        effective = weight * confidence if score is not None else 0.0
        breakdown[agent] = {
            "score": score,
            "confidence": confidence,
            "weight": weight,
            "effective_weight": round(effective, 4),
        }
        if effective > 0:
            numerator += effective * score
            denominator += effective

    coverage = denominator / sum(weights.values()) if weights else 0.0
    final = round(numerator / denominator, 2) if denominator > 0 else None
    return final, round(coverage, 4), breakdown


class DecisionEngine:
    def __init__(self, weights: Weights):
        self.weights = weights

    def decide(
        self,
        ticker: str,
        run_id: str,
        signals: dict[str, tuple[float, float]],
        position: Position | None = None,
        current_price: float | None = None,
        today: date | None = None,
    ) -> TradeDecision:
        final, coverage, breakdown = aggregate_signals(
            signals, self.weights.signal_weights
        )
        t = self.weights.thresholds
        payload = {"signals": breakdown, "coverage": coverage}
        score_for_record = final if final is not None else 50.0
        sufficient = final is not None and coverage >= t.min_signal_coverage

        def decision(action: Action, reason: str, sizing: float | None = None):
            return TradeDecision(
                run_id=run_id, ticker=ticker, action=action,
                final_score=score_for_record, signal_breakdown=payload,
                sizing_hint=sizing, reason=reason,
            )

        if position is not None:
            exit_reason = self._check_exits(
                position, current_price, final if sufficient else None, today
            )
            if exit_reason:
                return decision(Action.SELL, exit_reason, sizing=position.qty)
            return decision(Action.HOLD, "holding; no exit condition met")

        if not sufficient:
            return decision(
                Action.HOLD,
                f"insufficient signal coverage ({coverage:.2f} < {t.min_signal_coverage})",
            )
        if final >= t.buy_score:
            return decision(Action.BUY, f"final score {final} >= buy threshold {t.buy_score}")
        if final >= t.watchlist_score:
            return decision(
                Action.WATCHLIST,
                f"final score {final} >= watchlist threshold {t.watchlist_score}",
            )
        return decision(Action.HOLD, f"final score {final} below watchlist threshold")

    def _check_exits(
        self,
        position: Position,
        current_price: float | None,
        final_score: float | None,
        today: date | None,
    ) -> str | None:
        policy = self.weights.exit_policy
        if current_price is not None and current_price > 0:
            stop = position.avg_cost * (1 - policy.stop_loss_pct / 100)
            target = position.avg_cost * (1 + policy.take_profit_pct / 100)
            if current_price <= stop:
                return (f"stop_loss: price {current_price:.2f} <= "
                        f"{stop:.2f} ({policy.stop_loss_pct}% below cost)")
            if current_price >= target:
                return (f"take_profit: price {current_price:.2f} >= "
                        f"{target:.2f} ({policy.take_profit_pct}% above cost)")
        if today is not None:
            held_days = (today - position.opened_at).days
            if held_days >= policy.max_hold_days:
                return f"max_hold: held {held_days} calendar days >= {policy.max_hold_days}"
        exit_score = self.weights.thresholds.exit_score
        if final_score is not None and final_score < exit_score:
            return f"score_decay: final score {final_score} < exit threshold {exit_score}"
        return None
