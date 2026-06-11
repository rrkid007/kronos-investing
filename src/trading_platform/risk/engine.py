"""Risk Engine — the deterministic safety layer. Pure Python. No AI.

Independently re-derives every constraint from raw inputs (decision, sizing,
account state) rather than trusting upstream agents — defense in depth: a bug
in the Portfolio Agent's sizing math must not be able to place an oversized
order.

Buy rules (all must pass):
    sane_inputs          action is buy, qty >= 1, price > 0
    min_final_score      decision.final_score >= min_final_score
    min_agent_scores     each configured agent floor (live agents only —
                         a dead/absent agent is not blocking)
    not_restricted       ticker not in restricted_assets
    position_limit       cost <= equity x max_position_pct
    sector_limit         sector exposure incl. this buy <= equity x max_sector_pct
    cash_sufficient      cost <= cash
    cash_reserve         cash - cost >= equity x min_cash_reserve_pct

Sell rules: exits must never be blocked by exposure rules (blocking a
stop-loss is itself a risk). Only sanity is checked:
    sane_inputs          action is sell, price > 0
    position_exists      a live position backs the sell
    qty_within_position  sell qty <= held qty

Every evaluation is logged to risk_events. requires_human_approval is
attached from config for the order layer (phase 8).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from trading_platform.agents.portfolio import PortfolioState
from trading_platform.core.config import RiskLimits, Watchlist
from trading_platform.core.models import Action, Position, TradeDecision

EPSILON = 1e-6  # float tolerance on boundary comparisons


class RiskCheck(BaseModel):
    rule: str
    passed: bool
    detail: str = ""


class RiskResult(BaseModel):
    ticker: str
    side: str
    approved: bool
    requires_human_approval: bool
    checks: list[RiskCheck] = Field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [f"{c.rule}: {c.detail}" for c in self.checks if not c.passed]


class RiskEngine:
    def __init__(self, risk: RiskLimits, watchlist: Watchlist):
        self.risk = risk
        self.watchlist = watchlist

    def evaluate_buy(
        self,
        decision: TradeDecision,
        qty: int,
        price: float,
        state: PortfolioState,
    ) -> RiskResult:
        checks: list[RiskCheck] = []

        def check(rule: str, passed: bool, detail: str) -> None:
            checks.append(RiskCheck(rule=rule, passed=passed, detail=detail))

        sane = decision.action == Action.BUY and qty >= 1 and price is not None and price > 0
        check("sane_inputs", sane,
              f"action={decision.action.value} qty={qty} price={price}")
        if not sane:
            return self._result(decision.ticker, "buy", checks)

        cost = qty * price

        check(
            "min_final_score",
            decision.final_score >= self.risk.min_final_score - EPSILON,
            f"final score {decision.final_score} vs minimum {self.risk.min_final_score}",
        )

        signals = decision.signal_breakdown.get("signals", {})
        for agent, floor in self.risk.min_agent_scores.items():
            sig = signals.get(agent) or {}
            score, confidence = sig.get("score"), sig.get("confidence", 0)
            if score is None or not confidence or confidence <= 0:
                check(f"min_agent_score:{agent}", True,
                      f"{agent} dead or absent — not blocking")
                continue
            check(
                f"min_agent_score:{agent}",
                score >= floor - EPSILON,
                f"{agent} score {score} vs floor {floor}",
            )

        check(
            "not_restricted",
            decision.ticker not in self.risk.restricted_assets,
            f"{decision.ticker} restricted" if decision.ticker in self.risk.restricted_assets
            else "not on restricted list",
        )

        position_cap = state.equity * self.risk.max_position_pct / 100
        check(
            "position_limit",
            cost <= position_cap + EPSILON,
            f"cost {cost:.2f} vs cap {position_cap:.2f} "
            f"({self.risk.max_position_pct}% of equity)",
        )

        sector = self.watchlist.sector_of(decision.ticker) or "Unknown"
        sector_cap = state.equity * self.risk.max_sector_pct / 100
        sector_after = state.sector_values.get(sector, 0.0) + cost
        check(
            "sector_limit",
            sector_after <= sector_cap + EPSILON,
            f"{sector} exposure after buy {sector_after:.2f} vs cap {sector_cap:.2f}",
        )

        check(
            "cash_sufficient",
            cost <= state.cash + EPSILON,
            f"cost {cost:.2f} vs cash {state.cash:.2f}",
        )

        reserve = state.equity * self.risk.min_cash_reserve_pct / 100
        check(
            "cash_reserve",
            state.cash - cost >= reserve - EPSILON,
            f"cash after buy {state.cash - cost:.2f} vs reserve floor {reserve:.2f}",
        )

        return self._result(decision.ticker, "buy", checks)

    def evaluate_sell(
        self,
        decision: TradeDecision,
        position: Position | None,
        price: float | None,
    ) -> RiskResult:
        checks: list[RiskCheck] = []

        def check(rule: str, passed: bool, detail: str) -> None:
            checks.append(RiskCheck(rule=rule, passed=passed, detail=detail))

        check("sane_inputs", decision.action == Action.SELL and price is not None and price > 0,
              f"action={decision.action.value} price={price}")
        check("position_exists", position is not None,
              "live position found" if position else "no position to sell")
        if position is not None:
            qty = decision.sizing_hint or 0
            check("qty_within_position", 0 < qty <= position.qty + EPSILON,
                  f"sell qty {qty} vs held {position.qty}")
        return self._result(decision.ticker, "sell", checks)

    def _result(self, ticker: str, side: str, checks: list[RiskCheck]) -> RiskResult:
        return RiskResult(
            ticker=ticker,
            side=side,
            approved=all(c.passed for c in checks),
            requires_human_approval=self.risk.require_human_approval,
            checks=checks,
        )


def log_risk_event(
    conn: sqlite3.Connection, run_id: str, result: RiskResult
) -> None:
    conn.execute(
        "INSERT INTO risk_events (run_id, ticker, approved, violations, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            run_id,
            result.ticker,
            int(result.approved),
            json.dumps(result.errors),
            datetime.now(tz=timezone.utc).isoformat(),
        ),
    )
    conn.commit()
