"""Portfolio Agent — sizing math, caps, trims, and rejections."""

from datetime import date
from pathlib import Path

import pytest

from tests.fixtures import make_ohlcv
from trading_platform.agents.portfolio import PortfolioAgent, PortfolioState, realized_vol
from trading_platform.core.config import load_config
from trading_platform.core.models import Action, Position, TradeDecision

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config")

agent = PortfolioAgent(CONFIG.risk, CONFIG.watchlist)
DF = make_ohlcv()  # default fixture: moderate vol


def buy_decision(ticker="AAPL", score=80.0):
    return TradeDecision(
        run_id="r1", ticker=ticker, action=Action.BUY, final_score=score
    )


def fresh_state(cash=100_000.0, positions=None, sector_values=None):
    positions = positions or []
    return PortfolioState(
        cash=cash,
        equity=cash + sum(p.qty * p.avg_cost for p in positions),
        positions=positions,
        sector_values=sector_values or {},
    )


def test_base_sizing_formula():
    a = agent.assess_buy(buy_decision(score=70.0), 100.0, DF, fresh_state())
    assert a.approved
    # base 10000 x score_scalar 0.75 x vol_scalar (0.25/vol clamped)
    vol = realized_vol(DF)
    expected = 10_000 * 0.75 * min(1.5, max(0.5, 0.25 / vol))
    assert a.target_value == pytest.approx(expected, rel=0.02)  # whole-share rounding


def test_higher_score_sizes_larger():
    # Higher-vol fixture keeps both sizes under the 15% cap so the pure
    # score-scalar ratio is observable.
    df = make_ohlcv(daily_vol=0.02)
    low = agent.assess_buy(buy_decision(score=70.0), 100.0, df, fresh_state())
    high = agent.assess_buy(buy_decision(score=100.0), 100.0, df, fresh_state())
    assert high.target_value > low.target_value
    assert high.target_value / low.target_value == pytest.approx(1.25 / 0.75, rel=0.02)


def test_single_position_cap_trims():
    # Score 100 + low vol would exceed 15% cap on a small price
    state = fresh_state(cash=100_000.0)
    calm = make_ohlcv(daily_vol=0.004)  # very low vol -> vol_scalar 1.5
    a = agent.assess_buy(buy_decision(score=100.0), 10.0, calm, state)
    assert a.approved
    assert a.target_value <= 15_000.0 + 10.0  # max_position_pct 15%
    assert any("max position cap" in c for c in a.checks)


def test_already_held_rejected():
    state = fresh_state(positions=[
        Position(ticker="AAPL", qty=10, avg_cost=100.0, opened_at=date(2026, 6, 1))
    ])
    a = agent.assess_buy(buy_decision("AAPL"), 100.0, DF, state)
    assert not a.approved
    assert "already held" in a.rejection


def test_max_positions_rejected():
    positions = [
        Position(ticker=f"T{i}", qty=1, avg_cost=100.0, opened_at=date(2026, 6, 1))
        for i in range(CONFIG.risk.sizing.max_positions)
    ]
    a = agent.assess_buy(buy_decision("AAPL"), 100.0, DF, fresh_state(positions=positions))
    assert not a.approved
    assert "max positions" in a.rejection


def test_sector_cap_trims_and_rejects():
    # Technology already at 35% of 100k equity; cap is 40% -> 5k room
    state = fresh_state(cash=65_000.0, sector_values={"Technology": 35_000.0})
    state.equity = 100_000.0
    a = agent.assess_buy(buy_decision("AAPL", score=100.0), 100.0, DF, state)
    assert a.approved
    assert a.target_value <= 5_000.0
    assert a.fit_score <= 85.0  # sector trim penalty

    # Sector full -> reject
    state2 = fresh_state(cash=60_000.0, sector_values={"Technology": 40_000.0})
    state2.equity = 100_000.0
    a2 = agent.assess_buy(buy_decision("AAPL"), 100.0, DF, state2)
    assert not a2.approved
    assert "sector Technology at cap" in a2.rejection


def test_cash_reserve_floor_trims_and_rejects():
    # equity 100k, reserve 10k. cash 12k -> only 2k available
    state = fresh_state(cash=12_000.0)
    state.equity = 100_000.0
    a = agent.assess_buy(buy_decision("AAPL", score=100.0), 100.0, DF, state)
    assert a.approved
    assert a.target_value <= 2_000.0
    assert any("available cash" in c for c in a.checks)

    # cash at the floor -> reject
    state2 = fresh_state(cash=10_500.0)
    state2.equity = 100_000.0
    a2 = agent.assess_buy(buy_decision("AAPL"), 100.0, DF, state2)
    assert not a2.approved
    assert "below minimum position" in a2.rejection


def test_below_minimum_size_rejected():
    # Price too high for the tiny target -> 0 whole shares
    state = fresh_state(cash=12_500.0)
    state.equity = 100_000.0
    a = agent.assess_buy(buy_decision("AAPL", score=70.0), 5_000.0, DF, state)
    assert not a.approved
    assert "below minimum" in a.rejection


def test_whole_share_quantity():
    a = agent.assess_buy(buy_decision(score=80.0), 333.0, DF, fresh_state())
    assert a.approved
    assert a.qty == int(a.target_value // 333.0) or a.target_value == a.qty * 333.0
    assert a.target_value == pytest.approx(a.qty * 333.0)


def test_greedy_state_consumption():
    """Two buys in the same sector: the second sees less room."""
    state = fresh_state(cash=100_000.0)
    first = agent.assess_buy(buy_decision("MSFT", score=100.0), 100.0, DF, state)
    assert first.approved
    state.apply_buy("MSFT", "Technology", first.target_value)

    assert "MSFT" in state.held_tickers
    second = agent.assess_buy(buy_decision("MSFT"), 100.0, DF, state)
    assert not second.approved  # now counts as held

    assert state.cash == pytest.approx(100_000.0 - first.target_value)
