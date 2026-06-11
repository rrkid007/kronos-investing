"""Decision Engine — aggregation math, thresholds, and exit policy."""

from datetime import date

import pytest

from trading_platform.agents.decision import DecisionEngine, aggregate_signals
from trading_platform.core.config import Weights
from trading_platform.core.models import Action, Position

WEIGHTS = Weights(
    signal_weights={
        "fundamentals": 0.35, "technical": 0.25, "kronos": 0.20,
        "news": 0.10, "sec_filing": 0.10,
    }
)
engine = DecisionEngine(WEIGHTS)
TODAY = date(2026, 6, 11)

FULL_CONF = {
    "fundamentals": (80.0, 1.0), "technical": (80.0, 1.0), "kronos": (80.0, 1.0),
    "news": (80.0, 1.0), "sec_filing": (80.0, 1.0),
}


def held(avg_cost=100.0, opened=date(2026, 6, 1), qty=10):
    return Position(ticker="TEST", qty=qty, avg_cost=avg_cost, opened_at=opened)


# --- aggregation -------------------------------------------------------------

def test_uniform_signals_aggregate_to_same_score():
    final, coverage, _ = aggregate_signals(FULL_CONF, WEIGHTS.signal_weights)
    assert final == 80.0
    assert coverage == 1.0


def test_weighted_average_golden_value():
    signals = {
        "fundamentals": (90.0, 1.0),  # 0.35 * 90
        "technical": (60.0, 1.0),     # 0.25 * 60
        "kronos": (50.0, 1.0),        # 0.20 * 50
        "news": (70.0, 1.0),          # 0.10 * 70
        "sec_filing": (80.0, 1.0),    # 0.10 * 80
    }
    final, coverage, _ = aggregate_signals(signals, WEIGHTS.signal_weights)
    assert final == pytest.approx(71.5)  # 31.5+15+10+7+8
    assert coverage == 1.0


def test_zero_confidence_signals_drop_out():
    signals = {
        "fundamentals": (90.0, 1.0),
        "technical": (10.0, 0.0),   # dead agent: must NOT drag the score
        "kronos": (90.0, 1.0),
    }
    final, coverage, breakdown = aggregate_signals(signals, WEIGHTS.signal_weights)
    assert final == 90.0
    assert coverage == pytest.approx(0.55)  # (0.35 + 0.20) / 1.0
    assert breakdown["technical"]["effective_weight"] == 0


def test_confidence_weights_tilt_the_average():
    signals = {"fundamentals": (90.0, 1.0), "technical": (40.0, 0.2)}
    final, _, _ = aggregate_signals(signals, WEIGHTS.signal_weights)
    # 0.35*90 + 0.05*40 over 0.40 -> 83.75
    assert final == pytest.approx(83.75)


def test_no_signals_returns_none():
    final, coverage, _ = aggregate_signals({}, WEIGHTS.signal_weights)
    assert final is None
    assert coverage == 0.0


# --- entry thresholds --------------------------------------------------------

def test_buy_watchlist_hold_thresholds():
    def decide_with(score):
        signals = {k: (score, 1.0) for k in WEIGHTS.signal_weights}
        return engine.decide("TEST", "r1", signals)

    assert decide_with(75.0).action == Action.BUY
    assert decide_with(70.0).action == Action.BUY        # boundary inclusive
    assert decide_with(65.0).action == Action.WATCHLIST
    assert decide_with(60.0).action == Action.WATCHLIST  # boundary inclusive
    assert decide_with(55.0).action == Action.HOLD


def test_insufficient_coverage_blocks_entry():
    # Only news at low confidence: coverage 0.10*0.4 = 0.04 << 0.25
    decision = engine.decide("TEST", "r1", {"news": (95.0, 0.4)})
    assert decision.action == Action.HOLD
    assert "insufficient signal coverage" in decision.reason


def test_decision_carries_breakdown_and_coverage():
    decision = engine.decide("TEST", "r1", FULL_CONF)
    assert decision.signal_breakdown["coverage"] == 1.0
    assert set(decision.signal_breakdown["signals"]) == set(WEIGHTS.signal_weights)


# --- exit policy -------------------------------------------------------------

def test_stop_loss_fires_even_with_strong_score():
    decision = engine.decide(
        "TEST", "r1", FULL_CONF,             # score 80: signals say keep
        position=held(avg_cost=100.0), current_price=91.9, today=TODAY,
    )
    assert decision.action == Action.SELL
    assert decision.reason.startswith("stop_loss")
    assert decision.sizing_hint == 10  # sell the whole position


def test_take_profit_fires():
    decision = engine.decide(
        "TEST", "r1", FULL_CONF,
        position=held(avg_cost=100.0), current_price=120.5, today=TODAY,
    )
    assert decision.action == Action.SELL
    assert decision.reason.startswith("take_profit")


def test_max_hold_fires_on_calendar_days():
    decision = engine.decide(
        "TEST", "r1", FULL_CONF,
        position=held(opened=date(2026, 4, 1)), current_price=105.0, today=TODAY,
    )
    assert decision.action == Action.SELL
    assert decision.reason.startswith("max_hold")  # 71 days >= 60


def test_score_decay_fires_below_exit_threshold():
    weak = {k: (40.0, 1.0) for k in WEIGHTS.signal_weights}
    decision = engine.decide(
        "TEST", "r1", weak,
        position=held(), current_price=102.0, today=TODAY,
    )
    assert decision.action == Action.SELL
    assert decision.reason.startswith("score_decay")


def test_score_decay_suspended_on_thin_coverage():
    # Weak score but only one low-confidence signal -> not trustworthy enough to exit.
    decision = engine.decide(
        "TEST", "r1", {"news": (10.0, 0.3)},
        position=held(), current_price=102.0, today=TODAY,
    )
    assert decision.action == Action.HOLD


def test_price_exits_fire_without_any_signals():
    decision = engine.decide(
        "TEST", "r1", {},
        position=held(avg_cost=100.0), current_price=80.0, today=TODAY,
    )
    assert decision.action == Action.SELL
    assert decision.reason.startswith("stop_loss")


def test_healthy_position_holds():
    decision = engine.decide(
        "TEST", "r1", FULL_CONF,
        position=held(), current_price=105.0, today=TODAY,
    )
    assert decision.action == Action.HOLD
    assert "no exit condition" in decision.reason


def test_held_ticker_never_rebuys():
    # Score 80 would be a buy — but the ticker is already held.
    decision = engine.decide(
        "TEST", "r1", FULL_CONF,
        position=held(), current_price=105.0, today=TODAY,
    )
    assert decision.action != Action.BUY


def test_stop_loss_beats_score_decay_priority():
    weak = {k: (10.0, 1.0) for k in WEIGHTS.signal_weights}
    decision = engine.decide(
        "TEST", "r1", weak,
        position=held(avg_cost=100.0), current_price=85.0, today=TODAY,
    )
    assert decision.reason.startswith("stop_loss")  # not score_decay
