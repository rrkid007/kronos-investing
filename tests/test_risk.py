"""Risk Engine — exhaustive rule coverage: pass, fail, and boundary per rule.

This is the safety layer; these tests are the spec.
"""

from datetime import date
from pathlib import Path


from trading_platform.agents.portfolio import PortfolioState
from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.core.models import Action, Position, TradeDecision
from trading_platform.risk.engine import RiskEngine, log_risk_event

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config")
# Limits under test (from config): max_position 15%, max_sector 40%,
# cash reserve 10%, min_final_score 70, kronos floor 25.

engine = RiskEngine(CONFIG.risk, CONFIG.watchlist)


def buy(ticker="AAPL", score=80.0, kronos=(60.0, 0.9)):
    signals = {
        "kronos": {"score": kronos[0], "confidence": kronos[1],
                   "weight": 0.2, "effective_weight": 0.18},
    }
    return TradeDecision(
        run_id="r1", ticker=ticker, action=Action.BUY, final_score=score,
        signal_breakdown={"signals": signals, "coverage": 0.8},
    )


def sell(ticker="AAPL", qty=10.0):
    return TradeDecision(
        run_id="r1", ticker=ticker, action=Action.SELL, final_score=30.0,
        sizing_hint=qty, reason="stop_loss",
    )


def state(cash=100_000.0, equity=100_000.0, sector_values=None, positions=None):
    return PortfolioState(
        cash=cash, equity=equity,
        sector_values=sector_values or {}, positions=positions or [],
    )


def failed_rules(result):
    return {c.rule for c in result.checks if not c.passed}


# --- happy path --------------------------------------------------------------

def test_clean_buy_approved():
    result = engine.evaluate_buy(buy(), qty=10, price=100.0, state=state())
    assert result.approved, result.errors
    assert result.requires_human_approval is True  # config: manual approval
    assert result.errors == []


# --- sane_inputs -------------------------------------------------------------

def test_wrong_action_rejected():
    decision = sell()
    result = engine.evaluate_buy(decision, qty=10, price=100.0, state=state())
    assert not result.approved
    assert "sane_inputs" in failed_rules(result)


def test_zero_qty_rejected():
    result = engine.evaluate_buy(buy(), qty=0, price=100.0, state=state())
    assert "sane_inputs" in failed_rules(result)


def test_zero_or_none_price_rejected():
    assert "sane_inputs" in failed_rules(engine.evaluate_buy(buy(), 10, 0.0, state()))
    assert "sane_inputs" in failed_rules(engine.evaluate_buy(buy(), 10, None, state()))


# --- min_final_score ---------------------------------------------------------

def test_final_score_below_minimum_rejected():
    result = engine.evaluate_buy(buy(score=69.99), qty=10, price=100.0, state=state())
    assert "min_final_score" in failed_rules(result)


def test_final_score_exactly_at_minimum_passes():
    result = engine.evaluate_buy(buy(score=70.0), qty=10, price=100.0, state=state())
    assert "min_final_score" not in failed_rules(result)


# --- min_agent_scores --------------------------------------------------------

def test_kronos_below_floor_blocks_buy():
    result = engine.evaluate_buy(buy(kronos=(24.9, 0.9)), 10, 100.0, state())
    assert "min_agent_score:kronos" in failed_rules(result)


def test_kronos_exactly_at_floor_passes():
    result = engine.evaluate_buy(buy(kronos=(25.0, 0.9)), 10, 100.0, state())
    assert "min_agent_score:kronos" not in failed_rules(result)


def test_dead_kronos_is_not_blocking():
    result = engine.evaluate_buy(buy(kronos=(10.0, 0.0)), 10, 100.0, state())
    assert "min_agent_score:kronos" not in failed_rules(result)
    assert result.approved


def test_absent_agent_is_not_blocking():
    decision = buy()
    decision.signal_breakdown["signals"] = {}
    result = engine.evaluate_buy(decision, 10, 100.0, state())
    assert result.approved


# --- not_restricted ----------------------------------------------------------

def test_restricted_asset_rejected():
    restricted_engine = RiskEngine(
        CONFIG.risk.model_copy(update={"restricted_assets": ["AAPL"]}),
        CONFIG.watchlist,
    )
    result = restricted_engine.evaluate_buy(buy("AAPL"), 10, 100.0, state())
    assert "not_restricted" in failed_rules(result)
    # other tickers unaffected
    ok = restricted_engine.evaluate_buy(buy("MSFT"), 10, 100.0, state())
    assert "not_restricted" not in failed_rules(ok)


# --- position_limit ----------------------------------------------------------

def test_position_over_cap_rejected():
    # 15% of 100k = 15000; 151 x 100 = 15100
    result = engine.evaluate_buy(buy(), qty=151, price=100.0, state=state())
    assert "position_limit" in failed_rules(result)


def test_position_exactly_at_cap_passes():
    result = engine.evaluate_buy(buy(), qty=150, price=100.0, state=state())
    assert "position_limit" not in failed_rules(result)


# --- sector_limit ------------------------------------------------------------

def test_sector_breach_rejected():
    # Technology at 39%; +1100 -> 40.1% > 40%
    s = state(sector_values={"Technology": 39_000.0})
    result = engine.evaluate_buy(buy("AAPL"), qty=11, price=100.0, state=s)
    assert "sector_limit" in failed_rules(result)


def test_sector_exactly_at_cap_passes():
    s = state(sector_values={"Technology": 39_000.0})
    result = engine.evaluate_buy(buy("AAPL"), qty=10, price=100.0, state=s)
    assert "sector_limit" not in failed_rules(result)


def test_unknown_sector_tracked_as_unknown():
    s = state(sector_values={"Unknown": 39_500.0})
    result = engine.evaluate_buy(buy("ZZZT"), qty=10, price=100.0, state=s)
    assert "sector_limit" in failed_rules(result)


# --- cash rules --------------------------------------------------------------

def test_cost_above_cash_rejected():
    s = state(cash=900.0, equity=100_000.0)
    result = engine.evaluate_buy(buy(), qty=10, price=100.0, state=s)
    assert "cash_sufficient" in failed_rules(result)


def test_reserve_floor_breach_rejected():
    # equity 100k -> reserve 10k. cash 10.9k, cost 1000 -> 9.9k < 10k
    s = state(cash=10_900.0, equity=100_000.0)
    result = engine.evaluate_buy(buy(), qty=10, price=100.0, state=s)
    assert "cash_reserve" in failed_rules(result)


def test_reserve_floor_exactly_met_passes():
    s = state(cash=11_000.0, equity=100_000.0)
    result = engine.evaluate_buy(buy(), qty=10, price=100.0, state=s)
    assert "cash_reserve" not in failed_rules(result)


# --- multiple violations all reported ---------------------------------------

def test_all_violations_named():
    s = state(cash=500.0, equity=100_000.0, sector_values={"Technology": 40_000.0})
    result = engine.evaluate_buy(
        buy(score=50.0, kronos=(10.0, 0.9)), qty=200, price=100.0, state=s
    )
    assert not result.approved
    named = failed_rules(result)
    assert {"min_final_score", "min_agent_score:kronos", "position_limit",
            "sector_limit", "cash_sufficient", "cash_reserve"} <= named
    assert len(result.errors) >= 6  # every violation reported, not just the first


# --- sells -------------------------------------------------------------------

def position(qty=10.0):
    return Position(ticker="AAPL", qty=qty, avg_cost=100.0, opened_at=date(2026, 6, 1))


def test_sell_with_position_approved():
    result = engine.evaluate_sell(sell(qty=10), position(qty=10), price=95.0)
    assert result.approved


def test_sell_never_blocked_by_exposure_rules():
    """A stop-loss must clear even when the account is in terrible shape —
    no exposure rule appears in sell checks at all."""
    result = engine.evaluate_sell(sell(qty=10), position(qty=10), price=95.0)
    rules = {c.rule for c in result.checks}
    assert rules == {"sane_inputs", "position_exists", "qty_within_position"}


def test_sell_without_position_rejected():
    result = engine.evaluate_sell(sell(), None, price=95.0)
    assert "position_exists" in failed_rules(result)


def test_sell_more_than_held_rejected():
    result = engine.evaluate_sell(sell(qty=11), position(qty=10), price=95.0)
    assert "qty_within_position" in failed_rules(result)


def test_sell_exact_position_qty_passes():
    result = engine.evaluate_sell(sell(qty=10), position(qty=10), price=95.0)
    assert "qty_within_position" not in failed_rules(result)


# --- determinism & logging ---------------------------------------------------

def test_evaluation_is_deterministic():
    a = engine.evaluate_buy(buy(), 10, 100.0, state())
    b = engine.evaluate_buy(buy(), 10, 100.0, state())
    assert a.model_dump() == b.model_dump()


def test_risk_event_logged(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_db(conn)
    conn.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('r1', '2026-06-11', '2026-06-11T00:00:00', 'running')"
    )
    blocked = engine.evaluate_buy(buy(score=10.0), 10, 100.0, state())
    log_risk_event(conn, "r1", blocked)

    row = conn.execute("SELECT * FROM risk_events").fetchone()
    assert row["ticker"] == "AAPL"
    assert row["approved"] == 0
    assert "min_final_score" in row["violations"]
    conn.close()
