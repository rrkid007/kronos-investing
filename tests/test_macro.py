"""Macro regime — classifier boundaries, FRED parsing, cache fallback, sizing."""

from datetime import date

import pytest

import trading_platform.data.macro as macro_mod
from trading_platform.core.config import MacroSettings
from trading_platform.core.db import connect, init_db
from trading_platform.data.macro import MacroSnapshot, refresh_macro
from trading_platform.risk.regime import classify_regime

SETTINGS = MacroSettings()


def snap(curve=1.0, vix=15.0, hy=3.0):
    return MacroSnapshot(yield_curve=curve, vix=vix, hy_oas=hy,
                         as_of={"vix": "2026-06-10"})


# --- classifier ----------------------------------------------------------------

def test_calm_regime():
    a = classify_regime(snap(), SETTINGS)
    assert a.regime == "calm"
    assert a.stress_score == 0
    assert a.sizing_scalar == 1.0


def test_full_stress_regime():
    a = classify_regime(snap(curve=-0.5, vix=45.0, hy=8.0), SETTINGS)
    assert a.regime == "stress"
    assert a.stress_score == 6
    assert a.max_score == 6
    assert a.sizing_scalar == 0.5


def test_caution_regime():
    # one elevated + one stress + one calm = 3/6 -> caution
    a = classify_regime(snap(curve=1.0, vix=22.0, hy=6.5), SETTINGS)
    assert a.regime == "caution"
    assert a.stress_score == 3
    assert a.sizing_scalar == 0.75


def test_component_boundaries():
    # exactly at each threshold -> the higher tier (>= semantics)
    a = classify_regime(snap(vix=20.0), SETTINGS)
    vix = next(c for c in a.components if c.indicator == "vix")
    assert vix.points == 1
    a = classify_regime(snap(vix=30.0), SETTINGS)
    assert next(c for c in a.components if c.indicator == "vix").points == 2

    a = classify_regime(snap(hy=4.0), SETTINGS)
    assert next(c for c in a.components if c.indicator == "hy_oas").points == 1
    a = classify_regime(snap(hy=6.0), SETTINGS)
    assert next(c for c in a.components if c.indicator == "hy_oas").points == 2

    # curve: < semantics (0.5 flat boundary itself is NOT flat)
    a = classify_regime(snap(curve=0.5), SETTINGS)
    assert next(c for c in a.components if c.indicator == "yield_curve").points == 0
    a = classify_regime(snap(curve=0.49), SETTINGS)
    assert next(c for c in a.components if c.indicator == "yield_curve").points == 1
    a = classify_regime(snap(curve=-0.01), SETTINGS)
    assert next(c for c in a.components if c.indicator == "yield_curve").points == 2


def test_insufficient_indicators_is_unknown_and_neutral():
    a = classify_regime(MacroSnapshot(vix=50.0), SETTINGS)  # one dial only
    assert a.regime == "unknown"
    assert a.sizing_scalar == 1.0  # a blind read must not move sizes


def test_two_of_three_still_classifies():
    a = classify_regime(MacroSnapshot(vix=45.0, hy_oas=8.0), SETTINGS)
    assert a.regime == "stress"
    assert a.max_score == 4


def test_summary_strings():
    assert "sizing x0.5" in classify_regime(snap(curve=-1, vix=40, hy=9), SETTINGS).summary()
    assert "unscaled" in classify_regime(MacroSnapshot(), SETTINGS).summary()


# --- fetch + cache ---------------------------------------------------------------

FRED_CSV = """DATE,VIXCLS
2026-06-08,17.2
2026-06-09,.
2026-06-10,18.4
"""


class FakeResp:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    yield c
    c.close()


def test_fetch_parses_and_skips_missing_values(conn, monkeypatch):
    monkeypatch.setattr(macro_mod.requests, "get",
                        lambda *a, **k: FakeResp(FRED_CSV.replace("VIXCLS", "X")))
    snapshot = refresh_macro(conn, as_of=date(2026, 6, 11))
    # all three series served the same canned data
    assert snapshot.vix == pytest.approx(18.4)   # latest non-missing
    assert snapshot.n_available == 3
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM macro_indicators WHERE series='VIXCLS'"
    ).fetchone()[0]
    assert n_rows == 2  # '.' row dropped


def test_fetch_failure_falls_back_to_fresh_cache(conn, monkeypatch):
    conn.execute(
        "INSERT INTO macro_indicators (series, date, value, fetched_at) "
        "VALUES ('VIXCLS', '2026-06-09', 21.5, '')"
    )
    conn.commit()

    def boom(*a, **k):
        raise ConnectionError("fred down")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    snapshot = refresh_macro(conn, as_of=date(2026, 6, 11))
    assert snapshot.vix == pytest.approx(21.5)   # cached value survives the outage
    assert snapshot.yield_curve is None          # nothing cached for this one


def test_stale_cache_ignored(conn, monkeypatch):
    conn.execute(
        "INSERT INTO macro_indicators (series, date, value, fetched_at) "
        "VALUES ('VIXCLS', '2026-05-01', 21.5, '')"
    )
    conn.commit()
    monkeypatch.setattr(macro_mod.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    snapshot = refresh_macro(conn, as_of=date(2026, 6, 11), max_staleness_days=7)
    assert snapshot.vix is None  # 41 days old: not trustworthy


# --- sizing effect ----------------------------------------------------------------

def test_regime_scalar_scales_position_size():
    from pathlib import Path

    from tests.fixtures import make_ohlcv
    from trading_platform.agents.portfolio import PortfolioAgent, PortfolioState
    from trading_platform.core.config import load_config
    from trading_platform.core.models import Action, TradeDecision

    config = load_config(Path(__file__).resolve().parents[1] / "config")
    agent = PortfolioAgent(config.risk, config.watchlist)
    df = make_ohlcv(daily_vol=0.02)  # below caps so scaling is visible
    decision = TradeDecision(run_id="r", ticker="AAPL", action=Action.BUY,
                             final_score=80.0)

    def state():
        return PortfolioState(cash=100_000.0, equity=100_000.0)

    full = agent.assess_buy(decision, 100.0, df, state(), regime_scalar=1.0)
    halved = agent.assess_buy(decision, 100.0, df, state(), regime_scalar=0.5)
    assert full.approved and halved.approved
    assert halved.target_value == pytest.approx(full.target_value * 0.5, rel=0.02)
    assert any("macro regime scaling" in c for c in halved.checks)
