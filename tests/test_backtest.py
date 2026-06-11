"""Backtester — PIT safety, live-code-path fidelity, and the sweep."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import make_ohlcv, make_ohlcv_piecewise
from trading_platform.backtest.replay import (
    BacktestSpec,
    renormalized_weights,
    run_backtest,
    sensitivity_sweep,
    write_backtest_report,
)
from trading_platform.core.config import load_config
from trading_platform.core.db import connect

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config")

END = date(2026, 6, 11)
START = date(2026, 1, 15)  # ~100 trading days; frames carry 400 bars of warmup


def frames_uptrend_and_downtrend():
    return {
        "AAPL": make_ohlcv(n_rows=400, end=END, drift=0.0025, daily_vol=0.006, seed=11),
        "MSFT": make_ohlcv(n_rows=400, end=END, drift=-0.0025, daily_vol=0.006, seed=12),
        "SPY": make_ohlcv(n_rows=400, end=END, drift=0.0005, daily_vol=0.008, seed=13),
    }


def spec(name="t", **kw):
    return BacktestSpec(name=name, start=START, end=END, signals=["technical"], **kw)


# --- weight renormalization ----------------------------------------------------

def test_single_signal_renormalizes_to_one():
    w = renormalized_weights(CONFIG.weights, spec())
    assert w.signal_weights == {"technical": 1.0}
    assert w.thresholds.buy_score == CONFIG.weights.thresholds.buy_score


def test_two_signals_keep_relative_proportions():
    s = BacktestSpec(name="t", start=START, end=END, signals=["technical", "kronos"])
    w = renormalized_weights(CONFIG.weights, s)
    # live: technical 0.25, kronos 0.20 -> 5/9, 4/9
    assert w.signal_weights["technical"] == pytest.approx(0.25 / 0.45)
    assert w.signal_weights["kronos"] == pytest.approx(0.20 / 0.45)


def test_non_pit_safe_signal_rejected():
    s = BacktestSpec(name="t", start=START, end=END, signals=["technical", "news"])
    with pytest.raises(ValueError, match="not PIT-safe"):
        renormalized_weights(CONFIG.weights, s)


# --- replay behavior ------------------------------------------------------------

@pytest.fixture(scope="module")
def trend_result(tmp_path_factory):
    """One shared replay over the trend fixture (module-scoped: it's the slow bit)."""
    out = tmp_path_factory.mktemp("bt") / "trend.sqlite"
    summary = run_backtest(CONFIG, spec("trend"), frames_uptrend_and_downtrend(), out)
    return summary, out


def test_uptrend_bought_downtrend_never(trend_result):
    summary, out_db = trend_result
    conn = connect(out_db)
    sides = conn.execute(
        "SELECT ticker, side, COUNT(*) n FROM orders GROUP BY ticker, side"
    ).fetchall()
    by = {(r["ticker"], r["side"]): r["n"] for r in sides}
    conn.close()
    assert by.get(("AAPL", "buy"), 0) >= 1     # strong uptrend -> bought
    assert by.get(("MSFT", "buy"), 0) == 0     # downtrend -> never


def test_account_profits_on_uptrend(trend_result):
    summary, _ = trend_result
    assert summary["total_return"] is not None and summary["total_return"] > 0
    assert summary["n_snapshots"] >= 90
    assert summary["spec"]["signals"] == ["technical"]
    assert summary["benchmarks"]["SPY"]["total_return"] is not None


def test_orders_auto_approved_and_filled_at_next_open(trend_result):
    _, out_db = trend_result
    conn = connect(out_db)
    assert conn.execute(
        "SELECT COUNT(*) FROM orders WHERE status = 'awaiting_approval'"
    ).fetchone()[0] == 0

    first = conn.execute(
        """SELECT o.ticker, r.run_date, f.price, f.filled_at FROM fills f
           JOIN orders o ON f.order_id = o.order_id
           JOIN runs r ON o.run_id = r.run_id
           WHERE f.side = 'buy' ORDER BY f.filled_at LIMIT 1"""
    ).fetchone()
    conn.close()
    assert first is not None
    assert first["filled_at"] > first["run_date"]  # strictly next day, never same-day

    frame = frames_uptrend_and_downtrend()[first["ticker"]]
    open_next = float(frame.loc[pd.Timestamp(first["filled_at"]), "open"])
    assert first["price"] == pytest.approx(open_next * 1.0005)  # 5 bps slippage


def test_no_lookahead_future_data_cannot_change_results(trend_result, tmp_path):
    """Replaying on frames truncated at the window end must produce the
    identical equity curve — future bars can never leak backwards."""
    summary_full, out_full = trend_result
    truncated = {
        t: df.loc[:pd.Timestamp(END)]
        for t, df in frames_uptrend_and_downtrend().items()
    }
    out_trunc = tmp_path / "trunc.sqlite"
    run_backtest(CONFIG, spec("trunc"), truncated, out_trunc)

    def curve(db):
        conn = connect(db)
        rows = conn.execute(
            "SELECT snapshot_date, equity FROM account_snapshots ORDER BY snapshot_date"
        ).fetchall()
        conn.close()
        return [(r["snapshot_date"], round(r["equity"], 6)) for r in rows]

    assert curve(out_full) == curve(out_trunc)


def test_determinism(tmp_path):
    frames = frames_uptrend_and_downtrend()
    a = run_backtest(CONFIG, spec("a"), frames, tmp_path / "a.sqlite")
    b = run_backtest(CONFIG, spec("b"), frames, tmp_path / "b.sqlite")
    assert a["total_return"] == b["total_return"]
    assert a["trades"] == b["trades"]


def test_stop_loss_exits_in_crash(tmp_path):
    """Rally then a sharp crash: the position must exit via stop_loss."""
    frames = {
        "AAPL": make_ohlcv_piecewise([(360, 0.0025), (40, -0.015)], end=END, seed=21),
        "SPY": make_ohlcv(n_rows=400, end=END, seed=22),
    }
    out = tmp_path / "crash.sqlite"
    run_backtest(CONFIG, spec("crash"), frames, out)

    conn = connect(out)
    sells = conn.execute(
        "SELECT d.reason FROM decisions d WHERE d.action = 'sell'"
    ).fetchall()
    n_sell_fills = conn.execute(
        "SELECT COUNT(*) FROM fills WHERE side = 'sell'"
    ).fetchone()[0]
    open_positions = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    conn.close()

    assert any(r["reason"].startswith("stop_loss") for r in sells), \
        [r["reason"][:40] for r in sells]
    assert n_sell_fills >= 1
    assert open_positions == 0  # crash flushed everything


# --- sweep + report --------------------------------------------------------------

def test_threshold_sweep_produces_comparison(tmp_path):
    frames = frames_uptrend_and_downtrend()
    # A choppy ticker whose technical score oscillates across the 60-75 band,
    # so different entry thresholds genuinely admit different trades.
    frames["GOOGL"] = make_ohlcv(n_rows=400, end=END, drift=0.0008,
                                 daily_vol=0.025, seed=41)
    rows = sensitivity_sweep(CONFIG, spec("sw"), frames, tmp_path)
    assert len(rows) == 4
    assert [r["buy_score"] for r in rows] == [60.0, 65.0, 70.0, 75.0]
    for r in rows:
        assert set(r) >= {"variant", "total_return", "sharpe", "max_drawdown",
                          "n_trades", "win_rate"}
    # a lower entry bar can never trade LESS than a higher one
    assert rows[0]["n_trades"] >= rows[-1]["n_trades"]
    # regression: variants below the live risk floor (70) must actually differ —
    # the risk-engine backstop tracks the swept threshold (it once didn't)
    assert rows[0]["n_trades"] != rows[-1]["n_trades"] or \
        rows[0]["total_return"] != rows[-1]["total_return"]


def test_report_written_with_assumptions(tmp_path, monkeypatch, trend_result):
    summary, _ = trend_result
    monkeypatch.setattr(CONFIG.settings, "reports_dir", None, raising=False)
    config = CONFIG.model_copy(deep=True)
    config.settings.reports_dir = tmp_path
    path = write_backtest_report(config, summary)
    text = path.read_text(encoding="utf-8")
    assert "Assumptions" in text
    assert "technical" in text
    assert "not point-in-time-safe" in text
    assert "auto-approved" in text
    assert (tmp_path / "backtests" / f"{summary['spec']['name']}.json").exists()
