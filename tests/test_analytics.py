"""Performance analytics — every metric validated against hand-computed values.

Reference values were computed independently with explicit numpy arithmetic
(see comments), not by calling the module under test.
"""

import numpy as np
import pandas as pd
import pytest

from trading_platform.analytics.performance import (
    annualized_return,
    benchmark_comparison,
    closed_trades,
    equity_curve,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    signal_hit_rates,
    total_return,
    trade_stats,
)
from trading_platform.core.db import connect, init_db


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    init_db(c)
    # Seed run + order so fill fixtures satisfy the FK chain.
    c.execute(
        "INSERT INTO runs (run_id, run_date, started_at, status) "
        "VALUES ('seed', '2026-05-01', '', 'completed')"
    )
    c.execute(
        "INSERT INTO orders (order_id, run_id, ticker, side, qty, status, created_at) "
        "VALUES ('o', 'seed', 'X', 'buy', 1, 'filled', '')"
    )
    c.commit()
    yield c
    c.close()


def insert_snapshots(conn, equities, start="2026-05-04"):
    dates = pd.bdate_range(start=start, periods=len(equities))
    for d, eq in zip(dates, equities):
        conn.execute(
            "INSERT INTO account_snapshots (snapshot_date, cash, equity, "
            "unrealized_pnl, realized_pnl, created_at) VALUES (?, 0, ?, 0, 0, '')",
            (d.date().isoformat(), eq),
        )
    conn.commit()


def insert_fill(conn, ticker, side, qty, price, filled_at, commission=0.0):
    conn.execute(
        "INSERT INTO fills (fill_id, order_id, ticker, side, qty, price, slippage, "
        "commission, filled_at) VALUES (?, 'o', ?, ?, ?, ?, 0, ?, ?)",
        (f"{ticker}-{side}-{filled_at}", ticker, side, qty, price, commission, filled_at),
    )
    conn.commit()


# --- curve metrics: hand-computed case 1 --------------------------------------
# equity [100000, 101000, 100500, 102000]
# returns [0.01, -0.0049505, 0.01492537]
# total = 0.02; sharpe = 10.21155; maxdd = -0.0049505; annualized = 4.27733

CASE1 = [100_000.0, 101_000.0, 100_500.0, 102_000.0]


def test_total_return_hand_value(conn):
    insert_snapshots(conn, CASE1)
    assert total_return(equity_curve(conn)) == pytest.approx(0.02)


def test_sharpe_hand_value(conn):
    insert_snapshots(conn, CASE1)
    assert sharpe_ratio(equity_curve(conn)) == pytest.approx(10.211553, abs=1e-4)


def test_max_drawdown_hand_value(conn):
    insert_snapshots(conn, CASE1)
    assert max_drawdown(equity_curve(conn)) == pytest.approx(-0.00495049, abs=1e-6)


def test_annualized_return_hand_value(conn):
    insert_snapshots(conn, CASE1)
    assert annualized_return(equity_curve(conn)) == pytest.approx(4.277332, abs=1e-4)


def test_degenerate_curves(conn):
    assert total_return(equity_curve(conn)) is None  # empty
    insert_snapshots(conn, [100_000.0])
    eq = equity_curve(conn)
    assert total_return(eq) is None  # single point
    assert sharpe_ratio(eq) is None
    assert max_drawdown(eq) is None


def test_flat_curve_sharpe_undefined(conn):
    insert_snapshots(conn, [100_000.0] * 5)
    assert sharpe_ratio(equity_curve(conn)) is None  # zero variance


# --- closed trades ------------------------------------------------------------

def test_closed_trades_exact_pnl(conn):
    insert_fill(conn, "AAPL", "buy", 10, 100.05, "2026-06-01")
    insert_fill(conn, "AAPL", "sell", 10, 110.0, "2026-06-05")
    insert_fill(conn, "MSFT", "buy", 5, 200.0, "2026-06-02")
    insert_fill(conn, "MSFT", "sell", 5, 190.0, "2026-06-08")

    trades = closed_trades(conn)
    assert len(trades) == 2
    aapl = next(t for t in trades if t["ticker"] == "AAPL")
    assert aapl["pnl"] == pytest.approx((110.0 - 100.05) * 10)  # 99.50
    assert aapl["return_pct"] == pytest.approx(9.9450, abs=1e-3)
    msft = next(t for t in trades if t["ticker"] == "MSFT")
    assert msft["pnl"] == pytest.approx(-50.0)

    stats = trade_stats(trades)
    assert stats["n_trades"] == 2
    assert stats["wins"] == 1
    assert stats["win_rate"] == 0.5
    assert stats["total_pnl"] == pytest.approx(49.50)


def test_addon_buy_then_sell_uses_weighted_cost(conn):
    insert_fill(conn, "NVDA", "buy", 4, 50.0, "2026-06-01")
    insert_fill(conn, "NVDA", "buy", 4, 60.0, "2026-06-02")
    insert_fill(conn, "NVDA", "sell", 8, 58.0, "2026-06-09", commission=1.0)

    trades = closed_trades(conn)
    assert len(trades) == 1
    # avg cost (4x50 + 4x60)/8 = 55; pnl (58-55)*8 - 1 = 23
    assert trades[0]["entry_price"] == pytest.approx(55.0)
    assert trades[0]["pnl"] == pytest.approx(23.0)


def test_partial_sell_produces_trade_and_keeps_lot(conn):
    insert_fill(conn, "AAPL", "buy", 10, 100.0, "2026-06-01")
    insert_fill(conn, "AAPL", "sell", 4, 105.0, "2026-06-03")
    insert_fill(conn, "AAPL", "sell", 6, 110.0, "2026-06-05")

    trades = closed_trades(conn)
    assert len(trades) == 2
    assert trades[0]["pnl"] == pytest.approx((105.0 - 100.0) * 4)
    assert trades[1]["pnl"] == pytest.approx((110.0 - 100.0) * 6)


def test_empty_trades_stats():
    stats = trade_stats([])
    assert stats["n_trades"] == 0
    assert stats["win_rate"] is None


# --- benchmarks ----------------------------------------------------------------

def test_benchmark_comparison(conn):
    dates = pd.bdate_range("2026-05-04", periods=10)
    for i, d in enumerate(dates):
        conn.execute(
            "INSERT INTO price_cache (ticker, date, adj_close) VALUES ('SPY', ?, ?)",
            (d.date().isoformat(), 100.0 + i),  # 100 -> 109
        )
    conn.commit()
    out = benchmark_comparison(
        conn, dates[0].date().isoformat(), dates[-1].date().isoformat(), ["SPY", "QQQ"]
    )
    assert out["SPY"]["total_return"] == pytest.approx(0.09)
    assert out["QQQ"]["total_return"] is None  # no data


# --- signal hit rates -----------------------------------------------------------

def seed_hit_rate_fixture(conn):
    """30 business days of +1%/day prices; calls on day 5 resolve, day 25 pending."""
    dates = pd.bdate_range("2026-04-01", periods=30)
    for i, d in enumerate(dates):
        conn.execute(
            "INSERT INTO price_cache (ticker, date, adj_close) VALUES ('TEST', ?, ?)",
            (d.date().isoformat(), 100.0 * (1.01 ** i)),
        )
    call_day = dates[5].date().isoformat()
    late_day = dates[25].date().isoformat()
    for rid, rdate in [("r1", call_day), ("r2", late_day)]:
        conn.execute(
            "INSERT INTO runs (run_id, run_date, started_at, status) "
            "VALUES (?, ?, '', 'completed')", (rid, rdate),
        )

    def score(run_id, agent, score, conf):
        conn.execute(
            "INSERT INTO agent_scores (run_id, agent, ticker, score, confidence, "
            "direction, details, created_at) VALUES (?, ?, 'TEST', ?, ?, '', '{}', '')",
            (run_id, agent, score, conf),
        )

    score("r1", "technical", 80.0, 0.9)      # bullish call, rising market -> hit
    score("r1", "kronos", 20.0, 0.9)         # bearish call -> miss
    score("r1", "news", 50.0, 0.9)           # mid-band: not a call
    score("r1", "fundamentals", 80.0, 0.0)   # dead agent: excluded
    score("r2", "technical", 80.0, 0.9)      # too recent: no forward data yet
    conn.commit()


def test_signal_hit_rates(conn):
    seed_hit_rate_fixture(conn)
    stats = signal_hit_rates(conn, horizon_days=10)

    assert stats["technical"] == {"n_calls": 1, "hits": 1, "hit_rate": 1.0}
    assert stats["kronos"] == {"n_calls": 1, "hits": 0, "hit_rate": 0.0}
    assert "news" not in stats          # never made a directional call
    assert "fundamentals" not in stats  # zero-confidence excluded


# --- the done criterion: a simulated month ---------------------------------------
# 21 snapshots: +0.2% x10, -1% x3, +0.5% x7 (hand-computed via explicit numpy):
# end equity 102505.0428, total 0.0250504, annualized 0.3658088,
# sharpe 3.9325712, maxdd -0.029701

def test_simulated_month_all_metrics(conn):
    equities = [100_000.0]
    for d in [0.002] * 10 + [-0.01] * 3 + [0.005] * 7:
        equities.append(equities[-1] * (1 + d))
    insert_snapshots(conn, equities)

    insert_fill(conn, "AAPL", "buy", 10, 100.05, "2026-06-01")
    insert_fill(conn, "AAPL", "sell", 10, 110.0, "2026-06-05")
    insert_fill(conn, "MSFT", "buy", 5, 200.0, "2026-06-02")
    insert_fill(conn, "MSFT", "sell", 5, 190.0, "2026-06-08")

    summary = performance_summary(conn, benchmarks=["SPY"])
    assert summary["n_snapshots"] == 21
    assert summary["total_return"] == pytest.approx(0.0250504, abs=1e-6)
    assert summary["annualized_return"] == pytest.approx(0.3658088, abs=1e-5)
    assert summary["sharpe"] == pytest.approx(3.9325712, abs=1e-5)
    assert summary["max_drawdown"] == pytest.approx(-0.029701, abs=1e-6)
    assert summary["trades"]["win_rate"] == 0.5
    assert summary["trades"]["total_pnl"] == pytest.approx(49.50)
    assert summary["window"]["start"] == "2026-05-04"
    # equity end value cross-check against the hand computation
    eq = equity_curve(conn)
    assert float(eq.iloc[-1]) == pytest.approx(102_505.0428, abs=0.01)
    assert np.isclose(eq.iloc[0], 100_000.0)
