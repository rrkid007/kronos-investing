"""Performance analytics over the live audit trail.

Everything derives from three tables the platform already writes:
account_snapshots (equity curve), fills (closed trades), and agent_scores
joined to price_cache (per-signal hit rates, PLAN.md S6 — the evidence that
will eventually retune the signal weights).

Conventions:
- Sharpe assumes a zero risk-free rate, annualized over 252 trading days.
- A "call" for hit-rate purposes is score >= 60 (bullish) or <= 40 (bearish)
  with confidence > 0; mid-band scores aren't calls. A bullish call hits when
  the forward `horizon_days` return is positive; bearish when negative.
- Closed trades reconstruct weighted-average cost from fills chronologically —
  the same convention the broker applies.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --- equity curve ------------------------------------------------------------

def equity_curve(conn: sqlite3.Connection) -> pd.Series:
    df = pd.read_sql_query(
        "SELECT snapshot_date, equity FROM account_snapshots ORDER BY snapshot_date",
        conn, parse_dates=["snapshot_date"], index_col="snapshot_date",
    )
    return df["equity"]


def total_return(equity: pd.Series) -> float | None:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return None
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_return(equity: pd.Series) -> float | None:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return None
    periods = len(equity) - 1
    return float((equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / periods) - 1.0)


def sharpe_ratio(equity: pd.Series) -> float | None:
    """Zero risk-free rate, annualized. None when undefined (flat or short)."""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return None
    std = float(returns.std(ddof=1))
    if std == 0 or math.isnan(std):
        return None
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float | None:
    if len(equity) < 2:
        return None
    return float((equity / equity.cummax() - 1.0).min())


# --- closed trades -----------------------------------------------------------

def closed_trades(conn: sqlite3.Connection) -> list[dict]:
    """Reconstruct round-trip trades from fills (weighted-average cost)."""
    fills = conn.execute(
        "SELECT ticker, side, qty, price, commission, filled_at FROM fills "
        "ORDER BY filled_at, fill_id"
    ).fetchall()

    open_lots: dict[str, dict] = {}  # ticker -> {qty, avg_cost, entry_date}
    trades: list[dict] = []
    for f in fills:
        ticker = f["ticker"]
        if f["side"] == "buy":
            lot = open_lots.get(ticker)
            if lot is None:
                open_lots[ticker] = {
                    "qty": f["qty"], "avg_cost": f["price"], "entry_date": f["filled_at"],
                }
            else:
                total = lot["qty"] + f["qty"]
                lot["avg_cost"] = (lot["qty"] * lot["avg_cost"] + f["qty"] * f["price"]) / total
                lot["qty"] = total
        else:
            lot = open_lots.get(ticker)
            if lot is None:
                continue  # sell without tracked entry; skip rather than invent
            qty = min(f["qty"], lot["qty"])
            pnl = (f["price"] - lot["avg_cost"]) * qty - f["commission"]
            trades.append({
                "ticker": ticker,
                "qty": qty,
                "entry_price": round(lot["avg_cost"], 6),
                "exit_price": f["price"],
                "entry_date": lot["entry_date"][:10],
                "exit_date": f["filled_at"][:10],
                "pnl": round(pnl, 6),
                "return_pct": round((f["price"] - lot["avg_cost"]) / lot["avg_cost"] * 100, 4),
            })
            lot["qty"] -= qty
            if lot["qty"] <= 1e-9:
                del open_lots[ticker]
    return trades


def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0, "wins": 0, "win_rate": None,
                "avg_return_pct": None, "total_pnl": 0.0}
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return {
        "n_trades": len(trades),
        "wins": wins,
        "win_rate": round(wins / len(trades), 4),
        "avg_return_pct": round(float(np.mean([t["return_pct"] for t in trades])), 4),
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
    }


# --- benchmarks ----------------------------------------------------------------

def benchmark_comparison(
    conn: sqlite3.Connection, start_date: str, end_date: str, tickers: list[str]
) -> dict[str, dict]:
    """Benchmark total return over [start_date, end_date] from the price cache."""
    out: dict[str, dict] = {}
    for ticker in tickers:
        df = pd.read_sql_query(
            "SELECT date, adj_close FROM price_cache "
            "WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
            conn, params=(ticker, start_date, end_date),
        )
        if len(df) < 2:
            out[ticker] = {"total_return": None, "n_days": len(df)}
            continue
        out[ticker] = {
            "total_return": round(float(df["adj_close"].iloc[-1] / df["adj_close"].iloc[0] - 1), 6),
            "n_days": len(df),
        }
    return out


# --- per-signal hit rates (S6) -------------------------------------------------

def signal_hit_rates(conn: sqlite3.Connection, horizon_days: int = 10) -> dict[str, dict]:
    """Did each agent's directional calls agree with forward returns?

    Only calls old enough to have `horizon_days` of forward data count;
    recent calls are pending, not misses.
    """
    scores = conn.execute(
        """
        SELECT a.agent, a.ticker, a.score, r.run_date FROM agent_scores a
        JOIN runs r ON a.run_id = r.run_id
        WHERE a.confidence > 0 AND (a.score >= 60 OR a.score <= 40)
        """
    ).fetchall()

    price_series: dict[str, pd.Series] = {}

    def forward_return(ticker: str, run_date: str) -> float | None:
        if ticker not in price_series:
            df = pd.read_sql_query(
                "SELECT date, adj_close FROM price_cache WHERE ticker = ? ORDER BY date",
                conn, params=(ticker,), index_col="date",
            )
            price_series[ticker] = df["adj_close"]
        series = price_series[ticker]
        idx = series.index.searchsorted(run_date)
        if idx >= len(series) or series.index[idx] != run_date:
            return None  # no bar on the call date
        if idx + horizon_days >= len(series):
            return None  # not enough forward history yet
        return float(series.iloc[idx + horizon_days] / series.iloc[idx] - 1.0)

    stats: dict[str, dict] = {}
    for row in scores:
        fwd = forward_return(row["ticker"], row["run_date"])
        if fwd is None:
            continue
        agent_stats = stats.setdefault(row["agent"], {"n_calls": 0, "hits": 0})
        bullish = row["score"] >= 60
        hit = (fwd > 0) if bullish else (fwd < 0)
        agent_stats["n_calls"] += 1
        agent_stats["hits"] += int(hit)

    for agent_stats in stats.values():
        agent_stats["hit_rate"] = (
            round(agent_stats["hits"] / agent_stats["n_calls"], 4)
            if agent_stats["n_calls"] else None
        )
    return stats


# --- aggregate ----------------------------------------------------------------

def performance_summary(conn: sqlite3.Connection, benchmarks: list[str]) -> dict:
    equity = equity_curve(conn)
    trades = closed_trades(conn)
    summary = {
        "n_snapshots": len(equity),
        "total_return": _round(total_return(equity)),
        "annualized_return": _round(annualized_return(equity)),
        "sharpe": _round(sharpe_ratio(equity)),
        "max_drawdown": _round(max_drawdown(equity)),
        "trades": trade_stats(trades),
        "signal_hit_rates": signal_hit_rates(conn),
    }
    if len(equity) >= 2:
        start = equity.index[0].date().isoformat()
        end = equity.index[-1].date().isoformat()
        summary["benchmarks"] = benchmark_comparison(conn, start, end, benchmarks)
        summary["window"] = {"start": start, "end": end}
    return summary


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None
