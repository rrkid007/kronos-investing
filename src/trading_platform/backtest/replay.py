"""Historical replay backtester (PLAN.md S2, phase 12).

Point-in-time honesty:
- Only PIT-safe signals replay: technical (always) and kronos (opt-in —
  expensive). Fundamentals/news/SEC snapshots are current-only and would
  inject look-ahead bias; they are validated forward by the live paper
  track record instead.
- Each simulated day D, agents see price history sliced to <= D. Decisions
  price at D's close; orders fill at D+1's open — identical to live.

Fidelity through reuse, not reimplementation: the replay drives the very
same TechnicalAgent / DecisionEngine / PortfolioAgent / RiskEngine /
submit_order / fill_order / snapshot_account code the live pipeline uses
(including pipeline's sizing/risk/order helpers), against a scratch SQLite
database that ends up holding a complete audit trail per backtest.

Differences from live, by necessity (stated in every report):
- Orders auto-approve (no human in a replay).
- Signal weights renormalize over the enabled signals (technical-only ->
  {technical: 1.0}); thresholds and exit policy come from live config.
- Account math uses raw closes (no dividends); benchmark comparison uses
  adjusted closes — the comparison therefore flatters the benchmark.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from trading_platform.agents.decision import DecisionEngine
from trading_platform.agents.kronos import KronosAgent
from trading_platform.agents.technical import TechnicalAgent
from trading_platform.analytics.performance import performance_summary
from trading_platform.core.config import AppConfig, Weights
from trading_platform.core.db import connect, init_db
from trading_platform.core.runs import (
    complete_run,
    create_run,
    save_agent_result,
    save_decision,
)
from trading_platform.execution.account import load_positions
from trading_platform.execution.orders import fillable_orders
from trading_platform.execution.paper_broker import (
    FillError,
    ensure_account,
    fill_order,
    snapshot_account,
)
from trading_platform.pipeline import (
    _assess_buys,
    _load_signals,
    _risk_pass,
    _submit_orders,
)

logger = logging.getLogger(__name__)

PIT_SAFE_SIGNALS = {"technical", "kronos"}
WARMUP_BARS = 250  # quality-gate minimum; technical needs the 200DMA


class BacktestSpec(BaseModel):
    name: str
    start: date
    end: date
    signals: list[str] = Field(default_factory=lambda: ["technical"])
    weights_override: dict[str, float] | None = None
    buy_score_override: float | None = None


def renormalized_weights(live: Weights, spec: BacktestSpec) -> Weights:
    """Restrict live weights to the enabled signals, renormalized to 1.0."""
    unknown = set(spec.signals) - PIT_SAFE_SIGNALS
    if unknown:
        raise ValueError(f"signals not PIT-safe for backtesting: {sorted(unknown)}")
    if spec.weights_override:
        base = dict(spec.weights_override)
    else:
        base = {k: live.signal_weights[k] for k in spec.signals}
    total = sum(base.values())
    thresholds = live.thresholds.model_copy()
    if spec.buy_score_override is not None:
        thresholds.buy_score = spec.buy_score_override
    return Weights(
        signal_weights={k: v / total for k, v in base.items()},
        thresholds=thresholds,
        exit_policy=live.exit_policy.model_copy(),
    )


def load_frames(
    live_conn: sqlite3.Connection, tickers: list[str]
) -> dict[str, pd.DataFrame]:
    """Full cached history per ticker (warmup happens by slicing per day)."""
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, adj_close, volume "
            "FROM price_cache WHERE ticker = ? ORDER BY date",
            live_conn, params=(ticker,), parse_dates=["date"], index_col="date",
        )
        if not df.empty:
            frames[ticker] = df
    return frames


def _build_agents(config: AppConfig, signals: list[str]) -> dict:
    agents: dict = {}
    if "technical" in signals:
        agents["technical"] = TechnicalAgent()
    if "kronos" in signals:
        agents["kronos"] = KronosAgent(config.settings.kronos)
    return agents


def run_backtest(
    config: AppConfig,
    spec: BacktestSpec,
    frames: dict[str, pd.DataFrame],
    out_db: Path,
) -> dict:
    """Replay [start, end]; returns the summary dict. Full audit in out_db."""
    sim_config = config.model_copy(deep=True)
    sim_config.risk.require_human_approval = False  # auto-approve in replay
    weights = renormalized_weights(config.weights, spec)
    # The risk engine's score floor is a backstop equal to the live entry
    # threshold; when a sweep moves the entry threshold it must follow, or
    # variants below the live floor silently collapse into it.
    sim_config.risk.min_final_score = weights.thresholds.buy_score
    engine = DecisionEngine(weights)
    agents = _build_agents(config, spec.signals)

    watch = [t for t in config.watchlist.symbols if t in frames]
    trading_days = sorted({
        d.date() for t in watch for d in frames[t].index
        if spec.start <= d.date() <= spec.end
    })
    if not trading_days:
        raise ValueError(f"no trading days in window {spec.start}..{spec.end}")

    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()
    conn = connect(out_db)
    init_db(conn)
    ensure_account(conn, config.risk.paper_account.starting_cash)

    try:
        for day in trading_days:
            day_iso = day.isoformat()
            run_id = create_run(conn, day_iso)
            ts = pd.Timestamp(day)

            # 1. fill yesterday's approved orders at today's open (live pass 0)
            for order in fillable_orders(conn, day_iso):
                df = frames.get(order["ticker"])
                bar = df.loc[df.index == ts] if df is not None else None
                if bar is None or bar.empty:
                    continue  # halted/missing: stays pending, same as live
                try:
                    fill_order(conn, order, day, float(bar["open"].iloc[0]),
                               sim_config.risk.fill_model)
                except FillError as exc:
                    conn.execute(
                        "UPDATE orders SET status = 'cancelled', notes = ? "
                        "WHERE order_id = ?", (str(exc), order["order_id"]),
                    )
                    conn.commit()

            positions = load_positions(conn)
            held = {p.ticker: p for p in positions}

            # 2. agents score point-in-time slices
            sliced: dict[str, pd.DataFrame] = {}
            for ticker in watch:
                df_slice = frames[ticker].loc[:ts]
                if len(df_slice) < WARMUP_BARS:
                    continue
                sliced[ticker] = df_slice
                for agent in agents.values():
                    save_agent_result(conn, agent.analyze(ticker, run_id, df_slice))

            # 3. decisions -> sizing -> risk -> orders: live code paths
            decisions = {}
            for ticker, df_slice in sliced.items():
                decision = engine.decide(
                    ticker, run_id,
                    signals=_load_signals(conn, run_id, ticker),
                    position=held.get(ticker),
                    current_price=float(df_slice["close"].iloc[-1]),
                    today=day,
                )
                decisions[ticker] = decision
                save_decision(conn, decision)

            _assess_buys(conn, sim_config, decisions, sliced, positions)
            _risk_pass(conn, sim_config, run_id, decisions, sliced, positions, held)
            _submit_orders(conn, sim_config, run_id, day_iso, decisions, sliced)

            closes = {
                t: float(df.loc[ts, "close"]) for t, df in frames.items()
                if ts in df.index
            }
            snapshot_account(conn, day_iso, closes)
            complete_run(conn, run_id)

        summary = performance_summary(conn, benchmarks=[])
        summary["benchmarks"] = _benchmark_returns(
            frames, config.settings.benchmarks, trading_days[0], trading_days[-1]
        )
        summary["spec"] = {
            "name": spec.name,
            "window": {"start": trading_days[0].isoformat(),
                       "end": trading_days[-1].isoformat()},
            "n_trading_days": len(trading_days),
            "signals": spec.signals,
            "weights": weights.signal_weights,
            "buy_score": weights.thresholds.buy_score,
        }
        return summary
    finally:
        conn.close()


def _benchmark_returns(
    frames: dict, benchmarks: list[str], start: date, end: date
) -> dict:
    out = {}
    for ticker in benchmarks:
        df = frames.get(ticker)
        if df is None:
            out[ticker] = {"total_return": None}
            continue
        window = df.loc[pd.Timestamp(start):pd.Timestamp(end), "adj_close"]
        out[ticker] = {
            "total_return": (
                round(float(window.iloc[-1] / window.iloc[0] - 1), 6)
                if len(window) >= 2 else None
            ),
        }
    return out


def sensitivity_sweep(
    config: AppConfig,
    base: BacktestSpec,
    frames: dict[str, pd.DataFrame],
    out_dir: Path,
) -> list[dict]:
    """Run a grid of variants and return one summary row per variant.

    Two or more signals: sweep the weight split between the first two.
    Single signal: sweep the buy-score entry threshold instead.
    """
    variants: list[BacktestSpec] = []
    if len(base.signals) >= 2:
        a, b = base.signals[0], base.signals[1]
        for w in (0.3, 0.45, 0.6, 0.75):
            variants.append(base.model_copy(update={
                "name": f"{base.name}-{a}{int(w * 100)}",
                "weights_override": {a: w, b: 1.0 - w},
            }))
    else:
        for threshold in (60.0, 65.0, 70.0, 75.0):
            variants.append(base.model_copy(update={
                "name": f"{base.name}-buy{int(threshold)}",
                "buy_score_override": threshold,
            }))

    rows = []
    for variant in variants:
        summary = run_backtest(config, variant, frames,
                               out_dir / f"{variant.name}.sqlite")
        rows.append({
            "variant": variant.name,
            "weights": summary["spec"]["weights"],
            "buy_score": summary["spec"]["buy_score"],
            "total_return": summary["total_return"],
            "sharpe": summary["sharpe"],
            "max_drawdown": summary["max_drawdown"],
            "n_trades": summary["trades"]["n_trades"],
            "win_rate": summary["trades"]["win_rate"],
        })
    return rows


def write_backtest_report(
    config: AppConfig, summary: dict, sweep_rows: list[dict] | None = None
) -> Path:
    spec = summary["spec"]
    report_dir = config.reports_dir / "backtests"
    report_dir.mkdir(parents=True, exist_ok=True)
    pct = lambda v: f"{v * 100:.2f}%" if v is not None else "n/a"  # noqa: E731

    lines = [
        f"# Backtest — {spec['name']}",
        "",
        f"Window: {spec['window']['start']} → {spec['window']['end']} "
        f"({spec['n_trading_days']} trading days)",
        "",
        "## Assumptions (read before trusting any number)",
        "",
        f"- Signals replayed: **{', '.join(spec['signals'])}** — fundamentals/news/SEC",
        "  are excluded as not point-in-time-safe (PLAN.md S2).",
        f"- Weights renormalized over enabled signals: {spec['weights']}",
        f"- Entry threshold (buy_score): {spec['buy_score']}",
        "- Risk-engine score floor tracks the entry threshold "
        f"({spec['buy_score']}).",
        "- Orders auto-approved; fills at next open with "
        f"{config.risk.fill_model.slippage_bps} bps slippage, "
        f"${config.risk.fill_model.commission_per_trade} commission.",
        "- Account math on raw closes (no dividends); benchmarks use adjusted",
        "  closes — the comparison flatters the benchmark.",
        "",
        "## Results",
        "",
        f"- Total return: **{pct(summary['total_return'])}**  |  "
        f"Annualized: {pct(summary['annualized_return'])}",
        f"- Sharpe: {summary['sharpe'] if summary['sharpe'] is not None else 'n/a'}  |  "
        f"Max drawdown: {pct(summary['max_drawdown'])}",
        f"- Trades: {summary['trades']['n_trades']} closed, "
        f"win rate {pct(summary['trades']['win_rate'])}, "
        f"total P&L ${summary['trades']['total_pnl']:,.2f}",
    ]
    for bench, b in summary.get("benchmarks", {}).items():
        lines.append(f"- {bench} buy-and-hold same window: {pct(b['total_return'])}")
    lines.append("")

    if sweep_rows:
        lines += [
            "## Sensitivity Sweep",
            "",
            "| Variant | Weights | Buy ≥ | Return | Sharpe | Max DD | Trades | Win rate |",
            "|---------|---------|------:|-------:|-------:|-------:|-------:|---------:|",
        ]
        for r in sweep_rows:
            weights = ", ".join(f"{k}={v:.2f}" for k, v in r["weights"].items())
            lines.append(
                f"| {r['variant']} | {weights} | {r['buy_score']:.0f} "
                f"| {pct(r['total_return'])} "
                f"| {r['sharpe'] if r['sharpe'] is not None else 'n/a'} "
                f"| {pct(r['max_drawdown'])} | {r['n_trades']} "
                f"| {pct(r['win_rate'])} |"
            )
        lines.append("")

    path = report_dir / f"{spec['name']}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (report_dir / f"{spec['name']}.json").write_text(
        json.dumps({"summary": summary, "sweep": sweep_rows}, indent=2, default=str),
        encoding="utf-8",
    )
    return path
