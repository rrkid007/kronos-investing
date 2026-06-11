"""Run a historical replay backtest (technical-only by default; PIT-safe).

Examples:
  python scripts/run_backtest.py --years 2
  python scripts/run_backtest.py --start 2024-06-01 --end 2026-06-01 --sweep
  python scripts/run_backtest.py --signals technical,kronos   # slow: GPU advised
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.backtest.replay import (
    BacktestSpec,
    load_frames,
    run_backtest,
    sensitivity_sweep,
    write_backtest_report,
)
from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.core.logging_setup import setup_logging
from trading_platform.data.market_data import MarketDataService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--years", type=float, default=2.0,
                        help="window length when --start omitted (default 2)")
    parser.add_argument("--signals", default="technical",
                        help="comma list of PIT-safe signals (technical[,kronos])")
    parser.add_argument("--sweep", action="store_true",
                        help="also run the sensitivity sweep grid")
    parser.add_argument("--name", default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download history first (longer lookback)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=int(args.years * 365.25)))
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    name = args.name or f"bt-{start}-{end}-{'-'.join(signals)}"

    conn = connect(config.db_path)
    init_db(conn)
    if args.refresh:
        # window + 250-bar warmup + buffer
        lookback = (end - start).days + 600
        service = MarketDataService(conn, lookback_days=lookback)
        for ticker in config.watchlist.symbols + config.settings.benchmarks:
            result = service.refresh(ticker, as_of=end)
            status = "ok" if not result.error else f"ERROR {result.error}"
            print(f"refresh {ticker:7s} {status}")

    frames = load_frames(conn, config.watchlist.symbols + config.settings.benchmarks)
    conn.close()

    spec = BacktestSpec(name=name, start=start, end=end, signals=signals)
    out_dir = config.root / "data" / "backtests"

    print(f"replaying {start} -> {end} with signals {signals} ...")
    summary = run_backtest(config, spec, frames, out_dir / f"{name}.sqlite")

    sweep_rows = None
    if args.sweep:
        print("running sensitivity sweep ...")
        sweep_rows = sensitivity_sweep(config, spec, frames, out_dir)

    report = write_backtest_report(config, summary, sweep_rows)
    pct = lambda v: f"{v * 100:.2f}%" if v is not None else "n/a"  # noqa: E731
    print(f"\ntotal return {pct(summary['total_return'])} | "
          f"sharpe {summary['sharpe']} | max dd {pct(summary['max_drawdown'])} | "
          f"{summary['trades']['n_trades']} trades "
          f"(win rate {pct(summary['trades']['win_rate'])})")
    for bench, b in summary["benchmarks"].items():
        print(f"{bench}: {pct(b['total_return'])} buy-and-hold")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
