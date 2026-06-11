"""Print the live performance summary from the audit trail."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.analytics.performance import closed_trades, performance_summary
from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    parser.add_argument("--trades", action="store_true", help="list closed trades")
    args = parser.parse_args()

    config = load_config(args.config_dir)
    conn = connect(config.db_path)
    init_db(conn)

    summary = performance_summary(conn, config.settings.benchmarks)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        pct = lambda v: f"{v * 100:.2f}%" if v is not None else "n/a"  # noqa: E731
        print(f"snapshots: {summary['n_snapshots']}")
        print(f"total return: {pct(summary['total_return'])}  "
              f"annualized: {pct(summary['annualized_return'])}")
        print(f"sharpe: {summary['sharpe'] if summary['sharpe'] is not None else 'n/a'}  "
              f"max drawdown: {pct(summary['max_drawdown'])}")
        t = summary["trades"]
        print(f"trades: {t['n_trades']} closed, win rate {pct(t['win_rate'])}, "
              f"total P&L ${t['total_pnl']:,.2f}")
        for bench, b in (summary.get("benchmarks") or {}).items():
            print(f"{bench}: {pct(b['total_return'])} over the same window")
        if summary["signal_hit_rates"]:
            print("signal hit rates (10d forward):")
            for agent, s in sorted(summary["signal_hit_rates"].items()):
                print(f"  {agent}: {s['hits']}/{s['n_calls']} = {pct(s['hit_rate'])}")

    if args.trades:
        for t in closed_trades(conn):
            print(f"{t['exit_date']} {t['ticker']:7s} {t['qty']:g} "
                  f"{t['entry_price']:.2f} -> {t['exit_price']:.2f} "
                  f"pnl ${t['pnl']:,.2f} ({t['return_pct']:+.2f}%)")
    conn.close()


if __name__ == "__main__":
    main()
