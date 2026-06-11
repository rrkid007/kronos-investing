"""Discovery: screen the S&P 500 universe for watchlist candidates.

Weekly cadence is plenty — watchlist churn should be slow.

  python scripts/discover_stocks.py                 # run a discovery batch
  python scripts/discover_stocks.py --add NVO       # approve a suggestion
  python scripts/discover_stocks.py --dismiss NVO   # dismiss a suggestion
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.core.logging_setup import setup_logging
from trading_platform.data.market_data import MarketDataService
from trading_platform.data.universe import refresh_universe
from trading_platform.discovery.screener import screen, weakest_incumbent


def run_discovery(conn, config) -> None:
    n = refresh_universe(conn, max_age_days=config.settings.discovery.universe_max_age_days)
    print(f"universe: {n} members")
    print(f"screening up to {config.settings.discovery.max_candidates} candidates "
          f"(fundamentals floor {config.settings.discovery.fundamentals_floor}) ...")
    suggestions = screen(conn, config, MarketDataService(conn))

    if not suggestions:
        print("no candidates cleared the screen this batch")
    else:
        print(f"\n{'ticker':7s} {'sector':24s} {'tech':>5s} {'fund':>5s} "
              f"{'bonus':>5s} {'total':>6s}")
        for s in suggestions:
            print(f"{s.ticker:7s} {s.sector:24s} {s.technical_score:5.1f} "
                  f"{s.fundamental_score:5.1f} {s.sector_gap_bonus:5.1f} "
                  f"{s.combined_score:6.1f}")
            print(f"        {s.rationale}")

    weakest = weakest_incumbent(conn, config)
    if (weakest and
            len(config.watchlist.symbols) >= config.settings.discovery.max_watchlist_size):
        print(f"\nwatchlist at capacity — weakest incumbent: {weakest['ticker']} "
              f"(avg final score {weakest['avg_final_score']} over {weakest['n_runs']} runs)")
    elif weakest:
        print(f"\nweakest incumbent (FYI): {weakest['ticker']} "
              f"(avg final score {weakest['avg_final_score']})")

    _write_report(config, suggestions, weakest)
    print("\napprove with: python scripts/discover_stocks.py --add TICKER")


def _write_report(config, suggestions, weakest) -> None:
    report_dir = config.reports_dir / "discovery"
    report_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).date().isoformat()
    lines = [
        f"# Watchlist Discovery — {today}", "",
        "Suggestions only — nothing changes until a human runs `--add`.", "",
    ]
    if suggestions:
        lines += [
            "| Ticker | Sector | Technical | Fundamentals | Gap bonus | Combined |",
            "|--------|--------|----------:|-------------:|----------:|---------:|",
        ]
        lines += [
            f"| {s.ticker} | {s.sector} | {s.technical_score:.1f} "
            f"| {s.fundamental_score:.1f} | +{s.sector_gap_bonus:.0f} "
            f"| **{s.combined_score:.1f}** |"
            for s in suggestions
        ]
        lines += [""] + [f"- **{s.ticker}** — {s.rationale}" for s in suggestions]
    else:
        lines += ["No candidates cleared the screen this batch."]
    if weakest:
        lines += ["", f"Weakest incumbent: **{weakest['ticker']}** "
                  f"(avg final score {weakest['avg_final_score']} "
                  f"over {weakest['n_runs']} runs)"]
    (report_dir / f"{today}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_to_watchlist(conn, config, config_dir: Path, ticker: str) -> None:
    ticker = ticker.upper()
    if ticker in config.watchlist.symbols:
        sys.exit(f"{ticker} is already on the watchlist")
    if len(config.watchlist.symbols) >= config.settings.discovery.max_watchlist_size:
        weakest = weakest_incumbent(conn, config)
        hint = f" (weakest incumbent: {weakest['ticker']})" if weakest else ""
        sys.exit(f"watchlist at max size "
                 f"({config.settings.discovery.max_watchlist_size}) — remove a "
                 f"member from config/watchlist.yaml first{hint}")

    row = conn.execute(
        "SELECT sector FROM watchlist_suggestions WHERE ticker = ? "
        "UNION SELECT sector FROM universe WHERE ticker = ? LIMIT 1",
        (ticker, ticker),
    ).fetchone()
    if row is None:
        sys.exit(f"{ticker} not found in suggestions or universe — add manually "
                 f"to config/watchlist.yaml if intended")

    path = config_dir / "watchlist.yaml"
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += f"  - symbol: {ticker}\n    sector: {row['sector']}\n"
    path.write_text(text, encoding="utf-8")

    conn.execute(
        "UPDATE watchlist_suggestions SET status = 'added' WHERE ticker = ?", (ticker,)
    )
    conn.commit()
    print(f"added {ticker} ({row['sector']}) to {path}")
    print("it will be scored from the next daily run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--add", metavar="TICKER")
    parser.add_argument("--dismiss", metavar="TICKER")
    parser.add_argument("--force-universe", action="store_true",
                        help="refresh universe even if fresh")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)
    config_dir = Path(args.config_dir) if args.config_dir else config.root / "config"
    conn = connect(config.db_path)
    init_db(conn)

    if args.add:
        add_to_watchlist(conn, config, config_dir, args.add)
    elif args.dismiss:
        conn.execute(
            "UPDATE watchlist_suggestions SET status = 'dismissed' WHERE ticker = ?",
            (args.dismiss.upper(),),
        )
        conn.commit()
        print(f"dismissed {args.dismiss.upper()}")
    else:
        if args.force_universe:
            refresh_universe(conn, force=True)
        run_discovery(conn, config)
    conn.close()


if __name__ == "__main__":
    main()
