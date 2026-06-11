"""Refresh the price cache for the whole watchlist and report data quality."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.core.config import load_config
from trading_platform.core.db import connect, init_db
from trading_platform.core.logging_setup import setup_logging
from trading_platform.data.market_data import MarketDataService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None, help="config directory (default: <repo>/config)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)
    conn = connect(config.db_path)
    init_db(conn)
    service = MarketDataService(conn)

    failures = 0
    for symbol in config.watchlist.symbols:
        data = service.refresh_and_validate(symbol)
        status = "OK  " if data.ok else "FAIL"
        print(f"{status} {symbol:7s} {data.detail()}")
        if not data.ok:
            failures += 1
    conn.close()

    if failures:
        print(f"\n{failures} ticker(s) failed the data quality gate")
        sys.exit(1)
    print("\nall tickers passed the data quality gate")


if __name__ == "__main__":
    main()
