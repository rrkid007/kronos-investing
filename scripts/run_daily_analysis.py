"""Run the daily analysis pipeline (entry point for the scheduler)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.core.config import load_config
from trading_platform.core.logging_setup import setup_logging
from trading_platform.pipeline import run_daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="run date YYYY-MM-DD (default: today)")
    parser.add_argument("--config-dir", default=None, help="config directory (default: <repo>/config)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)
    run_id = run_daily(config, run_date=args.date)
    print(f"run complete: {run_id}")


if __name__ == "__main__":
    main()
