"""Initialize (or migrate) the SQLite database."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_platform.core.config import load_config
from trading_platform.core.db import SCHEMA_VERSION, connect, init_db
from trading_platform.core.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None, help="config directory (default: <repo>/config)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)
    conn = connect(config.db_path)
    init_db(conn)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    print(f"database ready at {config.db_path} (schema v{version}/{SCHEMA_VERSION})")


if __name__ == "__main__":
    main()
