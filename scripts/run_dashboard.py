"""Launch the dashboard (FastAPI + HTMX). Local-only by default."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from trading_platform.core.config import load_config
from trading_platform.dashboard.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config_dir)
    host = args.host or config.settings.dashboard.host
    port = args.port or config.settings.dashboard.port
    print(f"dashboard: http://{host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
