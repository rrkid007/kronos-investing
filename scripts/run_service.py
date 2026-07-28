"""Launch the paid Kronos forecast service (x402). Local-only by default.

Payments are off unless X402_ENABLED=1 and X402_PAY_TO are set — see
src/trading_platform/service/README.md.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from trading_platform.core.config import load_config
from trading_platform.service.app import create_service_app
from trading_platform.service.config import ServiceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = ServiceConfig.from_env()
    host = args.host or cfg.host
    port = args.port or cfg.port

    # Kronos model settings come from the platform config so the service and
    # the pipeline forecast with the identical model; everything payment-related
    # comes from the environment.
    kronos_settings = load_config(args.config_dir).settings.kronos

    app = create_service_app(kronos_settings=kronos_settings, config=cfg)

    mode = f"PAID on {cfg.network}" if app.state.payments_active else "FREE (payments disabled)"
    print(f"kronos service [{mode}]: http://{host}:{port}/v1/kronos/schema")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
