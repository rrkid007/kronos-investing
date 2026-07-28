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

from trading_platform.service.app import create_service_app
from trading_platform.service.config import ServiceConfig, load_kronos_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None,
                        help="path to a settings.yaml holding the kronos block "
                             "(default <repo>/config/settings.yaml)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the startup forecast; the model then loads "
                             "lazily on the first paid request")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = ServiceConfig.from_env()
    if args.settings:
        cfg.settings_path = args.settings
    if args.no_warmup:
        cfg.warmup_on_startup = False

    host = args.host or cfg.host
    port = args.port or cfg.port

    # Only the `kronos:` block is read — watchlist/weights/risk_limits are the
    # trading pipeline's concern and must not be able to block this service.
    kronos_settings = load_kronos_settings(cfg.settings_path)

    app = create_service_app(kronos_settings=kronos_settings, config=cfg)

    mode = f"PAID on {cfg.network}" if app.state.payments_active else "FREE (payments disabled)"
    print(f"kronos service [{mode}] model={kronos_settings.model_id}")
    print(f"  http://{host}:{port}/v1/kronos/schema")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
