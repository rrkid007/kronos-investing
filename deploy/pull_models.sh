#!/usr/bin/env bash
# Prefetch all models so the first scheduled run doesn't pay download time.
set -euo pipefail

echo "==> pulling LLM (qwen2.5:32b-instruct, ~20GB)"
ollama pull qwen2.5:32b-instruct

echo "==> prefetching Kronos weights from HuggingFace"
uv run python - <<'EOF'
from trading_platform.core.config import load_config
from trading_platform.forecast.kronos_forecaster import KronosForecaster

config = load_config()
forecaster = KronosForecaster(config.settings.kronos)
forecaster._load()
print(f"kronos ready: {config.settings.kronos.model_id} on {forecaster.device} "
      f"(loaded in {forecaster.load_seconds}s)")
EOF

echo "==> done"
