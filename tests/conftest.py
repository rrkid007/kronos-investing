import shutil
from pathlib import Path

import pytest

from trading_platform.core.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_config(tmp_path):
    """Real config files, but db and reports redirected into a temp dir."""
    config_dir = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", config_dir)
    settings = (config_dir / "settings.yaml").read_text(encoding="utf-8")
    settings = settings.replace(
        "db_path: db/investment_research.sqlite", "db_path: db/test.sqlite"
    )
    (config_dir / "settings.yaml").write_text(settings, encoding="utf-8")
    return load_config(config_dir)
