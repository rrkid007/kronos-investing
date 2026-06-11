from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_platform.core.config import Watchlist, Weights, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_config_loads_and_validates():
    config = load_config(REPO_ROOT / "config")
    assert len(config.watchlist.symbols) == 10
    assert "AAPL" in config.watchlist.symbols
    assert config.watchlist.sector_of("BRK-B") == "Financials"
    assert config.settings.kronos.horizon_days == 10
    assert config.settings.llm.model.startswith("qwen2.5")
    assert config.risk.require_human_approval is True
    assert config.risk.paper_account.starting_cash == 100_000.0


def test_db_path_resolves_relative_to_root():
    config = load_config(REPO_ROOT / "config")
    assert config.db_path.is_absolute()
    assert config.db_path == REPO_ROOT / "db" / "investment_research.sqlite"


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Weights(signal_weights={"technical": 0.5, "kronos": 0.4})


def test_watchlist_rejects_duplicates():
    with pytest.raises(ValidationError, match="duplicate"):
        Watchlist(
            tickers=[
                {"symbol": "AAPL", "sector": "Technology"},
                {"symbol": "AAPL", "sector": "Technology"},
            ]
        )
