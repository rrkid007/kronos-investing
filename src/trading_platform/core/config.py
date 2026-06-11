"""Configuration loading and validation.

All runtime configuration lives in YAML files under config/. Code never
hardcodes watchlists, weights, or limits — changing any of those must never
require a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class LLMSettings(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:32b-instruct"
    timeout_seconds: int = 120


class KronosSettings(BaseModel):
    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    horizon_days: int = 10
    context_candles: int = 400  # input window; model max_context is 512
    max_context: int = 512
    sample_count: int = 8       # forecast paths per ticker; dispersion -> confidence
    temperature: float = 1.0
    top_p: float = 0.9


class NewsSettings(BaseModel):
    finnhub_api_key_env: str = "FINNHUB_API_KEY"
    rss_feeds: list[str] = Field(default_factory=list)
    lookback_days: int = 7
    max_items: int = 25


class SECSettings(BaseModel):
    # EDGAR requires a descriptive User-Agent with contact info.
    identity: str = "Peak Logic info@peaklogic.ai"
    max_risk_chars: int = 10_000
    max_mdna_chars: int = 6_000


class NotificationSettings(BaseModel):
    enabled: bool = False
    webhook_url: str = ""


class ScheduleSettings(BaseModel):
    run_after_close_local: str = "17:30"
    weekdays_only: bool = True


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"  # local-only by default; no auth layer
    port: int = 8420


class Settings(BaseModel):
    db_path: Path = Path("db/investment_research.sqlite")
    reports_dir: Path = Path("reports")
    timezone: str = "America/New_York"
    benchmarks: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    llm: LLMSettings = LLMSettings()
    kronos: KronosSettings = KronosSettings()
    news: NewsSettings = NewsSettings()
    sec: SECSettings = SECSettings()
    notifications: NotificationSettings = NotificationSettings()
    schedule: ScheduleSettings = ScheduleSettings()
    dashboard: DashboardSettings = DashboardSettings()


class WatchlistEntry(BaseModel):
    symbol: str
    sector: str


class Watchlist(BaseModel):
    tickers: list[WatchlistEntry]

    @field_validator("tickers")
    @classmethod
    def no_duplicate_symbols(cls, v: list[WatchlistEntry]) -> list[WatchlistEntry]:
        symbols = [t.symbol for t in v]
        dupes = {s for s in symbols if symbols.count(s) > 1}
        if dupes:
            raise ValueError(f"duplicate watchlist symbols: {sorted(dupes)}")
        return v

    @property
    def symbols(self) -> list[str]:
        return [t.symbol for t in self.tickers]

    def sector_of(self, symbol: str) -> str | None:
        for t in self.tickers:
            if t.symbol == symbol:
                return t.sector
        return None


class Thresholds(BaseModel):
    buy_score: float = 70.0
    watchlist_score: float = 60.0
    exit_score: float = 45.0
    # Minimum weighted signal coverage (sum of weight x confidence / sum of
    # weight) required to open a position or trigger a score-decay exit.
    min_signal_coverage: float = 0.25


class ExitPolicy(BaseModel):
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 20.0
    max_hold_days: int = 60


class Weights(BaseModel):
    signal_weights: dict[str, float]
    thresholds: Thresholds = Thresholds()
    exit_policy: ExitPolicy = ExitPolicy()

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "Weights":
        total = sum(self.signal_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"signal_weights must sum to 1.0, got {total}")
        return self


class PaperAccount(BaseModel):
    starting_cash: float = 100_000.0


class SizingSettings(BaseModel):
    base_position_pct: float = 10.0     # base position as % of equity
    min_position_value: float = 1000.0  # reject buys smaller than this
    target_vol: float = 0.25            # annualized; scales size inversely with vol
    max_positions: int = 8


class FillModel(BaseModel):
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0


class RiskLimits(BaseModel):
    max_position_pct: float = 15.0
    max_sector_pct: float = 40.0
    min_cash_reserve_pct: float = 10.0
    min_final_score: float = 70.0
    restricted_assets: list[str] = Field(default_factory=list)
    # Per-agent score floors for buys, e.g. {kronos: 25}. An agent that is
    # dead (zero confidence) or absent is NOT blocking — overall signal
    # quality is the decision layer's coverage floor's job.
    min_agent_scores: dict[str, float] = Field(default_factory=dict)
    require_human_approval: bool = True
    paper_account: PaperAccount = PaperAccount()
    sizing: SizingSettings = SizingSettings()
    fill_model: FillModel = FillModel()


class AppConfig(BaseModel):
    root: Path
    settings: Settings
    watchlist: Watchlist
    weights: Weights
    risk: RiskLimits

    @property
    def db_path(self) -> Path:
        p = self.settings.db_path
        return p if p.is_absolute() else self.root / p

    @property
    def reports_dir(self) -> Path:
        p = self.settings.reports_dir
        return p if p.is_absolute() else self.root / p


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path | str | None = None) -> AppConfig:
    """Load and validate all configuration. config_dir defaults to <repo>/config."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[3] / "config"
    config_dir = Path(config_dir)
    return AppConfig(
        root=config_dir.parent,
        settings=Settings(**_load_yaml(config_dir / "settings.yaml")),
        watchlist=Watchlist(**_load_yaml(config_dir / "watchlist.yaml")),
        weights=Weights(**_load_yaml(config_dir / "weights.yaml")),
        risk=RiskLimits(**_load_yaml(config_dir / "risk_limits.yaml")),
    )
