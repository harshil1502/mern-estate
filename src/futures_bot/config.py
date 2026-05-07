from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Loaded from .env / process env. Never read directly into config files."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tradovate_username: str = ""
    tradovate_password: str = ""
    tradovate_app_id: str = "futures-bot"
    tradovate_app_version: str = "0.0.1"
    tradovate_cid: str = ""
    tradovate_secret: str = ""
    tradovate_env: Literal["demo", "live"] = "demo"
    log_level: str = "INFO"


class RunnerConfig(BaseModel):
    mode: Literal["paper", "live", "backtest"] = "paper"
    loop_interval_s: float = 5.0


class InstrumentConfig(BaseModel):
    symbol: str
    exchange: str = "CME"
    tick_size: float
    point_value: float


class StrategyConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class RiskConfig(BaseModel):
    max_position: int = 1
    per_trade_risk_usd: float = 50.0
    daily_loss_limit_usd: float = 200.0
    kill_switch_drawdown_usd: float = 500.0


class BacktestConfig(BaseModel):
    start: date
    end: date
    initial_equity_usd: float = 10_000.0


class SizingConfig(BaseModel):
    """Free-form sizer spec; `type` selects the implementation."""

    type: str = "fixed"
    # remaining keys are passed through to the sizer constructor
    model_config = {"extra": "allow"}


class AppConfig(BaseModel):
    runner: RunnerConfig
    instrument: InstrumentConfig
    strategy: StrategyConfig
    risk: RiskConfig
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    backtest: BacktestConfig | None = None


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return AppConfig.model_validate(raw)
