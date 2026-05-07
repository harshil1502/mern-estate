from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from futures_bot.brokers.tradovate import TradovateBroker
from futures_bot.config import Secrets, load_config
from futures_bot.logging_setup import configure_logging
from futures_bot.runner.paper import PaperRunner
from futures_bot.strategies import build_strategy

app = typer.Typer(add_completion=False, help="Futures algo bot — Tradovate / CME scaffold.")
log = logging.getLogger("futures_bot")


@app.command()
def paper(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config YAML."),
) -> None:
    """Run the strategy live against the configured (demo) broker."""
    secrets = Secrets()
    configure_logging(secrets.log_level)
    cfg = load_config(config)
    if cfg.runner.mode == "live" and secrets.tradovate_env != "live":
        raise typer.BadParameter("runner.mode=live but TRADOVATE_ENV != live")
    broker = TradovateBroker(secrets)
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    runner = PaperRunner(cfg, broker, strategy)
    asyncio.run(runner.run())


@app.command()
def backtest(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config YAML."),
    csv: Path = typer.Option(..., help="CSV with columns: ts,open,high,low,close,volume."),
) -> None:
    """Run the strategy over a historical CSV bar file."""
    import pandas as pd

    from futures_bot.backtest.engine import BacktestEngine
    from futures_bot.types import Bar

    configure_logging(Secrets().log_level)
    cfg = load_config(config)
    if cfg.backtest is None:
        raise typer.BadParameter("config.backtest section is required for backtest")

    df = pd.read_csv(csv, parse_dates=["ts"])
    bars = [
        Bar(ts=row.ts.to_pydatetime(),
            open=float(row.open), high=float(row.high), low=float(row.low),
            close=float(row.close), volume=float(row.volume))
        for row in df.itertuples(index=False)
    ]
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    engine = BacktestEngine(strategy, cfg.instrument, cfg.risk, cfg.backtest.initial_equity_usd)
    result = engine.run(bars)
    log.info(
        "Backtest done: trades=%d realized_pnl=%.2f final_equity=%.2f max_dd=%.2f",
        result.num_trades, result.realized_pnl, result.final_equity, result.max_drawdown,
    )


@app.command()
def strategies() -> None:
    """List registered strategies."""
    # importing the package side-effect-registers built-ins
    from futures_bot.strategies.registry import _REGISTRY  # noqa: PLC2701

    configure_logging("INFO")
    if not _REGISTRY:
        typer.echo("(none)")
        return
    for name in sorted(_REGISTRY):
        typer.echo(name)


if __name__ == "__main__":
    app()
