from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path

import typer

from futures_bot.brokers.paper import PaperBroker
from futures_bot.brokers.tradovate import TradovateBroker
from futures_bot.config import Secrets, load_config
from futures_bot.data.csv_feed import csv_bar_feed, list_csv_bars
from futures_bot.journal import TradeJournal
from futures_bot.logging_setup import configure_logging
from futures_bot.runner.paper import PaperRunner
from futures_bot.strategies import build_strategy

app = typer.Typer(add_completion=False, help="Futures algo bot — Tradovate / CME scaffold.")
log = logging.getLogger("futures_bot")


@app.command()
def paper(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config YAML."),
    journal: Path | None = typer.Option(None, help="Optional JSONL fill-journal output path."),
) -> None:
    """Run the strategy live against the configured Tradovate (demo) broker."""
    secrets = Secrets()
    configure_logging(secrets.log_level)
    cfg = load_config(config)
    if cfg.runner.mode == "live" and secrets.tradovate_env != "live":
        raise typer.BadParameter("runner.mode=live but TRADOVATE_ENV != live")
    broker = TradovateBroker(secrets)
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)

    journal_ctx = TradeJournal(journal) if journal else nullcontext()
    with journal_ctx as j:
        on_signal = j if isinstance(j, TradeJournal) else None
        runner = PaperRunner(cfg, broker, strategy, on_signal=on_signal)
        asyncio.run(runner.run())


@app.command("paper-sim")
def paper_sim(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config YAML."),
    csv: Path = typer.Option(..., help="CSV with columns: ts,open,high,low,close,volume."),
    speed: float = typer.Option(0.0, help="0=as fast as possible; N=N×wall-clock pacing."),
    initial_equity: float = typer.Option(10_000.0, help="Starting paper equity (USD)."),
    slippage_ticks: float = typer.Option(0.5, help="Slippage applied to fills, in ticks."),
    commission: float = typer.Option(0.0, help="Per-contract commission in USD."),
    journal: Path = typer.Option(Path("paper-journal.jsonl"), help="JSONL fill-journal path."),
) -> None:
    """Run the strategy through the in-process PaperBroker against a CSV bar file."""
    secrets = Secrets()
    configure_logging(secrets.log_level)
    cfg = load_config(config)

    feed = csv_bar_feed(csv, speed_multiplier=speed)
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)

    with TradeJournal(journal) as j:
        broker = PaperBroker(
            instrument=cfg.instrument,
            bar_source=feed,
            initial_equity=initial_equity,
            slippage_ticks=slippage_ticks,
            commission_per_contract=commission,
            on_fill=j,
        )
        runner = PaperRunner(cfg, broker, strategy, on_signal=j)
        stats = asyncio.run(runner.run())

    typer.echo(f"\nPaper-sim done — {stats.summary()}")
    typer.echo(f"Journal: {journal}  ({len(broker.state.fills)} fills)")


@app.command()
def backtest(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config YAML."),
    csv: Path = typer.Option(..., help="CSV with columns: ts,open,high,low,close,volume."),
) -> None:
    """Run the strategy over a historical CSV bar file (in-process backtester)."""
    from futures_bot.backtest.engine import BacktestEngine

    configure_logging(Secrets().log_level)
    cfg = load_config(config)
    if cfg.backtest is None:
        raise typer.BadParameter("config.backtest section is required for backtest")

    bars = list_csv_bars(csv)
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
    from futures_bot.strategies.registry import _REGISTRY  # noqa: PLC2701

    configure_logging("INFO")
    if not _REGISTRY:
        typer.echo("(none)")
        return
    for name in sorted(_REGISTRY):
        typer.echo(name)


if __name__ == "__main__":
    app()
