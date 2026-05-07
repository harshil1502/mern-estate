from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import typer

from futures_bot.brokers.paper import PaperBroker
from futures_bot.brokers.tradovate import TradovateBroker
from futures_bot.config import Secrets, load_config
from futures_bot.data.csv_feed import csv_bar_feed, list_csv_bars, write_bars_csv
from futures_bot.data.synthetic import synthetic_bars
from futures_bot.execution.sizing import build_sizer
from futures_bot.journal import TradeJournal
from futures_bot.logging_setup import configure_logging
from futures_bot.research import (
    GoalSpec,
    ParamGrid,
    evaluate_goal,
    run_sweep,
    run_walk_forward,
)
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

    sizer = build_sizer(cfg.sizing.model_dump())
    journal_ctx = TradeJournal(journal) if journal else nullcontext()
    with journal_ctx as j:
        on_signal = j if isinstance(j, TradeJournal) else None
        runner = PaperRunner(cfg, broker, strategy, on_signal=on_signal, sizer=sizer)
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
    sizer = build_sizer(cfg.sizing.model_dump())

    with TradeJournal(journal) as j:
        broker = PaperBroker(
            instrument=cfg.instrument,
            bar_source=feed,
            initial_equity=initial_equity,
            slippage_ticks=slippage_ticks,
            commission_per_contract=commission,
            on_fill=j,
        )
        runner = PaperRunner(cfg, broker, strategy, on_signal=j, sizer=sizer)
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
    sizer = build_sizer(cfg.sizing.model_dump())
    engine = BacktestEngine(
        strategy, cfg.instrument, cfg.risk, cfg.backtest.initial_equity_usd, sizer=sizer,
    )
    result = engine.run(bars)
    log.info(
        "Backtest done: trades=%d realized_pnl=%.2f final_equity=%.2f max_dd=%.2f",
        result.num_trades, result.realized_pnl, result.final_equity, result.max_drawdown,
    )


@app.command("fetch-history")
def fetch_history(
    source: str = typer.Option("synth", help="Data source: yahoo | synth."),
    symbol: str = typer.Option("ES=F", help="Yahoo symbol (e.g. ES=F, MES=F, NQ=F)."),
    interval: str = typer.Option("1m", help="Yahoo interval: 1m|5m|15m|1h|1d|..."),
    period: str = typer.Option("5d", help="Yahoo period: 1d|5d|1mo|3mo|6mo|1y|max."),
    out: Path = typer.Option(Path("data/history.csv"), help="Output CSV path."),
    n: int = typer.Option(500, help="(synth) number of bars."),
    bar_seconds: int = typer.Option(60, help="(synth) bar duration in seconds."),
    start_price: float = typer.Option(4500.0, help="(synth) opening price."),
    annual_vol: float = typer.Option(0.16, help="(synth) annualized volatility."),
    seed: int = typer.Option(42, help="(synth) PRNG seed."),
) -> None:
    """Fetch historical bars and write a CSV the bot can replay.

    `--source synth` is fully offline and deterministic; `--source yahoo`
    pulls real data from Yahoo Finance (rate-limited; not exchange quality).
    """
    configure_logging(Secrets().log_level)
    if source == "synth":
        bars = synthetic_bars(
            n=n, start_price=start_price, bar_seconds=bar_seconds,
            annual_vol=annual_vol, seed=seed,
        )
    elif source == "yahoo":
        from futures_bot.data.yahoo import fetch_yahoo_bars
        bars = fetch_yahoo_bars(symbol, interval=interval, period=period)
    else:
        raise typer.BadParameter(f"unknown source '{source}' (use yahoo|synth)")

    rows = write_bars_csv(out, bars)
    typer.echo(f"Wrote {rows} bars to {out}")


@app.command()
def research(
    config: Path = typer.Option(Path("config/config.yaml"), help="Bot config (instrument/risk)."),
    grid: Path = typer.Option(
        Path("configs/research_grid.example.yaml"),
        help="Strategy grid YAML (see configs/research_grid.example.yaml).",
    ),
    csv: Path = typer.Option(..., help="CSV of bars to evaluate against."),
    account_size: float = typer.Option(150_000.0, help="Account size, USD."),
    target_daily_pnl: float = typer.Option(500.0, help="Target daily PnL, USD."),
    folds: int = typer.Option(5, help="Walk-forward folds (1 = single sweep, no WF)."),
    train_frac: float = typer.Option(0.7, help="In-sample fraction per fold."),
    select_by: str = typer.Option("sharpe", help="IS selection metric: sharpe|sortino|calmar|pnl."),
    top: int = typer.Option(5, help="Top-N to print per strategy in single-sweep mode."),
    sizer: str = typer.Option(
        "",
        help="Override sizer, e.g. 'fixed:qty=1' or 'atr:risk_per_trade_pct=0.005,atr_period=14'.",
    ),
    out_json: Path | None = typer.Option(None, help="Optional JSON dump of full report."),
) -> None:
    """Search strategies × params and report what (if anything) clears the daily-PnL goal."""
    import json

    import yaml

    configure_logging(Secrets().log_level)
    cfg = load_config(config)
    sizer_spec = _parse_sizer_override(sizer) if sizer else cfg.sizing.model_dump()
    sizer_factory = lambda: build_sizer(sizer_spec)  # noqa: E731 — fresh sizer per combo
    typer.echo(f"Sizer: {sizer_spec}")
    raw = yaml.safe_load(grid.read_text()) or {}
    if not isinstance(raw, dict) or not raw:
        raise typer.BadParameter(f"grid file {grid} is empty or malformed")

    bars = list_csv_bars(csv)
    if len(bars) < 100:
        raise typer.BadParameter(f"need >= 100 bars, got {len(bars)} in {csv}")

    goal = GoalSpec(target_daily_pnl_usd=target_daily_pnl, account_size_usd=account_size)
    typer.echo(
        f"\nGoal: ${target_daily_pnl:.0f}/day on ${account_size:,.0f} "
        f"= {goal.required_daily_return * 100:.3f}%/day "
        f"= {goal.required_annualized_return * 100:.0f}%/year"
    )
    typer.echo(f"Bars: {len(bars)}  ({bars[0].ts.date()} -> {bars[-1].ts.date()})\n")

    full_report: dict[str, Any] = {
        "goal": {
            "target_daily_pnl": target_daily_pnl,
            "account_size": account_size,
            "required_daily_return": goal.required_daily_return,
            "required_annualized_return": goal.required_annualized_return,
        },
        "n_bars": len(bars),
        "strategies": {},
    }

    for strategy_name, spec in raw.items():
        param_grid = ParamGrid(
            strategy_name=strategy_name,
            params=spec.get("params", {}) or {},
            fixed=spec.get("fixed", {}) or {},
        )
        typer.echo(f"=== {strategy_name} ({param_grid.size()} combos) ===")

        if folds <= 1:
            report = run_sweep(
                param_grid, bars, cfg.instrument, cfg.risk, account_size,
                sizer_factory=sizer_factory,
            )
            top_n = report.top(top, key=select_by)
            for r in top_n:
                _print_sweep_row(r, goal)
            full_report["strategies"][strategy_name] = {
                "mode": "single_sweep",
                "top": [
                    {"params": r.params, **r.metrics.as_row()} for r in top_n
                ],
            }
        else:
            wf = run_walk_forward(
                param_grid, bars, cfg.instrument, cfg.risk, account_size,
                n_folds=folds, train_frac=train_frac, select_by=select_by,
                sizer_factory=sizer_factory,
            )
            _print_walk_forward(wf, goal)
            full_report["strategies"][strategy_name] = {
                "mode": "walk_forward",
                "n_folds": len(wf.folds),
                "aggregated_oos_pnl": wf.aggregated_oos_pnl,
                "aggregated_oos_trades": wf.aggregated_oos_trades,
                "folds": [
                    {
                        "fold": f.fold,
                        "params": f.best_params,
                        "is_metrics": f.is_metrics.as_row(),
                        "oos_metrics": f.oos_metrics.as_row(),
                    } for f in wf.folds
                ],
            }
        typer.echo("")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(full_report, indent=2, default=str))
        typer.echo(f"Full report written to {out_json}")


def _parse_sizer_override(spec: str) -> dict[str, Any]:
    """Parse 'atr:risk_per_trade_pct=0.005,atr_period=14' into a sizer dict."""
    if ":" not in spec:
        return {"type": spec}
    type_part, rest = spec.split(":", 1)
    out: dict[str, Any] = {"type": type_part}
    for kv in rest.split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        try:
            out[k.strip()] = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def _print_sweep_row(r: Any, goal: GoalSpec) -> None:
    m = r.metrics
    assessment = evaluate_goal(m, goal)
    flag = "✓" if assessment.median_meets_target else "✗"
    typer.echo(
        f"  {flag} params={r.params}  "
        f"pnl={m.realized_pnl:+.2f}  sharpe={m.sharpe_annualized:.2f}  "
        f"trades={m.num_trades}  win={m.win_rate:.0%}  "
        f"med_day={m.median_daily_pnl:+.2f}  max_dd={m.max_drawdown:.2f}"
    )


def _print_walk_forward(wf: Any, goal: GoalSpec) -> None:
    if not wf.folds:
        typer.echo("  (no folds completed)")
        return
    total_pnl = wf.aggregated_oos_pnl
    avg_oos_sharpe = sum(f.oos_metrics.sharpe_annualized for f in wf.folds) / len(wf.folds)
    avg_oos_med_daily = sum(f.oos_metrics.median_daily_pnl for f in wf.folds) / len(wf.folds)
    typer.echo(
        f"  Walk-forward over {len(wf.folds)} folds, select_by={wf.select_by}"
    )
    for f in wf.folds:
        typer.echo(
            f"    fold {f.fold}: params={f.best_params}  "
            f"OOS pnl={f.oos_metrics.realized_pnl:+.2f}  "
            f"sharpe={f.oos_metrics.sharpe_annualized:.2f}  "
            f"trades={f.oos_metrics.num_trades}  "
            f"med_day={f.oos_metrics.median_daily_pnl:+.2f}"
        )
    flag = "✓" if avg_oos_med_daily >= goal.target_daily_pnl_usd else "✗"
    typer.echo(
        f"  {flag} aggregated OOS pnl={total_pnl:+.2f}  "
        f"avg sharpe={avg_oos_sharpe:.2f}  avg med_day={avg_oos_med_daily:+.2f}  "
        f"(target ${goal.target_daily_pnl_usd:.0f})"
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
