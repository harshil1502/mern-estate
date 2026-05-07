"""Run a strategy/param grid against a fixed set of bars and rank the results."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from futures_bot.backtest.engine import BacktestEngine
from futures_bot.config import InstrumentConfig, RiskConfig
from futures_bot.execution.sizing import FixedQty, PositionSizer
from futures_bot.research.grid import ParamGrid
from futures_bot.research.metrics import StrategyMetrics, compute_metrics
from futures_bot.strategies.registry import build_strategy
from futures_bot.types import Bar

log = logging.getLogger(__name__)


@dataclass
class SweepResult:
    strategy_name: str
    params: dict[str, Any]
    metrics: StrategyMetrics
    error: str = ""


@dataclass
class SweepReport:
    results: list[SweepResult] = field(default_factory=list)

    def top(self, n: int = 10, key: str = "sharpe") -> list[SweepResult]:
        ok = [r for r in self.results if not r.error]
        ok.sort(key=lambda r: _metric_value(r.metrics, key), reverse=True)
        return ok[:n]


def run_sweep(
    grid: ParamGrid,
    bars: list[Bar],
    instrument: InstrumentConfig,
    risk: RiskConfig,
    initial_equity: float,
    sizer_factory: Callable[[], PositionSizer] | None = None,
) -> SweepReport:
    """Run a grid against bars; `sizer_factory` builds a fresh sizer per combo."""
    report = SweepReport()
    n_combos = grid.size()
    log.info("Sweeping %s × %d combos against %d bars",
             grid.strategy_name, n_combos, len(bars))
    factory = sizer_factory or (lambda: FixedQty(1))
    for i, params in enumerate(grid.expand(), start=1):
        try:
            strategy = build_strategy(grid.strategy_name, params)
            engine = BacktestEngine(strategy, instrument, risk, initial_equity, sizer=factory())
            result = engine.run(bars)
            metrics = compute_metrics(result, bars, initial_equity)
            report.results.append(
                SweepResult(strategy_name=grid.strategy_name, params=params, metrics=metrics)
            )
        except Exception as e:
            log.warning("combo %d failed (%s): %s", i, params, e)
            report.results.append(
                SweepResult(
                    strategy_name=grid.strategy_name,
                    params=params,
                    metrics=_zero_metrics(),
                    error=str(e),
                )
            )
    return report


def _metric_value(metrics: StrategyMetrics, key: str) -> float:
    """Map a sort key to a metric value, defaulting to Sharpe.

    Zero-trade strategies are ranked below any strategy that actually traded,
    even a losing one. Otherwise they tie at 0 with all the no-signal variants
    and silently win the sweep, which is degenerate.
    """
    if metrics.num_trades == 0:
        return float("-inf")
    return {
        "sharpe": metrics.sharpe_annualized,
        "sortino": metrics.sortino_annualized,
        "calmar": metrics.calmar,
        "pnl": metrics.realized_pnl,
        "profit_factor": metrics.profit_factor if metrics.profit_factor != float("inf") else 1e9,
        "win_rate": metrics.win_rate,
        "median_daily_pnl": metrics.median_daily_pnl,
    }.get(key, metrics.sharpe_annualized)


def _zero_metrics() -> StrategyMetrics:
    return StrategyMetrics(
        final_equity=0.0,
        realized_pnl=0.0,
        max_drawdown=0.0,
        num_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        sharpe_annualized=0.0,
        sortino_annualized=0.0,
        calmar=0.0,
        median_daily_pnl=0.0,
        mean_daily_pnl=0.0,
        std_daily_pnl=0.0,
        pct_winning_days=0.0,
        n_trading_days=0,
    )
