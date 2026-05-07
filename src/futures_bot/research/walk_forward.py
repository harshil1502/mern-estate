"""Anchored walk-forward analysis.

Splits bars into K folds. For each fold:
  - in-sample window  = [0, fold_end - test_size]
  - out-of-sample     = [fold_end - test_size, fold_end]

We sweep the param grid on the in-sample window, pick the best params by
`select_by` metric, then evaluate those params on the out-of-sample window.
The OOS score is the honest reading — IS optimization is biased by definition.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from futures_bot.config import InstrumentConfig, RiskConfig
from futures_bot.execution.sizing import PositionSizer
from futures_bot.research.grid import ParamGrid
from futures_bot.research.metrics import StrategyMetrics
from futures_bot.research.sweep import run_sweep
from futures_bot.types import Bar

log = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold: int
    is_range: tuple[datetime, datetime]
    oos_range: tuple[datetime, datetime]
    best_params: dict[str, Any]
    is_metrics: StrategyMetrics
    oos_metrics: StrategyMetrics


@dataclass
class WalkForwardResult:
    strategy_name: str
    select_by: str
    folds: list[WalkForwardFold]

    @property
    def aggregated_oos_pnl(self) -> float:
        return sum(f.oos_metrics.realized_pnl for f in self.folds)

    @property
    def aggregated_oos_trades(self) -> int:
        return sum(f.oos_metrics.num_trades for f in self.folds)


def run_walk_forward(
    grid: ParamGrid,
    bars: list[Bar],
    instrument: InstrumentConfig,
    risk: RiskConfig,
    initial_equity: float,
    n_folds: int = 5,
    train_frac: float = 0.7,
    select_by: str = "sharpe",
    sizer_factory: Callable[[], PositionSizer] | None = None,
) -> WalkForwardResult:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if len(bars) < n_folds * 50:
        raise ValueError(f"need >= {n_folds * 50} bars for {n_folds} folds, got {len(bars)}")

    folds: list[WalkForwardFold] = []
    fold_size = len(bars) // n_folds

    for fold_idx in range(n_folds):
        end = (fold_idx + 1) * fold_size
        if fold_idx == n_folds - 1:
            end = len(bars)  # absorb remainder into last fold
        window = bars[:end]
        train_end = int(len(window) * train_frac)
        is_bars = window[:train_end]
        oos_bars = window[train_end:]
        if not is_bars or not oos_bars:
            continue

        is_report = run_sweep(
            grid, is_bars, instrument, risk, initial_equity, sizer_factory=sizer_factory,
        )
        top_in_sample = is_report.top(1, key=select_by)
        if not top_in_sample:
            log.warning("fold %d: no valid IS results — skipping", fold_idx)
            continue
        best = top_in_sample[0]

        # Re-run the winning params on OOS only.
        oos_grid = ParamGrid(strategy_name=grid.strategy_name, fixed=dict(best.params))
        oos_report = run_sweep(
            oos_grid, oos_bars, instrument, risk, initial_equity, sizer_factory=sizer_factory,
        )
        oos = oos_report.results[0]

        folds.append(
            WalkForwardFold(
                fold=fold_idx,
                is_range=(is_bars[0].ts, is_bars[-1].ts),
                oos_range=(oos_bars[0].ts, oos_bars[-1].ts),
                best_params=best.params,
                is_metrics=best.metrics,
                oos_metrics=oos.metrics,
            )
        )
        log.info(
            "fold %d: IS sharpe=%.2f -> OOS sharpe=%.2f pnl=%.2f trades=%d (params=%s)",
            fold_idx, best.metrics.sharpe_annualized,
            oos.metrics.sharpe_annualized, oos.metrics.realized_pnl,
            oos.metrics.num_trades, best.params,
        )

    return WalkForwardResult(strategy_name=grid.strategy_name, select_by=select_by, folds=folds)
