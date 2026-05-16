"""Strategy-evaluation metrics computed from a BacktestResult.

The numbers here are the truth-tape — what the research loop ranks on. Every
metric is computed defensively (NaN/zero-safe) so the sweep doesn't crash on
degenerate strategies that never trade.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from futures_bot.backtest.engine import BacktestResult
from futures_bot.types import Bar

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass
class StrategyMetrics:
    final_equity: float
    realized_pnl: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    sharpe_annualized: float
    sortino_annualized: float
    calmar: float
    median_daily_pnl: float
    mean_daily_pnl: float
    std_daily_pnl: float
    pct_winning_days: float
    n_trading_days: int
    daily_pnls: list[tuple[date, float]] = field(default_factory=list)

    def as_row(self) -> dict[str, float | int]:
        return {
            "final_equity": self.final_equity,
            "realized_pnl": self.realized_pnl,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe": self.sharpe_annualized,
            "sortino": self.sortino_annualized,
            "calmar": self.calmar,
            "median_daily_pnl": self.median_daily_pnl,
            "mean_daily_pnl": self.mean_daily_pnl,
            "std_daily_pnl": self.std_daily_pnl,
            "pct_winning_days": self.pct_winning_days,
            "n_trading_days": self.n_trading_days,
        }


def compute_metrics(
    result: BacktestResult,
    bars: list[Bar],
    initial_equity: float,
) -> StrategyMetrics:
    daily = _daily_pnls(result)
    daily_values = [pnl for _, pnl in daily]

    bars_per_year = _annualization_factor(bars)
    bar_returns = _bar_returns(result.equity_curve)

    sharpe = _annualized_sharpe(bar_returns, bars_per_year)
    sortino = _annualized_sortino(bar_returns, bars_per_year)

    annualized_return = _annualized_return(initial_equity, result.final_equity, bars, bars_per_year)
    calmar = annualized_return / result.max_drawdown if result.max_drawdown > 0 else 0.0

    win_rate, profit_factor = _trade_stats(result)

    return StrategyMetrics(
        final_equity=result.final_equity,
        realized_pnl=result.realized_pnl,
        max_drawdown=result.max_drawdown,
        num_trades=result.num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_annualized=sharpe,
        sortino_annualized=sortino,
        calmar=calmar,
        median_daily_pnl=statistics.median(daily_values) if daily_values else 0.0,
        mean_daily_pnl=statistics.fmean(daily_values) if daily_values else 0.0,
        std_daily_pnl=statistics.pstdev(daily_values) if len(daily_values) >= 2 else 0.0,
        pct_winning_days=(
            sum(1 for v in daily_values if v > 0) / len(daily_values) if daily_values else 0.0
        ),
        n_trading_days=len(daily_values),
        daily_pnls=daily,
    )


# --- internals ----------------------------------------------------------------


def _daily_pnls(result: BacktestResult) -> list[tuple[date, float]]:
    by_day: dict[date, float] = defaultdict(float)
    for trade in result.trades:
        by_day[trade.ts.date()] += trade.realized_pnl
    return sorted(by_day.items())


def _bar_returns(curve: list[tuple[object, float]]) -> list[float]:
    if len(curve) < 2:
        return []
    returns: list[float] = []
    prev = curve[0][1]
    for _, eq in curve[1:]:
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((eq - prev) / prev)
        prev = eq
    return returns


def _annualization_factor(bars: list[Bar]) -> float:
    """Bars-per-year, derived from median bar gap. Robust to weekend gaps."""
    if len(bars) < 2:
        return 0.0
    gaps = [
        (bars[i].ts - bars[i - 1].ts).total_seconds()
        for i in range(1, len(bars))
        if (bars[i].ts - bars[i - 1].ts).total_seconds() > 0
    ]
    if not gaps:
        return 0.0
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return 0.0
    return _SECONDS_PER_YEAR / median_gap


def _annualized_sharpe(returns: list[float], bars_per_year: float) -> float:
    if len(returns) < 2 or bars_per_year <= 0:
        return 0.0
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(bars_per_year)


def _annualized_sortino(returns: list[float], bars_per_year: float) -> float:
    if len(returns) < 2 or bars_per_year <= 0:
        return 0.0
    mean = statistics.fmean(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    downside_std = math.sqrt(sum(r * r for r in downside) / len(returns))
    if downside_std == 0:
        return 0.0
    return (mean / downside_std) * math.sqrt(bars_per_year)


def _annualized_return(
    initial: float, final: float, bars: list[Bar], bars_per_year: float
) -> float:
    if initial <= 0 or len(bars) < 2 or bars_per_year <= 0:
        return 0.0
    elapsed = (bars[-1].ts - bars[0].ts).total_seconds()
    if elapsed <= 0:
        return 0.0
    years = elapsed / _SECONDS_PER_YEAR
    if years <= 0:
        return 0.0
    growth = final / initial
    if growth <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def _trade_stats(result: BacktestResult) -> tuple[float, float]:
    realized = [t.realized_pnl for t in result.trades if t.realized_pnl != 0]
    if not realized:
        return 0.0, 0.0
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    win_rate = len(wins) / len(realized) if realized else 0.0
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    return win_rate, profit_factor
