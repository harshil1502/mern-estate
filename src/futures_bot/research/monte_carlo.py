"""Monte Carlo bootstrap of strategy results.

Given a backtest's realized trade PnLs, resample with replacement N times
to build a distribution of plausible aggregate PnLs / Sharpes / drawdowns.
This is the standard "is the result luck or edge?" diagnostic — if the 5th
percentile of bootstrapped PnL is still positive, the edge is robust;
otherwise the headline number is in noise territory.

Two kinds of resampling are supported:
  - resample_trades: i.i.d. bootstrap of per-trade PnLs (most common)
  - resample_blocks: stationary block bootstrap (preserves serial dependence)
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from futures_bot.backtest.engine import BacktestResult


@dataclass
class MCStats:
    n_iter: int
    mean_pnl: float
    median_pnl: float
    p05_pnl: float
    p95_pnl: float
    pct_profitable: float  # fraction of bootstrap paths with pnl > 0
    mean_sharpe: float
    p05_sharpe: float
    p95_sharpe: float
    mean_max_dd: float
    p95_max_dd: float
    verdict: str

    def as_row(self) -> dict[str, float]:
        return {
            "n_iter": self.n_iter,
            "mean_pnl": self.mean_pnl,
            "median_pnl": self.median_pnl,
            "p05_pnl": self.p05_pnl,
            "p95_pnl": self.p95_pnl,
            "pct_profitable": self.pct_profitable,
            "mean_sharpe": self.mean_sharpe,
            "p05_sharpe": self.p05_sharpe,
            "p95_sharpe": self.p95_sharpe,
            "mean_max_dd": self.mean_max_dd,
            "p95_max_dd": self.p95_max_dd,
        }


def bootstrap_trade_returns(
    result: BacktestResult,
    n_iter: int = 1000,
    block_size: int = 1,
    seed: int = 7,
) -> MCStats:
    """Resample the trade-PnL series and report distribution stats.

    block_size > 1 enables a simple stationary block bootstrap, which
    preserves clustering / autocorrelation in trade outcomes (important
    for trend-following strategies).
    """
    pnls = [t.realized_pnl for t in result.trades if t.realized_pnl != 0.0]
    if not pnls:
        return MCStats(
            n_iter=0, mean_pnl=0, median_pnl=0, p05_pnl=0, p95_pnl=0,
            pct_profitable=0, mean_sharpe=0, p05_sharpe=0, p95_sharpe=0,
            mean_max_dd=0, p95_max_dd=0,
            verdict="no realized trades — nothing to resample",
        )

    rng = random.Random(seed)
    n = len(pnls)
    pnl_dist: list[float] = []
    sharpe_dist: list[float] = []
    dd_dist: list[float] = []

    for _ in range(n_iter):
        sample = _draw(pnls, n, block_size, rng)
        pnl_dist.append(sum(sample))
        sharpe_dist.append(_sharpe(sample))
        dd_dist.append(_max_drawdown(sample))

    return MCStats(
        n_iter=n_iter,
        mean_pnl=statistics.fmean(pnl_dist),
        median_pnl=statistics.median(pnl_dist),
        p05_pnl=_percentile(pnl_dist, 0.05),
        p95_pnl=_percentile(pnl_dist, 0.95),
        pct_profitable=sum(1 for p in pnl_dist if p > 0) / n_iter,
        mean_sharpe=statistics.fmean(sharpe_dist),
        p05_sharpe=_percentile(sharpe_dist, 0.05),
        p95_sharpe=_percentile(sharpe_dist, 0.95),
        mean_max_dd=statistics.fmean(dd_dist),
        p95_max_dd=_percentile(dd_dist, 0.95),
        verdict=_verdict(pnl_dist, n_iter),
    )


def _draw(pnls: list[float], n: int, block_size: int, rng: random.Random) -> list[float]:
    if block_size <= 1:
        return [rng.choice(pnls) for _ in range(n)]
    # Stationary block bootstrap: pick a random start and run for block_size,
    # repeat until length n. Wraps around.
    out: list[float] = []
    while len(out) < n:
        start = rng.randint(0, len(pnls) - 1)
        for k in range(block_size):
            if len(out) >= n:
                break
            out.append(pnls[(start + k) % len(pnls)])
    return out


def _sharpe(returns: Iterable[float]) -> float:
    xs = list(returns)
    if len(xs) < 2:
        return 0.0
    mean = statistics.fmean(xs)
    std = statistics.pstdev(xs)
    if std == 0:
        return 0.0
    # Normalized per-trade Sharpe — leave annualization to the caller if needed.
    return mean / std * math.sqrt(len(xs))


def _max_drawdown(pnls: Iterable[float]) -> float:
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _verdict(pnl_dist: list[float], n_iter: int) -> str:
    if not pnl_dist:
        return "no data"
    p05 = _percentile(pnl_dist, 0.05)
    p95 = _percentile(pnl_dist, 0.95)
    pct_profitable = sum(1 for p in pnl_dist if p > 0) / n_iter
    if p05 > 0:
        return f"ROBUST: 5th-percentile bootstrap pnl is positive (${p05:,.0f})"
    if p95 < 0:
        return f"BAD: even the 95th-percentile bootstrap pnl is negative (${p95:,.0f})"
    if pct_profitable < 0.55:
        return (
            f"NOISE: only {pct_profitable:.0%} of bootstrap paths are profitable — "
            f"the headline number is not distinguishable from luck"
        )
    return (
        f"INCONCLUSIVE: {pct_profitable:.0%} profitable but 5th pct is "
        f"${p05:,.0f}, 95th is ${p95:,.0f}"
    )
