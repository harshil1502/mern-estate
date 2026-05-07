"""Daily-PnL goal evaluation.

The user's stated goal is "$X per day on a $Y account." We translate that
into a required Sharpe + win-rate, and report what fraction of trading days
in the test data actually cleared the bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from futures_bot.research.metrics import StrategyMetrics


@dataclass
class GoalSpec:
    target_daily_pnl_usd: float
    account_size_usd: float

    @property
    def required_daily_return(self) -> float:
        return self.target_daily_pnl_usd / self.account_size_usd

    @property
    def required_annualized_return(self) -> float:
        # Compound daily over ~252 trading days/year.
        r = self.required_daily_return
        return (1.0 + r) ** 252 - 1.0


@dataclass
class GoalAssessment:
    goal: GoalSpec
    median_daily_pnl: float
    mean_daily_pnl: float
    target_daily_pnl: float
    pct_days_meeting_target: float
    pct_winning_days: float
    median_meets_target: bool
    realistic_sharpe_for_target: float
    actual_sharpe: float
    verdict: str


def evaluate_goal(metrics: StrategyMetrics, goal: GoalSpec) -> GoalAssessment:
    daily_values = [pnl for _, pnl in metrics.daily_pnls]
    if not daily_values:
        return GoalAssessment(
            goal=goal,
            median_daily_pnl=0.0,
            mean_daily_pnl=0.0,
            target_daily_pnl=goal.target_daily_pnl_usd,
            pct_days_meeting_target=0.0,
            pct_winning_days=0.0,
            median_meets_target=False,
            realistic_sharpe_for_target=0.0,
            actual_sharpe=metrics.sharpe_annualized,
            verdict="no trading days — strategy never traded or test window was empty",
        )

    n_meeting = sum(1 for v in daily_values if v >= goal.target_daily_pnl_usd)
    pct_meeting = n_meeting / len(daily_values)

    # Required Sharpe to reliably hit target_daily_pnl on the median day,
    # given the strategy's observed vol.
    required_sharpe = _required_sharpe_for_daily_target(
        target_daily_pnl=goal.target_daily_pnl_usd,
        std_daily_pnl=metrics.std_daily_pnl,
    )

    median_meets = metrics.median_daily_pnl >= goal.target_daily_pnl_usd
    verdict = _verdict(metrics, goal, pct_meeting, median_meets)

    return GoalAssessment(
        goal=goal,
        median_daily_pnl=metrics.median_daily_pnl,
        mean_daily_pnl=metrics.mean_daily_pnl,
        target_daily_pnl=goal.target_daily_pnl_usd,
        pct_days_meeting_target=pct_meeting,
        pct_winning_days=metrics.pct_winning_days,
        median_meets_target=median_meets,
        realistic_sharpe_for_target=required_sharpe,
        actual_sharpe=metrics.sharpe_annualized,
        verdict=verdict,
    )


def _required_sharpe_for_daily_target(target_daily_pnl: float, std_daily_pnl: float) -> float:
    """Daily Sharpe needed for the median day to clear `target`, then annualized."""
    if std_daily_pnl <= 0:
        return float("inf") if target_daily_pnl > 0 else 0.0
    daily_sharpe = target_daily_pnl / std_daily_pnl
    return daily_sharpe * math.sqrt(252)


def _verdict(
    metrics: StrategyMetrics,
    goal: GoalSpec,
    pct_meeting: float,
    median_meets: bool,
) -> str:
    if metrics.num_trades == 0:
        return "strategy never traded — increase aggressiveness or shorten signal cadence"
    if median_meets:
        return f"PASSES median-day target on this window (median {metrics.median_daily_pnl:+.2f})"
    if pct_meeting > 0.5:
        return f"meets target on {pct_meeting:.0%} of days but median is below — inconsistent"
    if metrics.realized_pnl < 0:
        return f"FAILS — strategy is net losing on this window (PnL {metrics.realized_pnl:+.2f})"
    return (
        f"FAILS — median day {metrics.median_daily_pnl:+.2f} vs target "
        f"{goal.target_daily_pnl_usd:+.2f} ({pct_meeting:.0%} of days clear it)"
    )
