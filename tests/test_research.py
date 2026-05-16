from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_bot.backtest.engine import BacktestResult, Trade
from futures_bot.config import InstrumentConfig, RiskConfig
from futures_bot.data.synthetic import synthetic_bars
from futures_bot.research.goals import GoalSpec, evaluate_goal
from futures_bot.research.grid import ParamGrid
from futures_bot.research.metrics import compute_metrics
from futures_bot.research.sweep import run_sweep
from futures_bot.research.walk_forward import run_walk_forward
from futures_bot.types import Side


def _instr() -> InstrumentConfig:
    return InstrumentConfig(symbol="MES", exchange="CME", tick_size=0.25, point_value=5.0)


def _risk() -> RiskConfig:
    return RiskConfig(
        max_position=1, per_trade_risk_usd=50,
        daily_loss_limit_usd=10_000, kill_switch_drawdown_usd=10_000,
    )


# --- ParamGrid -------------------------------------------------------------


def test_grid_expand_cross_product():
    g = ParamGrid(
        "ema_crossover",
        params={"fast": [3, 5], "slow": [9, 21]},
        fixed={"bar_seconds": 60},
    )
    combos = list(g.expand())
    assert g.size() == 4
    assert len(combos) == 4
    for c in combos:
        assert c["bar_seconds"] == 60
        assert c["fast"] in (3, 5)
        assert c["slow"] in (9, 21)


def test_grid_empty_params_yields_fixed_only():
    g = ParamGrid("ema_crossover", fixed={"fast": 3, "slow": 9, "bar_seconds": 60})
    combos = list(g.expand())
    assert combos == [{"fast": 3, "slow": 9, "bar_seconds": 60}]


# --- metrics ---------------------------------------------------------------


def test_metrics_zero_safe_on_no_trades():
    bars = synthetic_bars(n=20, seed=1)
    result = BacktestResult(
        trades=[], equity_curve=[(b.ts, 10_000.0) for b in bars],
        final_equity=10_000.0, realized_pnl=0.0, max_drawdown=0.0,
    )
    m = compute_metrics(result, bars, initial_equity=10_000.0)
    assert m.sharpe_annualized == 0.0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.median_daily_pnl == 0.0


def test_metrics_groups_pnl_by_day():
    bars = synthetic_bars(n=10, seed=2)
    t0 = bars[0].ts
    trades = [
        Trade(ts=t0, side=Side.BUY, qty=1, price=100.0, realized_pnl=10.0),
        Trade(ts=t0 + timedelta(hours=1), side=Side.SELL, qty=1, price=110.0, realized_pnl=20.0),
        Trade(ts=t0 + timedelta(days=1), side=Side.BUY, qty=1, price=120.0, realized_pnl=-5.0),
    ]
    result = BacktestResult(
        trades=trades, equity_curve=[(b.ts, 10_000.0) for b in bars],
        final_equity=10_025.0, realized_pnl=25.0, max_drawdown=5.0,
    )
    m = compute_metrics(result, bars, initial_equity=10_000.0)
    assert m.n_trading_days == 2
    daily_lookup = dict(m.daily_pnls)
    assert daily_lookup[t0.date()] == 30.0
    assert daily_lookup[(t0 + timedelta(days=1)).date()] == -5.0


# --- sweep -----------------------------------------------------------------


def test_sweep_runs_all_combos_and_ranks():
    bars = synthetic_bars(n=400, seed=3)
    grid = ParamGrid(
        "ema_crossover",
        params={"fast": [3, 5], "slow": [21, 34]},
        fixed={"bar_seconds": 60},
    )
    report = run_sweep(grid, bars, _instr(), _risk(), initial_equity=10_000.0)
    assert len(report.results) == 4
    top = report.top(2, key="sharpe")
    assert len(top) == 2
    assert top[0].metrics.sharpe_annualized >= top[1].metrics.sharpe_annualized


def test_sweep_ranks_no_trade_strategies_last():
    """Zero-trade params shouldn't beat traded-but-losing params on Sharpe."""
    bars = synthetic_bars(n=300, seed=11)
    # Bollinger with z=10 will almost certainly never trigger.
    grid = ParamGrid(
        "bollinger_revert",
        params={"period": [20], "z": [10.0, 1.5]},
        fixed={"bar_seconds": 60, "allow_short": True},
    )
    report = run_sweep(grid, bars, _instr(), _risk(), initial_equity=10_000.0)
    top = report.top(2, key="sharpe")
    # The traded variant (z=1.5) must rank above the no-trade variant (z=10).
    assert top[0].params["z"] == 1.5
    assert top[0].metrics.num_trades > 0


def test_sweep_records_error_on_invalid_params():
    bars = synthetic_bars(n=200, seed=4)
    # fast >= slow is rejected by the strategy.
    grid = ParamGrid(
        "ema_crossover",
        params={"fast": [21], "slow": [21]},
        fixed={"bar_seconds": 60},
    )
    report = run_sweep(grid, bars, _instr(), _risk(), initial_equity=10_000.0)
    assert len(report.results) == 1
    assert report.results[0].error


# --- walk-forward ----------------------------------------------------------


def test_walk_forward_produces_one_fold_per_K():
    bars = synthetic_bars(n=2000, seed=7)
    grid = ParamGrid(
        "ema_crossover",
        params={"fast": [3, 5], "slow": [13, 21]},
        fixed={"bar_seconds": 60},
    )
    wf = run_walk_forward(
        grid, bars, _instr(), _risk(), initial_equity=10_000.0,
        n_folds=4, train_frac=0.7, select_by="sharpe",
    )
    assert len(wf.folds) == 4
    for f in wf.folds:
        # IS window precedes OOS window in every fold.
        assert f.is_range[1] <= f.oos_range[0]


# --- goals -----------------------------------------------------------------


def test_goal_spec_required_returns():
    g = GoalSpec(target_daily_pnl_usd=500, account_size_usd=150_000)
    assert abs(g.required_daily_return - 500 / 150_000) < 1e-9
    # ~120% annualized at 0.33%/day compounded
    assert g.required_annualized_return > 1.0


def test_goal_evaluation_flags_winning_strategy():
    bars = synthetic_bars(n=20, seed=8)
    t0 = bars[0].ts.replace(hour=0, minute=0)
    trades = [
        Trade(ts=t0 + timedelta(days=i), side=Side.BUY, qty=1, price=100, realized_pnl=600)
        for i in range(5)
    ]
    result = BacktestResult(
        trades=trades,
        equity_curve=[(b.ts, 150_000.0) for b in bars],
        final_equity=153_000.0, realized_pnl=3_000.0, max_drawdown=0.0,
    )
    metrics = compute_metrics(result, bars, initial_equity=150_000.0)
    goal = GoalSpec(target_daily_pnl_usd=500, account_size_usd=150_000)
    a = evaluate_goal(metrics, goal)
    assert a.median_meets_target
    assert a.median_daily_pnl == 600.0
    assert "PASSES" in a.verdict


def test_goal_evaluation_flags_losing_strategy():
    bars = synthetic_bars(n=10, seed=9)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    trades = [
        Trade(ts=t0 + timedelta(days=i), side=Side.SELL, qty=1, price=100, realized_pnl=-50)
        for i in range(3)
    ]
    result = BacktestResult(
        trades=trades, equity_curve=[(b.ts, 150_000.0) for b in bars],
        final_equity=149_850.0, realized_pnl=-150.0, max_drawdown=150.0,
    )
    metrics = compute_metrics(result, bars, initial_equity=150_000.0)
    a = evaluate_goal(metrics, GoalSpec(500, 150_000))
    assert not a.median_meets_target
    assert "FAILS" in a.verdict
