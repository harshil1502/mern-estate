from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_bot.backtest.engine import BacktestResult, Trade
from futures_bot.research.monte_carlo import bootstrap_trade_returns
from futures_bot.types import Side


def _result_from_pnls(pnls: list[float]) -> BacktestResult:
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    trades = [
        Trade(ts=t0 + timedelta(hours=i), side=Side.BUY, qty=1, price=100.0, realized_pnl=p)
        for i, p in enumerate(pnls)
    ]
    return BacktestResult(
        trades=trades, equity_curve=[],
        final_equity=10_000 + sum(pnls), realized_pnl=sum(pnls), max_drawdown=0,
    )


def test_bootstrap_zero_safe_on_no_trades():
    s = bootstrap_trade_returns(_result_from_pnls([]), n_iter=100)
    assert s.n_iter == 0
    assert "no realized" in s.verdict


def test_bootstrap_strong_winner_is_robust():
    # 50 winners, 1 loser — the bootstrap should almost never produce a
    # negative aggregated PnL.
    pnls = [100.0] * 50 + [-50.0]
    s = bootstrap_trade_returns(_result_from_pnls(pnls), n_iter=2000, seed=11)
    assert s.pct_profitable > 0.95
    assert s.p05_pnl > 0
    assert "ROBUST" in s.verdict


def test_bootstrap_strong_loser_is_bad():
    pnls = [-100.0] * 30 + [50.0] * 5
    s = bootstrap_trade_returns(_result_from_pnls(pnls), n_iter=2000, seed=12)
    assert s.pct_profitable < 0.05
    assert s.p95_pnl < 0
    assert "BAD" in s.verdict


def test_bootstrap_marginal_is_inconclusive_or_noise():
    # Roughly balanced wins/losses — neither robust nor bad.
    pnls = [50.0, -45.0, 30.0, -55.0, 60.0, -40.0, 20.0, -25.0]
    s = bootstrap_trade_returns(_result_from_pnls(pnls), n_iter=2000, seed=13)
    assert s.verdict.startswith(("NOISE", "INCONCLUSIVE"))


def test_block_bootstrap_runs_and_returns_stats():
    pnls = [10.0, 10.0, -5.0, 8.0, -3.0, 12.0, 9.0, -8.0, 7.0, 11.0] * 10
    s = bootstrap_trade_returns(
        _result_from_pnls(pnls), n_iter=500, block_size=5, seed=14,
    )
    assert s.n_iter == 500
    # Mean of bootstrap should be near the empirical mean
    empirical = sum(pnls)
    assert abs(s.mean_pnl - empirical) / max(1.0, abs(empirical)) < 0.2
