from __future__ import annotations

from futures_bot.backtest.engine import BacktestEngine
from futures_bot.config import InstrumentConfig, RiskConfig
from futures_bot.strategies.base import Strategy
from futures_bot.types import Bar, Position, Side, Signal
from tests.conftest import make_bars


class _AlwaysLongOnce(Strategy):
    """Goes long 1 on the second bar, exits on the fifth."""

    name = "test_strategy"

    def __init__(self) -> None:
        self.i = 0

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        self.i += 1
        if self.i == 2:
            return Signal(side=Side.BUY, target_qty=1, reason="enter")
        if self.i == 5:
            return Signal(side=Side.SELL, target_qty=0, reason="exit")
        return Signal()


def _instr() -> InstrumentConfig:
    return InstrumentConfig(symbol="MES", exchange="CME", tick_size=0.25, point_value=5.0)


def _risk() -> RiskConfig:
    return RiskConfig(max_position=1, per_trade_risk_usd=50, daily_loss_limit_usd=10_000,
                       kill_switch_drawdown_usd=10_000)


def test_backtester_realizes_profit_on_winner():
    bars = make_bars([100, 101, 102, 103, 104, 105])
    strat = _AlwaysLongOnce()
    engine = BacktestEngine(strat, _instr(), _risk(), initial_equity=10_000.0)
    result = engine.run(bars)
    # Filled buy at bar 3 open (102), sell at bar 6 open (105) — but bar 6 doesn't
    # exist, so exit fills at last close (105). PnL = (105-102) * 1 * 5 = 15.
    assert result.num_trades == 2
    assert result.realized_pnl == 15.0
    assert result.final_equity == 10_015.0


def test_equity_curve_length_matches_bars():
    bars = make_bars([100, 100, 100])
    engine = BacktestEngine(_AlwaysLongOnce(), _instr(), _risk(), initial_equity=10_000.0)
    result = engine.run(bars)
    assert len(result.equity_curve) == len(bars)


def test_no_signal_means_no_trades():
    class _DoNothing(Strategy):
        name = "noop"

        def on_bar(self, bar: Bar, position: Position) -> Signal:
            return Signal()

    bars = make_bars([100, 101, 99, 102])
    engine = BacktestEngine(_DoNothing(), _instr(), _risk(), initial_equity=10_000.0)
    result = engine.run(bars)
    assert result.num_trades == 0
    assert result.realized_pnl == 0.0
