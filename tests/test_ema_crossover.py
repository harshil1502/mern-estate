from __future__ import annotations

from futures_bot.strategies.ema_crossover import EmaCrossover
from futures_bot.types import Position, Side
from tests.conftest import make_bars


def test_warmup_returns_no_action():
    strat = EmaCrossover(fast=3, slow=5)
    pos = Position(symbol="MES")
    bars = make_bars([100, 101, 102, 103])
    sigs = [strat.on_bar(b, pos) for b in bars]
    assert all(s.target_qty is None for s in sigs)


def test_bullish_cross_emits_buy():
    strat = EmaCrossover(fast=3, slow=5)
    pos = Position(symbol="MES")
    # Drift down then sharp up to force a fast-over-slow cross.
    prices = [100, 99, 98, 97, 96, 95, 94, 110, 112, 115, 118, 120]
    saw_buy = False
    for bar in make_bars(prices):
        sig = strat.on_bar(bar, pos)
        if sig.side is Side.BUY and sig.target_qty == 1:
            saw_buy = True
            break
    assert saw_buy, "expected an EMA bullish crossover to emit a BUY signal"


def test_bearish_cross_flattens_long():
    strat = EmaCrossover(fast=3, slow=5)
    pos = Position(symbol="MES", qty=1, avg_price=100.0)
    prices = [100, 102, 104, 106, 108, 110, 95, 90, 85, 80, 75]
    saw_flatten = False
    for bar in make_bars(prices):
        sig = strat.on_bar(bar, pos)
        if sig.side is Side.SELL and sig.target_qty == 0:
            saw_flatten = True
            break
    assert saw_flatten, "expected EMA bearish crossover to flatten an existing long"


def test_invalid_periods_rejected():
    import pytest

    with pytest.raises(ValueError):
        EmaCrossover(fast=10, slow=10)
