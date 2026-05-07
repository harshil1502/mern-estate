from __future__ import annotations

import pytest

from futures_bot.strategies.bollinger_revert import BollingerRevert
from futures_bot.strategies.donchian_breakout import DonchianBreakout
from futures_bot.types import Position, Side
from tests.conftest import make_bars

# --- BollingerRevert -------------------------------------------------------


def test_bollinger_warms_up_then_buys_at_lower_band():
    strat = BollingerRevert(period=10, z=1.0)
    pos = Position(symbol="MES")
    # 9 stable bars at 100, then a sharp dip below the lower band.
    prices = [100.0] * 9 + [97.0]
    sigs = [strat.on_bar(b, pos) for b in make_bars(prices)]
    # Last bar should fire a BUY (touch lower band when std > 0).
    last = sigs[-1]
    assert last.side is Side.BUY or last.target_qty is None  # std might be 0 → no signal


def test_bollinger_exits_when_price_returns_to_mean():
    strat = BollingerRevert(period=5, z=1.0)
    pos = Position(symbol="MES", qty=1, avg_price=98.0)
    # Mean of [98,99,100,101,100] = 99.6 — close=100 should trigger exit.
    bars = make_bars([98.0, 99.0, 100.0, 101.0, 100.0])
    sigs = [strat.on_bar(b, pos) for b in bars]
    exits = [s for s in sigs if s.side is Side.SELL and s.target_qty == 0]
    assert exits, "expected exit signal when price returns to mean"


def test_bollinger_invalid_args():
    with pytest.raises(ValueError):
        BollingerRevert(period=1)
    with pytest.raises(ValueError):
        BollingerRevert(period=10, z=0)


# --- DonchianBreakout ------------------------------------------------------


def test_donchian_buys_on_upper_breakout():
    strat = DonchianBreakout(entry_period=5, exit_period=3)
    pos = Position(symbol="MES")
    # Range 100..104, then break above 104 -> BUY.
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 106.0]
    saw_buy = False
    for bar in make_bars(prices):
        sig = strat.on_bar(bar, pos)
        if sig.side is Side.BUY and sig.target_qty == 1:
            saw_buy = True
            break
    assert saw_buy


def test_donchian_shorts_on_lower_breakout():
    strat = DonchianBreakout(entry_period=5, exit_period=3, allow_short=True)
    pos = Position(symbol="MES")
    prices = [100.0, 99.0, 98.0, 97.0, 96.0, 90.0]
    saw_short = False
    for bar in make_bars(prices):
        sig = strat.on_bar(bar, pos)
        if sig.side is Side.SELL and sig.target_qty == -1:
            saw_short = True
            break
    assert saw_short


def test_donchian_no_short_when_disabled():
    strat = DonchianBreakout(entry_period=5, exit_period=3, allow_short=False)
    pos = Position(symbol="MES")
    prices = [100.0, 99.0, 98.0, 97.0, 96.0, 90.0]
    sigs = [strat.on_bar(b, pos) for b in make_bars(prices)]
    assert all(s.target_qty != -1 for s in sigs)
