from __future__ import annotations

import pytest

from futures_bot.data.synthetic import synthetic_bars


def test_seed_is_deterministic():
    a = synthetic_bars(n=50, seed=7)
    b = synthetic_bars(n=50, seed=7)
    assert [bar.close for bar in a] == [bar.close for bar in b]


def test_different_seeds_diverge():
    a = synthetic_bars(n=50, seed=1)
    b = synthetic_bars(n=50, seed=2)
    assert [bar.close for bar in a] != [bar.close for bar in b]


def test_prices_snap_to_tick():
    bars = synthetic_bars(n=200, tick_size=0.25, seed=3)
    for bar in bars:
        for price in (bar.open, bar.high, bar.low, bar.close):
            # 0.25 ticks divide cleanly: price * 4 should be close to integer
            assert abs(round(price * 4) - price * 4) < 1e-9


def test_high_ge_open_close_low():
    bars = synthetic_bars(n=200, seed=5)
    for bar in bars:
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low <= min(bar.open, bar.close)
        assert bar.high >= bar.low


def test_first_bar_starts_at_start_price():
    bars = synthetic_bars(n=10, start_price=4500.0, tick_size=0.25, seed=11)
    assert bars[0].open == 4500.0


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        synthetic_bars(n=0)
    with pytest.raises(ValueError):
        synthetic_bars(n=10, annual_vol=0.0)
