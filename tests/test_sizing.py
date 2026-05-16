from __future__ import annotations

import pytest

from futures_bot.config import InstrumentConfig
from futures_bot.execution.sizing import (
    FixedFractionalATR,
    FixedQty,
    VolatilityTarget,
    atr,
    build_sizer,
    true_range,
)
from futures_bot.types import Bar, Position, Side, Signal
from tests.conftest import make_bars


def _instr(point_value: float = 5.0) -> InstrumentConfig:
    return InstrumentConfig(symbol="MES", exchange="CME", tick_size=0.25, point_value=point_value)


# --- ATR helper -----------------------------------------------------------


def test_true_range_first_bar_is_high_minus_low():
    bar = Bar(ts=make_bars([100])[0].ts, open=100, high=102, low=99, close=101, volume=1)
    assert true_range(bar, prev_close=None) == pytest.approx(3.0)


def test_true_range_uses_prev_close_when_gap():
    bar = Bar(ts=make_bars([100])[0].ts, open=100, high=102, low=99, close=101, volume=1)
    # prev close above current high -> tr = max(3, |102-110|, |99-110|) = 11
    assert true_range(bar, prev_close=110.0) == pytest.approx(11.0)


def test_atr_handles_short_history():
    bars = make_bars([100, 101, 99])
    # only 2 TRs available, period=14 -> falls back to mean of available
    assert atr(bars, period=14) > 0


# --- FixedQty -------------------------------------------------------------


def test_fixed_qty_passes_through_for_exit():
    s = FixedQty(qty=3)
    out = s.size(Signal(side=Side.SELL, target_qty=0, reason="exit"),
                 Position(symbol="MES"), 10_000.0, _instr())
    assert out.target_qty == 0


def test_fixed_qty_scales_intent_to_configured_qty():
    s = FixedQty(qty=3)
    out = s.size(Signal(side=Side.BUY, target_qty=1, reason=""),
                 Position(symbol="MES"), 10_000.0, _instr())
    assert out.target_qty == 3
    out = s.size(Signal(side=Side.SELL, target_qty=-1, reason=""),
                 Position(symbol="MES"), 10_000.0, _instr())
    assert out.target_qty == -3


def test_fixed_qty_rejects_zero():
    with pytest.raises(ValueError):
        FixedQty(qty=0)


# --- FixedFractionalATR ---------------------------------------------------


def test_atr_sizer_falls_back_to_min_qty_with_no_history():
    s = FixedFractionalATR(risk_per_trade_pct=0.01, atr_period=10, atr_stop_mult=2)
    out = s.size(Signal(side=Side.BUY, target_qty=1, reason=""),
                 Position(symbol="MES"), 150_000.0, _instr())
    assert out.target_qty == 1


def test_atr_sizer_scales_qty_with_equity():
    s = FixedFractionalATR(risk_per_trade_pct=0.01, atr_period=5, atr_stop_mult=2, max_qty=200)
    bars = [
        Bar(ts=make_bars([0])[0].ts, open=4500, high=4501, low=4499, close=4500, volume=1),
        Bar(ts=make_bars([0])[0].ts, open=4500, high=4502, low=4498, close=4501, volume=1),
        Bar(ts=make_bars([0])[0].ts, open=4501, high=4503, low=4499, close=4502, volume=1),
        Bar(ts=make_bars([0])[0].ts, open=4502, high=4504, low=4500, close=4503, volume=1),
        Bar(ts=make_bars([0])[0].ts, open=4503, high=4505, low=4501, close=4504, volume=1),
        Bar(ts=make_bars([0])[0].ts, open=4504, high=4506, low=4502, close=4505, volume=1),
    ]
    for b in bars:
        s.update(b)
    # ATR ~= 4 (high-low + carry from prev close gap); pv=5, stop_mult=2
    # -> risk_per_contract ~= 4*2*5 = $40
    # 1% of $150k = $1500; qty = floor(1500/40) ~= 37
    out = s.size(Signal(side=Side.BUY, target_qty=1, reason=""),
                 Position(symbol="MES"), 150_000.0, _instr(point_value=5))
    assert out.target_qty is not None
    # Sanity: substantially more than 1, well below max_qty cap.
    assert 20 <= out.target_qty <= 80


def test_atr_sizer_scales_short_intent():
    s = FixedFractionalATR(risk_per_trade_pct=0.01, atr_period=3, atr_stop_mult=2)
    for b in make_bars([100, 101, 99, 102, 98]):
        s.update(b)
    out = s.size(Signal(side=Side.SELL, target_qty=-1, reason=""),
                 Position(symbol="MES"), 150_000.0, _instr())
    assert out.target_qty is not None and out.target_qty < 0


def test_atr_sizer_invalid_args():
    with pytest.raises(ValueError):
        FixedFractionalATR(risk_per_trade_pct=2.0)
    with pytest.raises(ValueError):
        FixedFractionalATR(atr_period=1)
    with pytest.raises(ValueError):
        FixedFractionalATR(atr_stop_mult=0)


# --- VolatilityTarget -----------------------------------------------------


def test_vol_target_scales_inversely_with_atr():
    s = VolatilityTarget(bar_std_target_usd=100.0, atr_period=3)
    # Low ATR -> bigger qty
    for b in make_bars([100.0, 100.1, 100.0, 100.1, 100.0]):
        s.update(b)
    out_lo = s.size(Signal(side=Side.BUY, target_qty=1, reason=""),
                    Position(symbol="MES"), 150_000.0, _instr(point_value=5))

    s2 = VolatilityTarget(bar_std_target_usd=100.0, atr_period=3)
    # High ATR -> smaller qty
    for b in make_bars([100.0, 110.0, 95.0, 115.0, 90.0]):
        s2.update(b)
    out_hi = s2.size(Signal(side=Side.BUY, target_qty=1, reason=""),
                     Position(symbol="MES"), 150_000.0, _instr(point_value=5))
    assert out_lo.target_qty is not None and out_hi.target_qty is not None
    assert out_lo.target_qty >= out_hi.target_qty


# --- factory --------------------------------------------------------------


def test_build_sizer_factory():
    assert isinstance(build_sizer({}), FixedQty)
    assert isinstance(build_sizer({"type": "fixed", "qty": 3}), FixedQty)
    assert isinstance(
        build_sizer({"type": "atr", "risk_per_trade_pct": 0.01}),
        FixedFractionalATR,
    )
    assert isinstance(
        build_sizer({"type": "vol_target", "bar_std_target_usd": 100}),
        VolatilityTarget,
    )
    with pytest.raises(ValueError):
        build_sizer({"type": "unknown"})
