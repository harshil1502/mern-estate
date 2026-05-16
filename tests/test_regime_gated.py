from __future__ import annotations

from futures_bot.research.regimes import RegimeDetector
from futures_bot.strategies.base import Strategy
from futures_bot.strategies.regime_gated import RegimeGatedStrategy
from futures_bot.types import Bar, Position, Side, Signal
from tests.conftest import make_bars


class _AlwaysBuy(Strategy):
    name = "always_buy"

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        if position.qty == 0:
            return Signal(side=Side.BUY, target_qty=1, reason="enter")
        return Signal()


class _AlwaysExit(Strategy):
    name = "always_exit"

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        if position.qty != 0:
            return Signal(side=Side.SELL, target_qty=0, reason="exit")
        return Signal()


class _FakeDetector(RegimeDetector):
    def __init__(self, sequence: list[str]) -> None:
        self._seq = list(sequence)
        self._regime = "warmup"

    def update(self, bar: Bar) -> str:
        self._regime = self._seq.pop(0) if self._seq else self._regime
        return self._regime

    @property
    def regime(self) -> str:
        return self._regime


def test_gate_blocks_entries_outside_allowed_regimes():
    det = _FakeDetector(["high_vol"] * 5)
    gated = RegimeGatedStrategy(_AlwaysBuy(), det, allowed_regimes=("calm", "low_vol"))
    pos = Position(symbol="MES")
    bars = make_bars([100, 101, 102, 103, 104])
    sigs = [gated.on_bar(b, pos) for b in bars]
    # All entries should be gated (no target_qty != 0).
    assert all(s.target_qty in (None, 0) or s.side is None for s in sigs)


def test_gate_allows_entries_in_allowed_regimes():
    det = _FakeDetector(["calm"] * 5)
    gated = RegimeGatedStrategy(_AlwaysBuy(), det, allowed_regimes=("calm",))
    pos = Position(symbol="MES")
    bars = make_bars([100, 101, 102, 103, 104])
    s = gated.on_bar(bars[0], pos)
    assert s.side is Side.BUY
    assert s.target_qty == 1


def test_gate_always_allows_exits():
    """Even in a disallowed regime, an exit signal must pass through."""
    det = _FakeDetector(["high_vol"] * 5)
    gated = RegimeGatedStrategy(_AlwaysExit(), det, allowed_regimes=("calm",))
    # Long position — strategy wants to exit.
    pos = Position(symbol="MES", qty=1, avg_price=100)
    s = gated.on_bar(make_bars([100])[0], pos)
    assert s.side is Side.SELL
    assert s.target_qty == 0


def test_gate_passes_through_no_action_signals():
    det = _FakeDetector(["calm"] * 5)
    gated = RegimeGatedStrategy(_AlwaysBuy(), det, allowed_regimes=("calm",))
    # When already long, _AlwaysBuy returns Signal() (no action). Should pass.
    pos = Position(symbol="MES", qty=1, avg_price=100)
    s = gated.on_bar(make_bars([100])[0], pos)
    assert s.side is None
    assert s.target_qty is None
