"""Donchian channel breakout.

Goes long when close prints above the previous N-bar high, short on previous
N-bar low. Exit on opposite breakout, with an optional faster `exit_period`
(< `entry_period`) to give back less profit. Trend-follower; gets chopped up
in range regimes — the natural pair to bollinger_revert in a sweep.
"""

from __future__ import annotations

from collections import deque

from futures_bot.strategies.base import Strategy
from futures_bot.strategies.registry import register_strategy
from futures_bot.types import Bar, Position, Side, Signal


@register_strategy("donchian_breakout")
class DonchianBreakout(Strategy):
    name = "donchian_breakout"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int | None = None,
        bar_seconds: int = 60,
        allow_short: bool = True,
    ) -> None:
        if entry_period < 2:
            raise ValueError("entry_period must be >= 2")
        if exit_period is not None and exit_period < 1:
            raise ValueError("exit_period must be >= 1")
        self.entry_period = entry_period
        self.exit_period = exit_period or max(2, entry_period // 2)
        self.bar_seconds = bar_seconds
        self.allow_short = allow_short
        self._highs: deque[float] = deque(maxlen=entry_period)
        self._lows: deque[float] = deque(maxlen=entry_period)
        self._exit_highs: deque[float] = deque(maxlen=self.exit_period)
        self._exit_lows: deque[float] = deque(maxlen=self.exit_period)

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        # Channels are computed BEFORE we update with this bar — entries are on
        # a break of the prior N-bar extremes, not the current one.
        entry_high = max(self._highs) if len(self._highs) == self.entry_period else None
        entry_low = min(self._lows) if len(self._lows) == self.entry_period else None
        exit_high = max(self._exit_highs) if len(self._exit_highs) == self.exit_period else None
        exit_low = min(self._exit_lows) if len(self._exit_lows) == self.exit_period else None

        # Update buffers for the next bar.
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._exit_highs.append(bar.high)
        self._exit_lows.append(bar.low)

        if entry_high is None or entry_low is None:
            return Signal(reason="warming up")

        # Exits first.
        if position.qty > 0 and exit_low is not None and bar.close < exit_low:
            return Signal(side=Side.SELL, target_qty=0, reason="long exit on lower channel")
        if position.qty < 0 and exit_high is not None and bar.close > exit_high:
            return Signal(side=Side.BUY, target_qty=0, reason="short exit on upper channel")

        # Entries (only when flat).
        if position.qty == 0:
            if bar.close > entry_high:
                return Signal(side=Side.BUY, target_qty=1, reason="upper channel breakout")
            if self.allow_short and bar.close < entry_low:
                return Signal(side=Side.SELL, target_qty=-1, reason="lower channel breakout")

        return Signal(reason="hold")
