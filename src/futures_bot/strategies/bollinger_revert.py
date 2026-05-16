"""Bollinger band mean-reversion.

Computes a rolling mean and population std over `period` closes. Goes long when
close prints below `mean - z*std` and flat when it returns to the mean; goes
short on the symmetric upper-band touch. Designed for chop / range regimes;
gets carried out by trends.
"""

from __future__ import annotations

from collections import deque

from futures_bot.strategies.base import Strategy
from futures_bot.strategies.registry import register_strategy
from futures_bot.types import Bar, Position, Side, Signal


@register_strategy("bollinger_revert")
class BollingerRevert(Strategy):
    name = "bollinger_revert"

    def __init__(
        self,
        period: int = 20,
        z: float = 2.0,
        bar_seconds: int = 60,
        allow_short: bool = True,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        if z <= 0:
            raise ValueError("z must be > 0")
        self.period = period
        self.z = z
        self.bar_seconds = bar_seconds
        self.allow_short = allow_short
        self._closes: deque[float] = deque(maxlen=period)

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        self._closes.append(bar.close)
        if len(self._closes) < self.period:
            return Signal(reason="warming up")

        mean = sum(self._closes) / self.period
        var = sum((c - mean) ** 2 for c in self._closes) / self.period
        std = var**0.5
        if std == 0:
            return Signal(reason="zero vol")

        upper = mean + self.z * std
        lower = mean - self.z * std

        # Exits first — return to mean closes any open position.
        if position.qty > 0 and bar.close >= mean:
            return Signal(side=Side.SELL, target_qty=0, reason="long exit at mean")
        if position.qty < 0 and bar.close <= mean:
            return Signal(side=Side.BUY, target_qty=0, reason="short exit at mean")

        # Entries — only when flat.
        if position.qty == 0:
            if bar.close <= lower:
                return Signal(side=Side.BUY, target_qty=1, reason="touch lower band")
            if self.allow_short and bar.close >= upper:
                return Signal(side=Side.SELL, target_qty=-1, reason="touch upper band")

        return Signal(reason="hold")
