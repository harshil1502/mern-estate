from __future__ import annotations

from collections import deque

from futures_bot.strategies.base import Strategy
from futures_bot.strategies.registry import register_strategy
from futures_bot.types import Bar, Position, Side, Signal


def _ema_alpha(period: int) -> float:
    return 2.0 / (period + 1)


@register_strategy("ema_crossover")
class EmaCrossover(Strategy):
    """Long-flat-short on fast/slow EMA crossover.

    Goes long when fast crosses above slow, flat when fast crosses below.
    Position sizing is 1 contract; risk/sizing is the executor's job.
    """

    name = "ema_crossover"

    def __init__(self, fast: int = 9, slow: int = 21, bar_seconds: int = 60) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self.bar_seconds = bar_seconds
        self._fast_alpha = _ema_alpha(fast)
        self._slow_alpha = _ema_alpha(slow)
        self._fast_ema: float | None = None
        self._slow_ema: float | None = None
        self._warmup: deque[float] = deque(maxlen=slow)
        self._prev_diff: float | None = None

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        price = bar.close
        self._warmup.append(price)

        if len(self._warmup) < self.slow:
            return Signal(reason="warming up")

        if self._fast_ema is None:
            seed_fast = sum(list(self._warmup)[-self.fast :]) / self.fast
            seed_slow = sum(self._warmup) / self.slow
            self._fast_ema = seed_fast
            self._slow_ema = seed_slow
        else:
            self._fast_ema = self._fast_alpha * price + (1 - self._fast_alpha) * self._fast_ema
            assert self._slow_ema is not None
            self._slow_ema = self._slow_alpha * price + (1 - self._slow_alpha) * self._slow_ema

        diff = self._fast_ema - self._slow_ema  # type: ignore[operator]
        prev = self._prev_diff
        self._prev_diff = diff

        if prev is None:
            return Signal(reason="seeding diff")

        # Bullish cross: go long 1
        if prev <= 0 < diff:
            return Signal(side=Side.BUY, target_qty=1, reason="ema fast crossed above slow")
        # Bearish cross: flatten
        if prev >= 0 > diff and position.qty > 0:
            return Signal(side=Side.SELL, target_qty=0, reason="ema fast crossed below slow")
        return Signal(reason="hold")
