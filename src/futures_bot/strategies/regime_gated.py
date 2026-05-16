"""Wraps a base strategy + regime detector. Only opens new positions when the
current regime is in `allowed_regimes`. Always honors exit signals so we
don't get stranded when the regime flips.
"""

from __future__ import annotations

from futures_bot.research.regimes import RegimeDetector
from futures_bot.strategies.base import Strategy
from futures_bot.types import Bar, Position, Signal


class RegimeGatedStrategy(Strategy):
    """Compose any Strategy with a RegimeDetector. Entries only when in regime."""

    name = "regime_gated"

    def __init__(
        self,
        base: Strategy,
        detector: RegimeDetector,
        allowed_regimes: tuple[str, ...] = ("calm", "low_vol", "mid_vol"),
    ) -> None:
        self.base = base
        self.detector = detector
        self.allowed_regimes = set(allowed_regimes)
        self.name = f"regime_gated({base.name})"

    def on_start(self) -> None:
        self.base.on_start()

    def on_stop(self) -> None:
        self.base.on_stop()

    def on_bar(self, bar: Bar, position: Position) -> Signal:
        regime = self.detector.update(bar)
        signal = self.base.on_bar(bar, position)

        # Always allow exits — regardless of regime, if we're in a position
        # the wrapped strategy says close, close.
        if signal.target_qty == 0 and signal.side is not None:
            return Signal(
                side=signal.side,
                target_qty=0,
                reason=f"{signal.reason} | regime={regime} (exit always)",
            )

        # New entries only in allowed regimes.
        if signal.target_qty is not None and signal.target_qty != 0:
            if regime not in self.allowed_regimes:
                return Signal(reason=f"{signal.reason} | gated by regime={regime}")
            return Signal(
                side=signal.side,
                target_qty=signal.target_qty,
                reason=f"{signal.reason} | regime={regime}",
            )

        # Hold/no-action signals pass through unchanged.
        if signal.side is None:
            return Signal(reason=f"{signal.reason} | regime={regime}")
        return signal
