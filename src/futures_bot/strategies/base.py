from __future__ import annotations

from abc import ABC, abstractmethod

from futures_bot.types import Bar, Position, Signal


class Strategy(ABC):
    """A pure function from (bar, position) -> signal.

    State (indicator buffers, etc.) lives on `self`; the runner feeds bars in
    order. Strategies must not place orders directly — return a Signal and let
    the executor translate it into orders, applying risk limits.
    """

    name: str = "strategy"

    @abstractmethod
    def on_bar(self, bar: Bar, position: Position) -> Signal: ...

    def on_start(self) -> None:
        """Optional hook called once before the first bar."""

    def on_stop(self) -> None:
        """Optional hook called when the runner shuts down."""
