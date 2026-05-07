from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from futures_bot.types import Bar, Order, OrderType, Position, Side


class Broker(ABC):
    """Minimal broker interface every connector must implement.

    Implementations may be REST-only, REST+WebSocket, or simulated. The runner
    depends only on this surface.
    """

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def get_position(self, symbol: str) -> Position: ...

    @abstractmethod
    async def get_account_equity(self) -> float: ...

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    async def historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_seconds: int,
    ) -> list[Bar]: ...

    @abstractmethod
    def stream_bars(self, symbol: str, bar_seconds: int) -> AsyncIterator[Bar]: ...
