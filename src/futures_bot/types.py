from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    WORKING = "WORKING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Tick:
    ts: datetime
    price: float
    size: float


@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    qty: int
    type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float | None = None


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    def unrealized_pnl(self, mark: float, point_value: float) -> float:
        if self.qty == 0:
            return 0.0
        return (mark - self.avg_price) * self.qty * point_value


@dataclass
class Signal:
    """Output of a strategy on each bar — what it wants the executor to do."""

    side: Side | None = None  # None == no action
    target_qty: int | None = None  # absolute target position; None means leave alone
    reason: str = ""
