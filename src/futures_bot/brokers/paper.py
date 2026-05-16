"""In-process simulated broker for paper trading without a live account.

Implements the full `Broker` surface against an arbitrary async bar source.
Orders are queued on `place_order` and filled at the *next* bar's open price
(plus slippage), which is the standard fill model for bar-by-bar paper sim
and avoids the look-ahead bias of "fill at current close."

Equity, realized PnL, and trade list are tracked on the broker so the runner
can report them via the same `Broker` API used in live mode.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from futures_bot.brokers.base import Broker
from futures_bot.config import InstrumentConfig
from futures_bot.types import Bar, Order, OrderStatus, OrderType, Position, Side

log = logging.getLogger(__name__)


@dataclass
class Fill:
    ts: datetime
    side: Side
    qty: int
    price: float
    realized_pnl: float
    position_after: int
    equity_after: float


@dataclass
class _Pending:
    order: Order
    placed_at: datetime | None


@dataclass
class PaperState:
    initial_equity: float
    realized_pnl: float = 0.0
    position: Position = field(default_factory=lambda: Position(symbol=""))
    fills: list[Fill] = field(default_factory=list)
    last_bar: Bar | None = None

    def equity(self, point_value: float) -> float:
        mark = self.last_bar.close if self.last_bar else self.position.avg_price
        unrealized = self.position.unrealized_pnl(mark, point_value) if mark else 0.0
        return self.initial_equity + self.realized_pnl + unrealized


class PaperBroker(Broker):
    def __init__(
        self,
        instrument: InstrumentConfig,
        bar_source: AsyncIterable[Bar],
        initial_equity: float = 10_000.0,
        slippage_ticks: float = 0.5,
        commission_per_contract: float = 0.0,
        on_fill: Callable[[Fill], Any] | None = None,
    ) -> None:
        self.instrument = instrument
        self._bar_source = bar_source
        self.slippage_ticks = slippage_ticks
        self.commission_per_contract = commission_per_contract
        self._on_fill = on_fill
        self.state = PaperState(
            initial_equity=initial_equity,
            position=Position(symbol=instrument.symbol),
        )
        self._pending: list[_Pending] = []
        self._connected = False
        self._lock = asyncio.Lock()

    # --- lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True
        log.info("PaperBroker connected (equity=%.2f, slippage_ticks=%.2f)",
                 self.state.initial_equity, self.slippage_ticks)

    async def close(self) -> None:
        self._connected = False

    # --- account / positions ------------------------------------------------

    async def get_position(self, symbol: str) -> Position:
        return Position(
            symbol=self.state.position.symbol,
            qty=self.state.position.qty,
            avg_price=self.state.position.avg_price,
            realized_pnl=self.state.position.realized_pnl,
        )

    async def get_account_equity(self) -> float:
        return self.state.equity(self.instrument.point_value)

    # --- orders -------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if order_type is not OrderType.MARKET:
            # Limit/stop orders need a deeper match engine — skip for the scaffold.
            raise NotImplementedError("PaperBroker only supports MARKET orders")
        order = Order(
            id=uuid.uuid4().hex,
            symbol=symbol,
            side=side,
            qty=qty,
            type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.WORKING,
        )
        self._pending.append(_Pending(order=order, placed_at=self._now()))
        return order

    async def cancel_order(self, order_id: str) -> None:
        for p in list(self._pending):
            if p.order.id == order_id:
                p.order.status = OrderStatus.CANCELLED
                self._pending.remove(p)
                return

    # --- market data --------------------------------------------------------

    async def historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_seconds: int,
    ) -> list[Bar]:
        return []

    async def stream_bars(  # type: ignore[override]
        self,
        symbol: str,
        bar_seconds: int,
    ) -> AsyncIterator[Bar]:
        async for bar in self._bar_source:
            # Fill any pending orders at THIS bar's open before we let the strategy
            # see the bar — preserves the next-bar-open semantics.
            if self._pending:
                async with self._lock:
                    for pending in list(self._pending):
                        self._fill(pending.order, bar)
                        self._pending.remove(pending)
            self.state.last_bar = bar
            yield bar

    # --- fill engine --------------------------------------------------------

    def _fill(self, order: Order, bar: Bar) -> None:
        slip = self.slippage_ticks * self.instrument.tick_size
        price = bar.open + (slip if order.side is Side.BUY else -slip)
        signed = order.qty if order.side is Side.BUY else -order.qty
        prev_qty = self.state.position.qty
        new_qty = prev_qty + signed
        realized = 0.0
        pos = self.state.position

        if prev_qty == 0:
            pos.qty = new_qty
            pos.avg_price = price
        elif (prev_qty > 0 and signed > 0) or (prev_qty < 0 and signed < 0):
            total_cost = pos.avg_price * abs(prev_qty) + price * abs(signed)
            pos.qty = new_qty
            pos.avg_price = total_cost / abs(new_qty) if new_qty != 0 else 0.0
        else:
            closing_qty = min(abs(prev_qty), abs(signed))
            direction = 1 if prev_qty > 0 else -1
            pv = self.instrument.point_value
            realized = (price - pos.avg_price) * closing_qty * direction * pv
            self.state.realized_pnl += realized
            pos.realized_pnl += realized
            pos.qty = new_qty
            if new_qty == 0:
                pos.avg_price = 0.0
            elif (prev_qty > 0) != (new_qty > 0):
                pos.avg_price = price

        # Commission charged on every fill.
        commission = self.commission_per_contract * order.qty
        self.state.realized_pnl -= commission
        pos.realized_pnl -= commission
        net_realized = realized - commission

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.avg_fill_price = price

        equity_after = self.state.equity(self.instrument.point_value)
        fill = Fill(
            ts=bar.ts,
            side=order.side,
            qty=order.qty,
            price=price,
            realized_pnl=net_realized,
            position_after=pos.qty,
            equity_after=equity_after,
        )
        self.state.fills.append(fill)
        log.info("FILL %s %d @ %.2f | pos=%d realized=%+.2f equity=%.2f",
                 order.side.value, order.qty, price, pos.qty, net_realized, equity_after)
        if self._on_fill is not None:
            try:
                self._on_fill(fill)
            except Exception:
                log.exception("on_fill callback raised")

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _now() -> datetime | None:
        return None
