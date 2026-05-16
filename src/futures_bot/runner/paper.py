from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from futures_bot.brokers.base import Broker
from futures_bot.config import AppConfig
from futures_bot.execution.risk import RiskManager, RiskRejection
from futures_bot.execution.sizing import FixedQty, PositionSizer
from futures_bot.strategies.base import Strategy
from futures_bot.types import OrderType

log = logging.getLogger(__name__)


@dataclass
class RunStats:
    bars_seen: int = 0
    orders_placed: int = 0
    halted_reason: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    final_equity: float = 0.0
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    journal_entries: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        pnl = self.final_equity - self.initial_equity
        pnl_pct = (pnl / self.initial_equity * 100) if self.initial_equity else 0.0
        return (
            f"bars={self.bars_seen} orders={self.orders_placed} "
            f"equity={self.final_equity:.2f} pnl={pnl:+.2f} ({pnl_pct:+.2f}%) "
            f"max_dd={self.max_drawdown:.2f}"
            + (f" halted={self.halted_reason}" if self.halted_reason else "")
        )


class PaperRunner:
    """Wires a Broker + Strategy + RiskManager into a live trading loop.

    The same loop drives both Tradovate paper/live and the in-process
    PaperBroker — only the Broker implementation changes.
    """

    def __init__(
        self,
        cfg: AppConfig,
        broker: Broker,
        strategy: Strategy,
        on_signal: Callable[[Any], None] | None = None,
        sizer: PositionSizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.strategy = strategy
        self.risk = RiskManager(cfg.risk)
        self.sizer = sizer or FixedQty(1)
        self.stats = RunStats()
        self._on_signal = on_signal
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> RunStats:
        await self.broker.connect()
        symbol = self.cfg.instrument.symbol
        bar_seconds = int(self.cfg.strategy.params.get("bar_seconds", 60))
        self.stats.started_at = datetime.now()
        try:
            self.strategy.on_start()
            self.stats.initial_equity = await self.broker.get_account_equity()
            self.stats.peak_equity = self.stats.initial_equity
            log.info("Streaming %s bars (%ds) for strategy=%s, initial_equity=%.2f",
                     symbol, bar_seconds, self.strategy.name, self.stats.initial_equity)

            async for bar in self.broker.stream_bars(symbol, bar_seconds):
                if self._stopped.is_set():
                    break
                self.stats.bars_seen += 1
                self.sizer.update(bar)

                position = await self.broker.get_position(symbol)
                equity = await self.broker.get_account_equity()
                self._track_equity(equity)
                self.risk.on_equity(equity)

                try:
                    raw = self.strategy.on_bar(bar, position)
                    sized = self.sizer.size(raw, position, equity, self.cfg.instrument)
                    signal = self.risk.vet_signal(sized, position)
                except RiskRejection as e:
                    self.stats.halted_reason = str(e)
                    log.warning("risk halted: %s", e)
                    break

                if self._on_signal is not None and signal.target_qty is not None:
                    self._on_signal(signal)

                order = self.risk.order_qty_for(signal, position)
                if order is None:
                    continue
                side, qty = order
                log.info("placing %s %d %s — %s", side.value, qty, symbol, signal.reason)
                await self.broker.place_order(symbol, side, qty, OrderType.MARKET)
                self.stats.orders_placed += 1
        finally:
            self.strategy.on_stop()
            try:
                self.stats.final_equity = await self.broker.get_account_equity()
            except Exception:
                self.stats.final_equity = self.stats.peak_equity
            self.stats.ended_at = datetime.now()
            await self.broker.close()
            log.info("Run summary: %s", self.stats.summary())
        return self.stats

    def _track_equity(self, equity: float) -> None:
        self.stats.peak_equity = max(self.stats.peak_equity, equity)
        dd = self.stats.peak_equity - equity
        if dd > self.stats.max_drawdown:
            self.stats.max_drawdown = dd
