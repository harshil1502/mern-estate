from __future__ import annotations

import asyncio
import logging

from futures_bot.brokers.base import Broker
from futures_bot.config import AppConfig
from futures_bot.execution.risk import RiskManager, RiskRejection
from futures_bot.strategies.base import Strategy
from futures_bot.types import OrderType

log = logging.getLogger(__name__)


class PaperRunner:
    """Wires a Broker + Strategy + RiskManager into a live trading loop.

    Use it against a demo broker account for paper trading; flip the broker to
    a live account to go live (with the same code path, intentionally).
    """

    def __init__(self, cfg: AppConfig, broker: Broker, strategy: Strategy) -> None:
        self.cfg = cfg
        self.broker = broker
        self.strategy = strategy
        self.risk = RiskManager(cfg.risk)
        self._stopped = asyncio.Event()

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        await self.broker.connect()
        try:
            self.strategy.on_start()
            symbol = self.cfg.instrument.symbol
            bar_seconds = int(self.cfg.strategy.params.get("bar_seconds", 60))
            log.info("Streaming %s bars (%ds) for strategy=%s",
                     symbol, bar_seconds, self.strategy.name)
            async for bar in self.broker.stream_bars(symbol, bar_seconds):
                if self._stopped.is_set():
                    break
                position = await self.broker.get_position(symbol)
                equity = await self.broker.get_account_equity()
                self.risk.on_equity(equity)
                try:
                    signal = self.risk.vet_signal(self.strategy.on_bar(bar, position), position)
                except RiskRejection as e:
                    log.warning("risk halted: %s", e)
                    break
                order = self.risk.order_qty_for(signal, position)
                if order is None:
                    continue
                side, qty = order
                log.info("placing %s %d %s — %s", side.value, qty, symbol, signal.reason)
                await self.broker.place_order(symbol, side, qty, OrderType.MARKET)
        finally:
            self.strategy.on_stop()
            await self.broker.close()
