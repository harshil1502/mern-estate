from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from futures_bot.config import InstrumentConfig, RiskConfig
from futures_bot.execution.risk import RiskManager, RiskRejection
from futures_bot.strategies.base import Strategy
from futures_bot.types import Bar, Position, Side

log = logging.getLogger(__name__)


@dataclass
class Trade:
    ts: datetime
    side: Side
    qty: int
    price: float
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    final_equity: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0

    @property
    def num_trades(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """Bar-by-bar event-driven backtester. Fills at next-bar open, no slippage.

    Deliberately minimal: no commissions, no partial fills, no funding.
    Replace with a richer model when a strategy needs it.
    """

    def __init__(
        self,
        strategy: Strategy,
        instrument: InstrumentConfig,
        risk: RiskConfig,
        initial_equity: float = 10_000.0,
    ) -> None:
        self.strategy = strategy
        self.instrument = instrument
        self.risk_cfg = risk
        self.initial_equity = initial_equity

    def run(self, bars: Iterable[Bar]) -> BacktestResult:
        position = Position(symbol=self.instrument.symbol)
        risk = RiskManager(self.risk_cfg)
        result = BacktestResult(final_equity=self.initial_equity)
        equity = self.initial_equity
        peak_equity = equity

        bars = list(bars)
        self.strategy.on_start()
        for i, bar in enumerate(bars):
            mark = bar.close
            unrealized = position.unrealized_pnl(mark, self.instrument.point_value)
            equity = self.initial_equity + result.realized_pnl + unrealized
            peak_equity = max(peak_equity, equity)
            result.max_drawdown = max(result.max_drawdown, peak_equity - equity)
            result.equity_curve.append((bar.ts, equity))
            risk.on_equity(equity)

            try:
                signal = risk.vet_signal(self.strategy.on_bar(bar, position), position)
            except RiskRejection as e:
                log.info("risk rejected at %s: %s", bar.ts, e)
                continue

            order = risk.order_qty_for(signal, position)
            if order is None:
                continue

            # Fill at next bar's open if available, else this bar's close.
            fill_price = bars[i + 1].open if i + 1 < len(bars) else bar.close
            side, qty = order
            self._apply_fill(position, side, qty, fill_price, result, risk, bar.ts)

        self.strategy.on_stop()
        result.final_equity = equity
        return result

    # --- helpers ------------------------------------------------------------

    def _apply_fill(
        self,
        position: Position,
        side: Side,
        qty: int,
        price: float,
        result: BacktestResult,
        risk: RiskManager,
        ts: datetime,
    ) -> None:
        signed = qty if side is Side.BUY else -qty
        prev_qty = position.qty
        new_qty = prev_qty + signed
        realized = 0.0

        if prev_qty == 0:
            position.qty = new_qty
            position.avg_price = price
        elif (prev_qty > 0 and signed > 0) or (prev_qty < 0 and signed < 0):
            # adding to position
            total_cost = position.avg_price * abs(prev_qty) + price * abs(signed)
            position.qty = new_qty
            position.avg_price = total_cost / abs(new_qty) if new_qty != 0 else 0.0
        else:
            # reducing or flipping
            closing_qty = min(abs(prev_qty), abs(signed))
            direction = 1 if prev_qty > 0 else -1
            pv = self.instrument.point_value
            realized = (price - position.avg_price) * closing_qty * direction * pv
            result.realized_pnl += realized
            risk.on_realized_pnl(realized)
            position.qty = new_qty
            if new_qty == 0:
                position.avg_price = 0.0
            elif (prev_qty > 0) != (new_qty > 0):
                # flipped sides — remainder opens at fill price
                position.avg_price = price

        result.trades.append(Trade(ts=ts, side=side, qty=qty, price=price, realized_pnl=realized))
