from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from futures_bot.config import RiskConfig
from futures_bot.types import Position, Side, Signal

log = logging.getLogger(__name__)


class RiskRejection(Exception):
    """Raised when risk limits forbid a signal from becoming an order."""


@dataclass
class RiskState:
    realized_pnl_today: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    last_pnl_date: date = field(default_factory=lambda: datetime.now(UTC).date())
    halted: bool = False


class RiskManager:
    """Enforces position caps, daily loss, and a kill switch on equity drawdown."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.state = RiskState()

    # --- equity / pnl tracking ---------------------------------------------

    def on_equity(self, equity: float) -> None:
        if self.state.peak_equity == 0.0:
            self.state.peak_equity = equity
            self.state.trough_equity = equity
            return
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.trough_equity = min(self.state.trough_equity, equity)
        drawdown = self.state.peak_equity - equity
        if drawdown >= self.cfg.kill_switch_drawdown_usd:
            log.error("Kill switch tripped: drawdown=%.2f >= %.2f",
                      drawdown, self.cfg.kill_switch_drawdown_usd)
            self.state.halted = True

    def on_realized_pnl(self, pnl: float) -> None:
        today = datetime.now(UTC).date()
        if today != self.state.last_pnl_date:
            self.state.realized_pnl_today = 0.0
            self.state.last_pnl_date = today
        self.state.realized_pnl_today += pnl
        if self.state.realized_pnl_today <= -self.cfg.daily_loss_limit_usd:
            log.error("Daily loss limit hit: %.2f <= -%.2f",
                      self.state.realized_pnl_today, self.cfg.daily_loss_limit_usd)
            self.state.halted = True

    # --- pre-trade checks ---------------------------------------------------

    def vet_signal(self, signal: Signal, position: Position) -> Signal:
        if self.state.halted:
            raise RiskRejection("risk halted")
        if signal.target_qty is None or signal.side is None:
            return signal
        target = signal.target_qty
        if abs(target) > self.cfg.max_position:
            log.warning("Clamping target qty %d -> %d (max_position)", target,
                        self.cfg.max_position)
            clamped = self.cfg.max_position if target > 0 else -self.cfg.max_position
            return Signal(side=signal.side, target_qty=clamped, reason=signal.reason)
        return signal

    def order_qty_for(self, signal: Signal, position: Position) -> tuple[Side, int] | None:
        """Translate a target-position signal into the (side, qty) delta to send."""
        if signal.target_qty is None:
            return None
        delta = signal.target_qty - position.qty
        if delta == 0:
            return None
        return (Side.BUY if delta > 0 else Side.SELL, abs(delta))
