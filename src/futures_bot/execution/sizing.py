"""Position sizing layer.

A PositionSizer slots between the strategy and the risk manager: the strategy
emits an *intent* (long/short with target_qty=±1), the sizer translates that
into an actual contract count using account equity + market state, and the
risk manager has the final say on caps. This is where most of the real
alpha-to-PnL conversion lives.

Built-ins:
  - FixedQty: trade exactly N contracts every time (legacy default).
  - FixedFractionalATR: risk a fixed % of equity per trade, sized by ATR.
  - VolatilityTarget: scale qty so per-bar PnL volatility hits a target.
"""

from __future__ import annotations

import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import deque

from futures_bot.config import InstrumentConfig
from futures_bot.types import Bar, Position, Signal

log = logging.getLogger(__name__)


class PositionSizer(ABC):
    """Translates a directional intent into a target contract count."""

    def update(self, bar: Bar) -> None:
        """Called by the runner on every bar before `size` is invoked."""

    @abstractmethod
    def size(
        self,
        signal: Signal,
        position: Position,
        equity: float,
        instrument: InstrumentConfig,
    ) -> Signal: ...


# --- helpers -----------------------------------------------------------------


def true_range(bar: Bar, prev_close: float | None) -> float:
    if prev_close is None:
        return bar.high - bar.low
    return max(
        bar.high - bar.low,
        abs(bar.high - prev_close),
        abs(bar.low - prev_close),
    )


def atr(bars: list[Bar] | deque[Bar], period: int = 14) -> float:
    """Simple-mean ATR over the last `period` bars (0.0 if not enough data)."""
    bars = list(bars)
    if len(bars) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i], bars[i - 1].close))
    if not trs:
        return 0.0
    if len(trs) < period:
        return statistics.fmean(trs)
    return statistics.fmean(trs[-period:])


# --- sizers ------------------------------------------------------------------


class FixedQty(PositionSizer):
    def __init__(self, qty: int = 1) -> None:
        if qty < 1:
            raise ValueError("FixedQty.qty must be >= 1")
        self.qty = qty

    def size(
        self,
        signal: Signal,
        position: Position,
        equity: float,
        instrument: InstrumentConfig,
    ) -> Signal:
        if signal.target_qty is None:
            return signal
        if signal.target_qty == 0:
            return signal
        sign = 1 if signal.target_qty > 0 else -1
        return Signal(side=signal.side, target_qty=sign * self.qty, reason=signal.reason)


class FixedFractionalATR(PositionSizer):
    """Risk a fixed fraction of equity per trade, with stop = atr_stop_mult × ATR.

    qty = floor( equity * risk_per_trade_pct / (ATR * atr_stop_mult * point_value) )
    Falls back to qty=1 if ATR isn't available yet.
    """

    def __init__(
        self,
        risk_per_trade_pct: float = 0.005,  # 0.5% of equity per trade
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_qty: int = 50,
        min_qty: int = 1,
    ) -> None:
        if not 0 < risk_per_trade_pct < 1:
            raise ValueError("risk_per_trade_pct must be in (0, 1)")
        if atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be > 0")
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_qty = max_qty
        self.min_qty = min_qty
        self._bars: deque[Bar] = deque(maxlen=atr_period + 1)

    def update(self, bar: Bar) -> None:
        self._bars.append(bar)

    def size(
        self,
        signal: Signal,
        position: Position,
        equity: float,
        instrument: InstrumentConfig,
    ) -> Signal:
        if signal.target_qty is None:
            return signal
        if signal.target_qty == 0:
            return signal  # exit signal — qty is fine as-is

        sign = 1 if signal.target_qty > 0 else -1
        a = atr(self._bars, self.atr_period)
        if a <= 0 or equity <= 0:
            return Signal(
                side=signal.side, target_qty=sign * self.min_qty,
                reason=signal.reason + " | atr=na",
            )

        risk_dollars = equity * self.risk_per_trade_pct
        risk_per_contract = a * self.atr_stop_mult * instrument.point_value
        if risk_per_contract <= 0:
            return Signal(
                side=signal.side, target_qty=sign * self.min_qty, reason=signal.reason,
            )

        raw_qty = int(risk_dollars // risk_per_contract)
        qty = max(self.min_qty, min(self.max_qty, raw_qty))
        return Signal(
            side=signal.side,
            target_qty=sign * qty,
            reason=f"{signal.reason} | atr={a:.2f} risk=${risk_dollars:.0f} qty={qty}",
        )


class VolatilityTarget(PositionSizer):
    """Scale position so a 1-bar PnL std targets `bar_std_target_usd`.

    Estimates per-contract bar PnL std from ATR (a coarse but workable proxy).
    qty = floor( bar_std_target_usd / (ATR * point_value) )
    """

    def __init__(
        self,
        bar_std_target_usd: float = 100.0,
        atr_period: int = 20,
        max_qty: int = 50,
        min_qty: int = 1,
    ) -> None:
        if bar_std_target_usd <= 0:
            raise ValueError("bar_std_target_usd must be > 0")
        if atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        self.target = bar_std_target_usd
        self.atr_period = atr_period
        self.max_qty = max_qty
        self.min_qty = min_qty
        self._bars: deque[Bar] = deque(maxlen=atr_period + 1)

    def update(self, bar: Bar) -> None:
        self._bars.append(bar)

    def size(
        self,
        signal: Signal,
        position: Position,
        equity: float,
        instrument: InstrumentConfig,
    ) -> Signal:
        if signal.target_qty is None or signal.target_qty == 0:
            return signal
        sign = 1 if signal.target_qty > 0 else -1
        a = atr(self._bars, self.atr_period)
        per_contract = a * instrument.point_value
        if per_contract <= 0:
            return Signal(
                side=signal.side, target_qty=sign * self.min_qty, reason=signal.reason,
            )
        raw_qty = int(math.floor(self.target / per_contract))
        qty = max(self.min_qty, min(self.max_qty, raw_qty))
        return Signal(
            side=signal.side,
            target_qty=sign * qty,
            reason=f"{signal.reason} | vol_target qty={qty}",
        )


# --- factory -----------------------------------------------------------------


def build_sizer(spec: dict) -> PositionSizer:
    """Construct a sizer from a config dict like {"type": "atr", "risk_per_trade_pct": 0.005}."""
    if not spec:
        return FixedQty()
    t = spec.get("type", "fixed").lower()
    params = {k: v for k, v in spec.items() if k != "type"}
    if t in ("fixed", "fixed_qty", "fixedqty"):
        return FixedQty(**params)
    if t in ("atr", "fixed_fractional_atr"):
        return FixedFractionalATR(**params)
    if t in ("vol_target", "volatility_target"):
        return VolatilityTarget(**params)
    raise ValueError(f"unknown sizer type: {t!r}")
