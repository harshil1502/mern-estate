from __future__ import annotations

import pytest

from futures_bot.config import RiskConfig
from futures_bot.execution.risk import RiskManager, RiskRejection
from futures_bot.types import Position, Side, Signal


def _cfg(**overrides) -> RiskConfig:
    base = dict(
        max_position=1,
        per_trade_risk_usd=50,
        daily_loss_limit_usd=200,
        kill_switch_drawdown_usd=500,
    )
    base.update(overrides)
    return RiskConfig(**base)


def test_clamps_target_qty_to_max_position():
    rm = RiskManager(_cfg(max_position=2))
    pos = Position(symbol="MES")
    sig = Signal(side=Side.BUY, target_qty=5, reason="aggressive")
    out = rm.vet_signal(sig, pos)
    assert out.target_qty == 2


def test_kill_switch_halts_on_drawdown():
    rm = RiskManager(_cfg(kill_switch_drawdown_usd=100))
    rm.on_equity(10_000.0)
    rm.on_equity(9_800.0)  # 200 dd > 100 limit
    pos = Position(symbol="MES")
    with pytest.raises(RiskRejection):
        rm.vet_signal(Signal(side=Side.BUY, target_qty=1), pos)


def test_order_qty_for_returns_delta():
    rm = RiskManager(_cfg())
    pos = Position(symbol="MES", qty=0)
    out = rm.order_qty_for(Signal(side=Side.BUY, target_qty=1), pos)
    assert out == (Side.BUY, 1)

    pos = Position(symbol="MES", qty=1)
    out = rm.order_qty_for(Signal(side=Side.SELL, target_qty=0), pos)
    assert out == (Side.SELL, 1)

    out = rm.order_qty_for(Signal(side=Side.BUY, target_qty=1), Position(symbol="MES", qty=1))
    assert out is None
