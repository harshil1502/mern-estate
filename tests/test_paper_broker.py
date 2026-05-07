from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from futures_bot.brokers.paper import PaperBroker
from futures_bot.config import InstrumentConfig
from futures_bot.types import Bar, OrderType, Side
from tests.conftest import make_bars


def _instr(tick_size: float = 0.25, point_value: float = 5.0) -> InstrumentConfig:
    return InstrumentConfig(
        symbol="MES", exchange="CME", tick_size=tick_size, point_value=point_value
    )


async def _aiter(bars: list[Bar]) -> AsyncIterator[Bar]:
    for b in bars:
        yield b


@pytest.mark.asyncio
async def test_market_buy_fills_at_next_bar_open_with_slippage():
    bars = make_bars([100.0, 101.0, 102.0])
    bars = [
        Bar(ts=bars[0].ts, open=100.0, high=100.5, low=99.5, close=100.0, volume=1.0),
        Bar(ts=bars[1].ts, open=101.0, high=101.5, low=100.5, close=101.0, volume=1.0),
        Bar(ts=bars[2].ts, open=102.0, high=102.5, low=101.5, close=102.0, volume=1.0),
    ]
    broker = PaperBroker(_instr(), _aiter(bars), initial_equity=10_000.0, slippage_ticks=2)
    await broker.connect()

    stream = broker.stream_bars("MES", 60).__aiter__()
    first = await stream.__anext__()
    assert first.open == 100.0  # bar 1 — no fill yet, position is flat

    await broker.place_order("MES", Side.BUY, 1, OrderType.MARKET)
    second = await stream.__anext__()
    assert second.open == 101.0
    pos = await broker.get_position("MES")
    # Slippage 2 ticks * 0.25 = 0.5; buy fills at 101.0 + 0.5 = 101.5.
    assert pos.qty == 1
    assert pos.avg_price == 101.5
    assert len(broker.state.fills) == 1
    assert broker.state.fills[0].price == 101.5


@pytest.mark.asyncio
async def test_round_trip_realizes_pnl_and_pays_commission():
    bars = [
        Bar(ts=make_bars([0])[0].ts, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        Bar(ts=make_bars([1])[0].ts, open=110.0, high=110.0, low=110.0, close=110.0, volume=1.0),
        Bar(ts=make_bars([2])[0].ts, open=120.0, high=120.0, low=120.0, close=120.0, volume=1.0),
    ]
    broker = PaperBroker(
        _instr(), _aiter(bars),
        initial_equity=10_000.0, slippage_ticks=0, commission_per_contract=2.5,
    )
    await broker.connect()

    stream = broker.stream_bars("MES", 60).__aiter__()
    await stream.__anext__()  # bar 0

    await broker.place_order("MES", Side.BUY, 1, OrderType.MARKET)
    await stream.__anext__()  # bar 1: fill buy at 110

    await broker.place_order("MES", Side.SELL, 1, OrderType.MARKET)
    await stream.__anext__()  # bar 2: fill sell at 120

    pos = await broker.get_position("MES")
    assert pos.qty == 0
    # PnL: (120-110) * 1 * 5 = 50; commissions: 2 * 2.5 = 5; net = 45.
    assert broker.state.realized_pnl == pytest.approx(45.0)
    equity = await broker.get_account_equity()
    assert equity == pytest.approx(10_045.0)


@pytest.mark.asyncio
async def test_unrealized_pnl_reflected_in_equity_while_open():
    bars = [
        Bar(ts=make_bars([0])[0].ts, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        Bar(ts=make_bars([1])[0].ts, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
        Bar(ts=make_bars([2])[0].ts, open=100.0, high=100.0, low=100.0, close=110.0, volume=1.0),
    ]
    broker = PaperBroker(_instr(), _aiter(bars), initial_equity=10_000.0, slippage_ticks=0)
    await broker.connect()

    stream = broker.stream_bars("MES", 60).__aiter__()
    await stream.__anext__()
    await broker.place_order("MES", Side.BUY, 1, OrderType.MARKET)
    await stream.__anext__()  # fill at 100
    await stream.__anext__()  # close moves to 110 (still long)

    equity = await broker.get_account_equity()
    # Long 1 @ 100, mark 110 — unrealized = 10 * 5 = 50.
    assert equity == pytest.approx(10_050.0)


@pytest.mark.asyncio
async def test_limit_order_not_supported():
    broker = PaperBroker(_instr(), _aiter([]), initial_equity=10_000.0)
    await broker.connect()
    with pytest.raises(NotImplementedError):
        await broker.place_order("MES", Side.BUY, 1, OrderType.LIMIT, limit_price=100.0)
