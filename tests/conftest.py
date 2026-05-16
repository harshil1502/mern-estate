from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_bot.types import Bar


def make_bars(prices: list[float], start: datetime | None = None, step_s: int = 60) -> list[Bar]:
    t0 = start or datetime(2025, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i, p in enumerate(prices):
        bars.append(
            Bar(
                ts=t0 + timedelta(seconds=i * step_s),
                open=p,
                high=p,
                low=p,
                close=p,
                volume=1.0,
            )
        )
    return bars
