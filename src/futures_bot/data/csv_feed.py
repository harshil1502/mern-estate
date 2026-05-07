"""CSV-backed bar replay for paper-trading simulations.

Expected CSV columns: ts (ISO-8601), open, high, low, close, volume. The
`speed_multiplier` controls real-time pacing — `0` plays as fast as possible
(no sleeps), `1` plays at wall-clock speed, `60` means 60x faster than
wall-clock.
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from pathlib import Path

from futures_bot.types import Bar


def list_csv_bars(path: str | Path) -> list[Bar]:
    """Eagerly load all bars from a CSV (used by the backtester)."""
    p = Path(path)
    bars: list[Bar] = []
    with p.open() as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames)
        for row in reader:
            bars.append(_row_to_bar(row))
    return bars


def write_bars_csv(path: str | Path, bars: Iterable[Bar]) -> int:
    """Write bars to CSV in the same schema list_csv_bars expects. Returns row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])
            n += 1
    return n


async def csv_bar_feed(
    path: str | Path,
    speed_multiplier: float = 0.0,
) -> AsyncIterator[Bar]:
    """Yield bars from a CSV, optionally pacing relative to bar timestamps."""
    bars = list_csv_bars(path)
    async for bar in iter_bars_paced(bars, speed_multiplier=speed_multiplier):
        yield bar


async def iter_bars_paced(
    bars: Iterable[Bar],
    speed_multiplier: float = 0.0,
) -> AsyncIterator[Bar]:
    prev_ts: datetime | None = None
    for bar in bars:
        if speed_multiplier > 0 and prev_ts is not None:
            wait_s = (bar.ts - prev_ts).total_seconds() / speed_multiplier
            if wait_s > 0:
                await asyncio.sleep(wait_s)
        prev_ts = bar.ts
        yield bar


_REQUIRED = {"ts", "open", "high", "low", "close", "volume"}


def _validate_columns(fields: Iterable[str] | None) -> None:
    if fields is None:
        raise ValueError("CSV has no header row")
    missing = _REQUIRED - set(fields)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")


def _row_to_bar(row: dict[str, str]) -> Bar:
    return Bar(
        ts=datetime.fromisoformat(row["ts"].replace("Z", "+00:00")),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )
