from __future__ import annotations

from pathlib import Path

import pytest

from futures_bot.data.csv_feed import csv_bar_feed, list_csv_bars


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "bars.csv"
    p.write_text("\n".join(["ts,open,high,low,close,volume", *rows]) + "\n")
    return p


def test_list_csv_bars_parses_iso_timestamps(tmp_path: Path):
    p = _write_csv(tmp_path, [
        "2025-01-01T13:30:00Z,100.0,100.5,99.5,100.25,10",
        "2025-01-01T13:31:00Z,100.25,100.75,100.0,100.5,12",
    ])
    bars = list_csv_bars(p)
    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[0].close == 100.25
    assert bars[1].volume == 12.0


def test_csv_missing_required_column_raises(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("ts,open,high,low,close\n2025-01-01T13:30:00Z,1,1,1,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        list_csv_bars(p)


@pytest.mark.asyncio
async def test_csv_bar_feed_yields_in_order(tmp_path: Path):
    p = _write_csv(tmp_path, [
        "2025-01-01T13:30:00Z,100,100,100,100,1",
        "2025-01-01T13:31:00Z,101,101,101,101,1",
        "2025-01-01T13:32:00Z,102,102,102,102,1",
    ])
    closes = [b.close async for b in csv_bar_feed(p, speed_multiplier=0)]
    assert closes == [100.0, 101.0, 102.0]
