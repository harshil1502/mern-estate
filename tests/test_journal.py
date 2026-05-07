from __future__ import annotations

import json
from pathlib import Path

from futures_bot.brokers.paper import Fill
from futures_bot.journal import TradeJournal
from futures_bot.types import Side
from tests.conftest import make_bars


def test_journal_writes_jsonl(tmp_path: Path):
    p = tmp_path / "journal.jsonl"
    fill = Fill(
        ts=make_bars([0])[0].ts,
        side=Side.BUY,
        qty=1,
        price=100.5,
        realized_pnl=0.0,
        position_after=1,
        equity_after=10_000.0,
    )
    with TradeJournal(p) as j:
        j.record(fill)
        j.record({"event": "halt", "reason": "kill_switch"})

    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["side"] == "BUY"
    assert first["price"] == 100.5
    second = json.loads(lines[1])
    assert second == {"event": "halt", "reason": "kill_switch"}


def test_journal_appends_across_sessions(tmp_path: Path):
    p = tmp_path / "j.jsonl"
    with TradeJournal(p) as j:
        j.record({"a": 1})
    with TradeJournal(p) as j:
        j.record({"a": 2})
    lines = p.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]
