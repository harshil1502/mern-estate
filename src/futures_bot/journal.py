"""Append-only JSONL trade journal.

One line per fill, flushed immediately so a crash mid-session still leaves a
recoverable record. Reading back is pandas-friendly: `pd.read_json(path, lines=True)`.
"""

from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)


class TradeJournal(AbstractContextManager["TradeJournal"]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO[str] | None = None

    def __enter__(self) -> TradeJournal:
        self._f = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None

    def record(self, entry: Any) -> None:
        if self._f is None:
            raise RuntimeError("TradeJournal must be used as a context manager")
        line = json.dumps(_to_jsonable(entry), default=_default_json) + "\n"
        self._f.write(line)
        self._f.flush()

    def __call__(self, entry: Any) -> None:
        """Allow the journal itself to be used as an `on_fill` callback."""
        self.record(entry)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _default_json(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):  # StrEnum
        return obj.value
    raise TypeError(f"not serializable: {type(obj).__name__}")
