from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamGrid:
    """A strategy name + a dict of param -> list of values to sweep."""

    strategy_name: str
    params: dict[str, list[Any]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)

    def expand(self) -> Iterator[dict[str, Any]]:
        if not self.params:
            yield dict(self.fixed)
            return
        keys = list(self.params.keys())
        for combo in itertools.product(*[self.params[k] for k in keys]):
            d = dict(self.fixed)
            d.update(dict(zip(keys, combo, strict=True)))
            yield d

    def size(self) -> int:
        if not self.params:
            return 1
        n = 1
        for vs in self.params.values():
            n *= len(vs)
        return n
