from __future__ import annotations

from collections.abc import Callable
from typing import Any

from futures_bot.strategies.base import Strategy

_REGISTRY: dict[str, Callable[..., Strategy]] = {}


def register_strategy(name: str) -> Callable[[type[Strategy]], type[Strategy]]:
    def deco(cls: type[Strategy]) -> type[Strategy]:
        _REGISTRY[name] = cls
        return cls

    return deco


def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)
