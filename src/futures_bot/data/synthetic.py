"""Deterministic synthetic bar generator (geometric Brownian motion).

Uses GBM with parameters tuned to roughly match daily characteristics of
front-month equity index futures (ES/MES/NQ/MNQ). Seeded for reproducibility
so a given (seed, params) always produces the same bars — useful for
parameter sweeps and regression tests.

Defaults assume a 1-minute bar and ~16% annualized vol, which is typical for
the S&P 500. Override `annual_vol` for other instruments.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

from futures_bot.types import Bar

_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def synthetic_bars(
    n: int = 500,
    start_price: float = 4500.0,
    bar_seconds: int = 60,
    annual_drift: float = 0.05,
    annual_vol: float = 0.16,
    tick_size: float = 0.25,
    intra_bar_range_ticks: int = 4,
    seed: int = 42,
    start: datetime | None = None,
) -> list[Bar]:
    """Generate `n` synthetic bars with realistic open/high/low/close geometry.

    Args:
        n: number of bars
        start_price: opening price of the first bar
        bar_seconds: bar duration; sets the dt for GBM scaling
        annual_drift: drift mu (e.g. 0.05 = 5% / yr)
        annual_vol: vol sigma (e.g. 0.16 = 16% / yr)
        tick_size: minimum price increment; bars are snapped to this
        intra_bar_range_ticks: typical hi-lo spread in ticks (random per bar)
        seed: PRNG seed for reproducibility
        start: timestamp of the first bar (default: 2025-01-01 13:30 UTC)
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if annual_vol <= 0:
        raise ValueError("annual_vol must be > 0")

    rng = random.Random(seed)
    dt = bar_seconds / _SECONDS_PER_YEAR
    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    diffusion = annual_vol * math.sqrt(dt)

    t0 = start or datetime(2025, 1, 1, 13, 30, tzinfo=UTC)
    price = start_price
    bars: list[Bar] = []
    for i in range(n):
        z = rng.gauss(0.0, 1.0)
        new_price = price * math.exp(drift + diffusion * z)
        open_p = price
        close_p = new_price
        # Synthesize a plausible high/low around open/close.
        body_hi = max(open_p, close_p)
        body_lo = min(open_p, close_p)
        wick_size = rng.uniform(0.0, intra_bar_range_ticks) * tick_size
        high = body_hi + wick_size * rng.random()
        low = body_lo - wick_size * rng.random()
        volume = max(1.0, rng.gauss(200, 60))

        bars.append(
            Bar(
                ts=t0 + timedelta(seconds=i * bar_seconds),
                open=_snap(open_p, tick_size),
                high=_snap(high, tick_size),
                low=_snap(low, tick_size),
                close=_snap(close_p, tick_size),
                volume=round(volume, 0),
            )
        )
        price = new_price
    return bars


def _snap(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size
