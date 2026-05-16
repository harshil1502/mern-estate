"""Yahoo Finance bar fetcher.

Yahoo's continuous front-month futures symbols (ES=F, MES=F, NQ=F, MNQ=F, CL=F, GC=F)
are usable for paper-sim and walk-forward analysis. Constraints worth knowing:

  - 1m bars: only ~7 calendar days of history
  - 2m/5m/15m: up to 60 days
  - 1h: up to 730 days
  - 1d: full history

Yahoo data is *not* exchange-quality — gaps, late prints, and volume that doesn't
match CME tape are common. Use it for end-to-end plumbing checks and rough
strategy screening, not for serious backtests.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from futures_bot.types import Bar

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"}


class YahooFetchError(RuntimeError):
    pass


def fetch_yahoo_bars(symbol: str, interval: str = "1m", period: str = "5d") -> list[Bar]:
    """Pull bars from Yahoo Finance into our internal Bar format.

    Args:
        symbol: e.g. "ES=F", "MES=F", "NQ=F".
        interval: 1m | 2m | 5m | 15m | 30m | 60m | 1h | 1d | 1wk | 1mo.
        period: 1d | 5d | 1mo | 3mo | 6mo | 1y | 2y | 5y | 10y | ytd | max.

    Raises YahooFetchError on empty/malformed responses (rate limiting,
    invalid symbol, etc.).
    """
    if interval not in _VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(_VALID_INTERVALS)}")

    try:
        import yfinance as yf
    except ImportError as e:
        raise YahooFetchError(
            "yfinance is not installed. Install with: pip install yfinance"
        ) from e

    log.info("Fetching %s %s/%s from Yahoo", symbol, interval, period)
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        raise YahooFetchError(
            f"empty response for {symbol} ({interval}/{period}); "
            f"likely rate-limited or invalid symbol"
        )
    return _df_to_bars(df, symbol)


def _df_to_bars(df, symbol: str) -> list[Bar]:
    """Convert a yfinance DataFrame to our Bar list. Handles MultiIndex columns."""
    # yfinance returns MultiIndex columns when the symbol has special chars.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df = df.droplevel(1, axis=1)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise YahooFetchError(f"yfinance response missing columns: {sorted(missing)}")

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        if hasattr(ts, "to_pydatetime"):
            ts_py = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            ts_py = ts
        else:
            ts_py = datetime.fromisoformat(str(ts))
        # yfinance can return NaN rows on partial bars — skip them.
        if any(_is_nan(row[c]) for c in ("Open", "High", "Low", "Close")):
            continue
        bars.append(
            Bar(
                ts=ts_py,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]) if not _is_nan(row["Volume"]) else 0.0,
            )
        )
    if not bars:
        raise YahooFetchError(f"all rows for {symbol} were NaN — try a different period")
    return bars


def _is_nan(x) -> bool:
    try:
        return x != x  # NaN is the only value where this is True
    except Exception:
        return False
