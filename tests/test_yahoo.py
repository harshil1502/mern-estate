from __future__ import annotations

import sys
import types
from datetime import datetime

import pytest

from futures_bot.data.yahoo import YahooFetchError, fetch_yahoo_bars


def _install_fake_yfinance(monkeypatch, df):
    fake = types.SimpleNamespace(download=lambda *a, **kw: df)
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        fetch_yahoo_bars("ES=F", interval="bogus")


def test_empty_response_raises(monkeypatch):
    pd = pytest.importorskip("pandas")
    _install_fake_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(YahooFetchError, match="empty response"):
        fetch_yahoo_bars("ES=F")


def test_parses_well_formed_df(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.to_datetime(["2025-05-04T13:30:00Z", "2025-05-04T13:31:00Z"])
    df = pd.DataFrame(
        {
            "Open": [4500.0, 4500.5],
            "High": [4501.0, 4501.0],
            "Low": [4499.5, 4500.25],
            "Close": [4500.5, 4500.75],
            "Volume": [120, 130],
        },
        index=idx,
    )
    _install_fake_yfinance(monkeypatch, df)
    bars = fetch_yahoo_bars("ES=F", interval="1m", period="1d")
    assert len(bars) == 2
    assert bars[0].open == 4500.0
    assert bars[0].close == 4500.5
    assert bars[1].volume == 130.0
    assert isinstance(bars[0].ts, datetime)


def test_skips_nan_rows(monkeypatch):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    idx = pd.to_datetime(["2025-05-04T13:30:00Z", "2025-05-04T13:31:00Z"])
    df = pd.DataFrame(
        {
            "Open": [4500.0, np.nan],
            "High": [4501.0, np.nan],
            "Low": [4499.5, np.nan],
            "Close": [4500.5, np.nan],
            "Volume": [120, np.nan],
        },
        index=idx,
    )
    _install_fake_yfinance(monkeypatch, df)
    bars = fetch_yahoo_bars("ES=F")
    assert len(bars) == 1
    assert bars[0].open == 4500.0


def test_handles_multiindex_columns(monkeypatch):
    """yfinance returns a MultiIndex when symbols contain special chars (e.g. '=F')."""
    pd = pytest.importorskip("pandas")
    idx = pd.to_datetime(["2025-05-04T13:30:00Z"])
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["ES=F"]]
    )
    df = pd.DataFrame([[4500.0, 4501.0, 4499.5, 4500.5, 120]], columns=cols, index=idx)
    _install_fake_yfinance(monkeypatch, df)
    bars = fetch_yahoo_bars("ES=F")
    assert len(bars) == 1
    assert bars[0].close == 4500.5
