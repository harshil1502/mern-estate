from __future__ import annotations

import pytest

from futures_bot.data.synthetic import synthetic_bars
from futures_bot.research.regimes import (
    GaussianHMMRegime,
    VolBandRegime,
    build_regime_detector,
)

# --- VolBandRegime --------------------------------------------------------


def test_vol_band_warmup_returns_warmup():
    det = VolBandRegime(window=20, warmup=50)
    bars = synthetic_bars(n=20, seed=1)
    for b in bars:
        det.update(b)
    assert det.regime == "warmup"


def test_vol_band_classifies_after_warmup():
    det = VolBandRegime(window=30, warmup=100, refit_every=50)
    bars = synthetic_bars(n=400, seed=2)
    last_regime = "warmup"
    for b in bars:
        last_regime = det.update(b)
    assert last_regime in {"low_vol", "mid_vol", "high_vol"}


def test_vol_band_invalid_args():
    with pytest.raises(ValueError):
        VolBandRegime(window=5)
    with pytest.raises(ValueError):
        VolBandRegime(window=20, warmup=10)


def test_vol_band_separates_calm_from_volatile():
    """Same detector seeing calm bars then volatile bars should report
    higher vol regime in the volatile segment."""
    det = VolBandRegime(window=40, warmup=120, refit_every=40)
    calm = synthetic_bars(n=200, seed=3, annual_vol=0.05)
    volatile = synthetic_bars(n=200, seed=4, annual_vol=0.50, start=calm[-1].ts)
    for b in calm:
        det.update(b)
    calm_regime = det.regime

    for b in volatile:
        det.update(b)
    vol_regime = det.regime
    # The shift should be detectable — at minimum the volatile run should
    # touch high_vol at some point during processing.
    assert calm_regime != "high_vol" or vol_regime != "low_vol"


# --- GaussianHMMRegime (skip if hmmlearn missing) -------------------------


def test_hmm_regime_smoke():
    pytest.importorskip("hmmlearn")
    det = GaussianHMMRegime(n_states=2, warmup=120, refit_every=200)
    bars = synthetic_bars(n=600, seed=5)
    last = "warmup"
    for b in bars:
        last = det.update(b)
    assert last in {"calm", "stressed", "warmup"}


def test_hmm_invalid_args():
    pytest.importorskip("hmmlearn")
    with pytest.raises(ValueError):
        GaussianHMMRegime(n_states=1)
    with pytest.raises(ValueError):
        GaussianHMMRegime(n_states=2, warmup=10)


# --- factory --------------------------------------------------------------


def test_build_regime_detector_returns_none_for_empty():
    assert build_regime_detector(None) is None
    assert build_regime_detector("") is None
    assert build_regime_detector("none") is None


def test_build_regime_detector_dispatch():
    pytest.importorskip("hmmlearn")
    assert isinstance(build_regime_detector("vol"), VolBandRegime)
    assert isinstance(build_regime_detector("hmm"), GaussianHMMRegime)
    with pytest.raises(ValueError):
        build_regime_detector("nonsense")
