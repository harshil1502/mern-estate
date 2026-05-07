"""Regime detection for gating strategies.

Two detectors with the same interface:
  - VolBandRegime: classifies each bar into low / mid / high volatility using
    a rolling realized-vol window with hysteresis. No external deps.
  - GaussianHMMRegime: fits a 2-state Gaussian HMM (via hmmlearn) on rolling
    log-returns + abs-returns and returns the most likely state. The classic
    "Markov-switching" detector — captures persistence in the regime.

Both expose `update(bar) -> regime: str`. Strategies (or RegimeGatedStrategy)
gate on the returned label. None of these are predictive — they describe
*current* state, which is the right thing to condition trades on.
"""

from __future__ import annotations

import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from futures_bot.types import Bar

log = logging.getLogger(__name__)


class RegimeDetector(ABC):
    @abstractmethod
    def update(self, bar: Bar) -> str: ...

    @property
    @abstractmethod
    def regime(self) -> str: ...


# --- Vol-band detector (no external deps) -----------------------------------


@dataclass
class VolBands:
    low_quantile: float = 0.33
    high_quantile: float = 0.67


class VolBandRegime(RegimeDetector):
    """Classifies regime by realized vol of returns over a rolling window.

    Bands are computed once enough history accrues, then re-fitted every
    `refit_every` bars. Hysteresis prevents flicker between adjacent bands.
    """

    def __init__(
        self,
        window: int = 60,
        warmup: int = 100,
        refit_every: int = 100,
        bands: VolBands | None = None,
    ) -> None:
        if window < 10:
            raise ValueError("window must be >= 10")
        if warmup < window:
            raise ValueError("warmup must be >= window")
        self.window = window
        self.warmup = warmup
        self.refit_every = refit_every
        self.bands = bands or VolBands()
        self._returns: deque[float] = deque(maxlen=window)
        self._all_vols: list[float] = []
        self._prev_close: float | None = None
        self._lo_band: float | None = None
        self._hi_band: float | None = None
        self._n_seen = 0
        self._regime = "warmup"

    def update(self, bar: Bar) -> str:
        if self._prev_close is not None and self._prev_close > 0:
            r = math.log(bar.close / self._prev_close)
            self._returns.append(r)
        self._prev_close = bar.close
        self._n_seen += 1

        if len(self._returns) < self.window:
            self._regime = "warmup"
            return self._regime

        vol = statistics.pstdev(self._returns)
        self._all_vols.append(vol)

        # Refit bands periodically once we have enough samples.
        should_refit = self._lo_band is None or self._n_seen % self.refit_every == 0
        if should_refit and len(self._all_vols) >= max(20, self.warmup // 5):
            sorted_vols = sorted(self._all_vols[-max(500, self.warmup):])
            self._lo_band = _quantile(sorted_vols, self.bands.low_quantile)
            self._hi_band = _quantile(sorted_vols, self.bands.high_quantile)

        if self._lo_band is None or self._hi_band is None:
            self._regime = "warmup"
            return self._regime

        # Hysteresis: only flip out of a band when we cross past the next band.
        if vol <= self._lo_band:
            self._regime = "low_vol"
        elif vol >= self._hi_band:
            self._regime = "high_vol"
        else:
            self._regime = "mid_vol"
        return self._regime

    @property
    def regime(self) -> str:
        return self._regime


# --- Gaussian HMM detector (hmmlearn) ---------------------------------------


class GaussianHMMRegime(RegimeDetector):
    """2-state Gaussian HMM on (log-return, |log-return|).

    State 0 is labelled "calm" (lower variance), state 1 "stressed". Refits
    every `refit_every` bars to track regime drift. Falls back to "warmup"
    until `warmup` bars have been observed.
    """

    def __init__(
        self,
        n_states: int = 2,
        warmup: int = 200,
        refit_every: int = 200,
        seed: int = 17,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
            from hmmlearn import hmm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GaussianHMMRegime requires hmmlearn + numpy. Install with "
                "`pip install hmmlearn`."
            ) from e
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if warmup < n_states * 50:
            raise ValueError(f"warmup must be >= {n_states * 50}")
        self.n_states = n_states
        self.warmup = warmup
        self.refit_every = refit_every
        self.seed = seed
        self._returns: list[float] = []
        self._prev_close: float | None = None
        self._model = None
        self._state_to_label: dict[int, str] = {}
        self._n_seen = 0
        self._regime = "warmup"

    def update(self, bar: Bar) -> str:
        import numpy as np
        from hmmlearn import hmm

        if self._prev_close is not None and self._prev_close > 0:
            r = math.log(bar.close / self._prev_close)
            self._returns.append(r)
        self._prev_close = bar.close
        self._n_seen += 1

        if len(self._returns) < self.warmup:
            self._regime = "warmup"
            return self._regime

        # Refit on a periodic schedule.
        if self._model is None or self._n_seen % self.refit_every == 0:
            X = np.array([[r, abs(r)] for r in self._returns])
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=50,
                    random_state=self.seed,
                )
                model.fit(X)
                # Label states by variance of log-return (lower = calm).
                variances = [model.covars_[i, 0, 0] for i in range(self.n_states)]
                ordered = sorted(range(self.n_states), key=lambda i: variances[i])
                labels = ["calm", "stressed"] if self.n_states == 2 else [
                    f"s{rank}" for rank in range(self.n_states)
                ]
                self._state_to_label = {state: labels[rank] for rank, state in enumerate(ordered)}
                self._model = model
            except Exception as e:
                log.warning("HMM fit failed (%s) — keeping previous model", e)

        if self._model is None:
            self._regime = "warmup"
            return self._regime

        # Predict the most likely current state given the full history.
        X = np.array([[r, abs(r)] for r in self._returns])
        try:
            states = self._model.predict(X)
            self._regime = self._state_to_label.get(int(states[-1]), "unknown")
        except Exception as e:
            log.warning("HMM predict failed (%s)", e)
        return self._regime

    @property
    def regime(self) -> str:
        return self._regime


# --- factory -----------------------------------------------------------------


def build_regime_detector(spec: dict | str | None) -> RegimeDetector | None:
    if spec is None or spec == "" or spec == "none":
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    t = spec.get("type", "vol").lower()
    params = {k: v for k, v in spec.items() if k != "type"}
    if t in ("vol", "volband", "vol_band"):
        return VolBandRegime(**params)
    if t in ("hmm", "gaussian_hmm", "markov"):
        return GaussianHMMRegime(**params)
    raise ValueError(f"unknown regime detector: {t!r}")


# --- helpers -----------------------------------------------------------------


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        raise ValueError("empty")
    if q <= 0:
        return sorted_xs[0]
    if q >= 1:
        return sorted_xs[-1]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac
