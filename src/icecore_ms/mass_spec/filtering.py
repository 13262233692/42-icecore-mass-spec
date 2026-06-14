"""
High-level Python interface for LA-ICP-MS isotope signal processing.

Wraps the C++ native extension for frequency-domain adaptive Kalman filtering
with a clean, NumPy-first API. If the compiled extension is unavailable a
pure-Python fallback is provided so that testing and light-weight usage still
work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as _ndimage
from scipy import signal as _scipy_signal

from .._core import _native, has_native_extension, require_native_extension


@dataclass
class IsotopeSignal:
    """Container for a single LA-ICP-MS isotope current time-series.

    Attributes:
        isotope: Name of the isotope, e.g. ``"18O"``, ``"16O"``, ``"d18O"``.
        time: Time axis in seconds.
        current: Raw detector current (nA or counts).
        sampling_rate_hz: Sampling frequency in Hz.
        depth_um: Optional per-sample ice-core depth in micrometers.
        attrs: Arbitrary metadata dictionary.
    """

    isotope: str
    time: np.ndarray
    current: np.ndarray
    sampling_rate_hz: float = 1.0
    depth_um: Optional[np.ndarray] = None
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64)
        self.current = np.asarray(self.current, dtype=np.float64)
        if self.time.shape != self.current.shape:
            raise ValueError(
                f"time and current must have same shape, got "
                f"{self.time.shape} vs {self.current.shape}"
            )
        if self.depth_um is not None:
            self.depth_um = np.asarray(self.depth_um, dtype=np.float64)
            if self.depth_um.shape != self.current.shape:
                raise ValueError("depth_um must match current shape")

    @property
    def n_samples(self) -> int:
        return int(self.current.size)

    @property
    def duration_s(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.n_samples > 1 else 0.0

    def __repr__(self) -> str:
        return (
            f"IsotopeSignal(isotope={self.isotope!r}, n_samples={self.n_samples}, "
            f"fs={self.sampling_rate_hz} Hz, duration={self.duration_s:.2f} s)"
        )


@dataclass
class FilterConfig:
    """Configuration for the adaptive Kalman filter pipeline.

    Attributes:
        process_noise_init: Initial process noise variance (Q).
        measurement_noise_init: Initial measurement noise variance (R).
        adaptation_rate: Gain for on-line noise parameter adaptation.
        innovation_momentum: Exponential averaging factor for noise trajectories.
        outlier_threshold_sigma: Z-score above which a sample is flagged as artifact.
        enable_frequency_domain: Pre-filter via FFT bandpass + STFT masking.
        low_cutoff_hz: Lower bandpass edge. Set 0 for low-pass only.
        high_cutoff_hz: Upper bandpass edge. Default Nyquist (fs/2).
        smooth: Run the bidirectional (forward + reverse) smoother.
        interpolation_window: Number of neighbours used to fill artifact samples.
    """

    process_noise_init: float = 1e-4
    measurement_noise_init: float = 1e-2
    adaptation_rate: float = 0.05
    innovation_momentum: float = 0.9
    outlier_threshold_sigma: float = 5.0
    enable_frequency_domain: bool = True
    low_cutoff_hz: float = 0.0
    high_cutoff_hz: float = 0.5
    smooth: bool = True
    interpolation_window: int = 5


def _fallback_bandpass(
    signal: np.ndarray, low: float, high: float, fs: float
) -> np.ndarray:
    """Pure-SciPy bandpass filter used when the native extension is unavailable."""
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 1.0 - 1e-6)
    if high_n <= low_n:
        return signal.copy()
    b, a = _scipy_signal.butter(4, [low_n, high_n], btype="band")
    return _scipy_signal.filtfilt(b, a, signal).astype(np.float64)


def _fallback_kalman_1d(
    signal: np.ndarray, config: FilterConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python 1D adaptive Kalman filter used as fallback."""
    n = signal.size
    x = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)
    innovations = np.zeros(n, dtype=np.float64)

    Q = float(config.process_noise_init)
    R = float(config.measurement_noise_init)

    x[0] = signal[0] if n > 0 else 0.0
    P[0] = R * 10.0

    running_mean = x[0]
    running_var = R

    for i in range(1, n):
        x_pred = x[i - 1]
        P_pred = P[i - 1] + Q

        innovation = signal[i] - x_pred
        z = abs(innovation) / (np.sqrt(running_var) + 1e-12)

        if z > config.outlier_threshold_sigma:
            x[i] = x_pred
            P[i] = P_pred + R * 10.0
            innovations[i] = 0.0
        else:
            K = P_pred / (P_pred + R + 1e-12)
            x[i] = x_pred + K * innovation
            P[i] = (1.0 - K) * P_pred
            innovations[i] = innovation

            target_Q = max(1e-6, config.adaptation_rate * abs(innovation ** 2 - P[i - 1] - R))
            target_R = max(1e-6, config.adaptation_rate * abs(innovation ** 2))
            Q = config.innovation_momentum * Q + (1.0 - config.innovation_momentum) * target_Q
            R = config.innovation_momentum * R + (1.0 - config.innovation_momentum) * target_R

        alpha = 0.01
        running_mean = (1.0 - alpha) * running_mean + alpha * signal[i]
        running_var = (1.0 - alpha) * running_var + alpha * (signal[i] - running_mean) ** 2

    return x, P, innovations


def _fallback_detect_outliers(
    signal: np.ndarray, filtered: np.ndarray, threshold_sigma: float
) -> np.ndarray:
    residuals = signal - filtered
    std = float(np.std(residuals))
    if std < 1e-12:
        return np.zeros_like(signal, dtype=bool)
    z = np.abs(residuals - np.mean(residuals)) / std
    return z > threshold_sigma


def _fallback_interpolate_artifacts(
    signal: np.ndarray, mask: np.ndarray, window: int
) -> np.ndarray:
    if not mask.any():
        return signal.copy()
    filled = signal.copy()
    idx = np.arange(signal.size)
    good = ~mask
    if not good.any():
        return filled
    filled[mask] = np.interp(idx[mask], idx[good], signal[good])
    if window > 1:
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = _ndimage.convolve1d(filled, kernel, mode="nearest")
        filled[mask] = smoothed[mask]
    return filled


def adaptive_kalman_filter(
    signal: np.ndarray | IsotopeSignal,
    config: Optional[FilterConfig] = None,
    *,
    use_native: bool = True,
) -> np.ndarray:
    """Apply frequency-domain adaptive Kalman filtering to a signal.

    Parameters
    ----------
    signal:
        Either a 1D NumPy array of samples or an :class:`IsotopeSignal`.
    config:
        Filter configuration. Defaults are used when ``None``.
    use_native:
        Try the compiled C++ extension first. If it is not built, fall back
        to the pure-SciPy/Python implementation.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as the input.
    """
    if isinstance(signal, IsotopeSignal):
        raw = signal.current
        fs = signal.sampling_rate_hz
    else:
        raw = np.asarray(signal, dtype=np.float64).ravel()
        fs = 1.0

    if raw.size == 0:
        return raw.copy()

    cfg = config or FilterConfig()

    if cfg.enable_frequency_domain and cfg.high_cutoff_hz < 0.5 * fs:
        high_hz = cfg.high_cutoff_hz if cfg.high_cutoff_hz > 0 else 0.5 * fs
        pre = _fallback_bandpass(raw, cfg.low_cutoff_hz, high_hz, fs)
    else:
        pre = raw

    if use_native and has_native_extension():
        native_cfg = _native.AdaptiveKalmanConfig()
        native_cfg.process_noise_init = cfg.process_noise_init
        native_cfg.measurement_noise_init = cfg.measurement_noise_init
        native_cfg.adaptation_rate = cfg.adaptation_rate
        native_cfg.innovation_momentum = cfg.innovation_momentum
        native_cfg.outlier_threshold_sigma = cfg.outlier_threshold_sigma
        native_cfg.enable_frequency_domain = False
        native_cfg.low_cutoff_hz = cfg.low_cutoff_hz
        native_cfg.high_cutoff_hz = cfg.high_cutoff_hz if cfg.high_cutoff_hz > 0 else 0.5 * fs
        native_cfg.sampling_rate_hz = fs

        if cfg.smooth:
            filtered = _native.adaptive_kalman_smooth(pre, native_cfg)
        else:
            filtered = _native.adaptive_kalman_filter(pre, native_cfg)
        return np.asarray(filtered, dtype=np.float64)

    if cfg.smooth:
        forward, _, _ = _fallback_kalman_1d(pre, cfg)
        backward, _, _ = _fallback_kalman_1d(pre[::-1], cfg)
        filtered = 0.5 * (forward + backward[::-1])
    else:
        filtered, _, _ = _fallback_kalman_1d(pre, cfg)
    return filtered


def suppress_artifacts(
    signal: np.ndarray | IsotopeSignal,
    config: Optional[FilterConfig] = None,
    *,
    use_native: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and interpolate physical artifacts (fractures, dust spikes).

    Returns
    -------
    (clean_signal, outlier_mask) : tuple[np.ndarray, np.ndarray]
        Cleaned signal and boolean mask indicating which samples were flagged.
    """
    if isinstance(signal, IsotopeSignal):
        raw = signal.current
    else:
        raw = np.asarray(signal, dtype=np.float64).ravel()

    if raw.size == 0:
        return raw.copy(), np.zeros(0, dtype=bool)

    cfg = config or FilterConfig()
    filtered = adaptive_kalman_filter(raw, cfg, use_native=use_native)

    if use_native and has_native_extension():
        mask = _native.detect_outliers(raw, filtered, cfg.outlier_threshold_sigma)
        mask = np.asarray(mask, dtype=bool)
        clean = _native.suppress_artifacts(raw, mask, cfg.interpolation_window)
        clean = np.asarray(clean, dtype=np.float64)
    else:
        mask = _fallback_detect_outliers(raw, filtered, cfg.outlier_threshold_sigma)
        clean = _fallback_interpolate_artifacts(raw, mask, cfg.interpolation_window)

    return clean, mask


def compute_isotope_ratio(
    numerator: IsotopeSignal | np.ndarray,
    denominator: IsotopeSignal | np.ndarray,
    reference_ratio: float = 1.0,
    *,
    apply_filter: bool = True,
    config: Optional[FilterConfig] = None,
) -> np.ndarray:
    """Compute a delta-notation isotope ratio such as δ¹⁸O.

    Parameters
    ----------
    numerator, denominator:
        Signals for the heavy and light isotopes, e.g. ¹⁸O and ¹⁶O currents.
    reference_ratio:
        Standard ratio used to compute the delta permil deviation. If 1.0
        the return value is simply the cleaned ratio ``num/den``.
    apply_filter:
        If ``True`` (default), run artifact suppression on both signals first.
    config:
        Filter configuration passed through to :func:`suppress_artifacts`.

    Returns
    -------
    np.ndarray
        Delta-permil values if ``reference_ratio`` is provided, otherwise the
        raw cleaned ratio.
    """
    def _get(sig: IsotopeSignal | np.ndarray) -> np.ndarray:
        if isinstance(sig, IsotopeSignal):
            return sig.current.copy()
        return np.asarray(sig, dtype=np.float64).ravel()

    num = _get(numerator)
    den = _get(denominator)

    if num.shape != den.shape:
        raise ValueError(
            f"numerator and denominator must have same shape, got {num.shape} vs {den.shape}"
        )

    if apply_filter:
        cfg = config or FilterConfig()
        num, _ = suppress_artifacts(num, cfg)
        den, _ = suppress_artifacts(den, cfg)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    if reference_ratio != 1.0:
        ratio = (ratio / reference_ratio - 1.0) * 1000.0

    return ratio
