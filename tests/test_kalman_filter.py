"""Tests for the LA-ICP-MS mass spectrometry signal processing module."""

from __future__ import annotations

import numpy as np
import pytest

from icecore_ms.mass_spec import (
    FilterConfig,
    IsotopeSignal,
    adaptive_kalman_filter,
    compute_isotope_ratio,
    suppress_artifacts,
)
from icecore_ms._core import has_native_extension


def _make_isotope_pulse(
    n_samples: int = 2000,
    n_pulses: int = 5,
    snr: float = 20.0,
    artifact_prob: float = 0.01,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)
    signal = np.zeros(n_samples, dtype=np.float64)

    pulse_centers = np.linspace(n_samples * 0.1, n_samples * 0.9, n_pulses, dtype=int)
    for c in pulse_centers:
        width = max(5, int(n_samples * 0.02))
        pulse = np.exp(-0.5 * ((t - c) / width) ** 2) * rng.uniform(0.5, 2.0)
        signal += pulse

    noise_std = signal.max() / (snr + 1e-9)
    signal += rng.normal(0, noise_std, n_samples)

    n_artifacts = int(n_samples * artifact_prob)
    artifact_idx = rng.choice(n_samples, size=n_artifacts, replace=False)
    signal[artifact_idx] += rng.uniform(10, 50, n_artifacts) * noise_std

    return t, signal, artifact_idx


def test_isotope_signal_construction():
    t = np.linspace(0, 10, 1000)
    y = np.sin(t)
    sig = IsotopeSignal(isotope="18O", time=t, current=y, sampling_rate_hz=100.0)
    assert sig.isotope == "18O"
    assert sig.n_samples == 1000
    assert sig.sampling_rate_hz == 100.0
    assert abs(sig.duration_s - 10.0) < 1e-6


def test_isotope_signal_shape_mismatch_raises():
    with pytest.raises(ValueError):
        IsotopeSignal(isotope="X", time=np.arange(10), current=np.arange(5))


def test_adaptive_kalman_filter_reduces_noise():
    t, raw, _ = _make_isotope_pulse(n_samples=500, n_pulses=3, snr=3.0, artifact_prob=0.0)
    filtered = adaptive_kalman_filter(raw, use_native=False)

    raw_std = np.std(raw)
    filt_std = np.std(filtered)
    assert filt_std < raw_std * 0.95, f"Filtered std {filt_std:.4f} should be < {raw_std * 0.95:.4f}"

    corr = np.corrcoef(raw, filtered)[0, 1]
    assert corr > 0.5, "Filtered signal should correlate with raw"


@pytest.mark.skipif(not has_native_extension(), reason="Native C++ extension not built")
def test_adaptive_kalman_filter_native_matches_python_qualitatively():
    t, raw, _ = _make_isotope_pulse(n_samples=500, snr=10.0, artifact_prob=0.0)
    cfg = FilterConfig(enable_frequency_domain=False, smooth=False)
    py_result = adaptive_kalman_filter(raw, cfg, use_native=False)
    native_result = adaptive_kalman_filter(raw, cfg, use_native=True)

    assert native_result.shape == py_result.shape
    corr = np.corrcoef(py_result, native_result)[0, 1]
    assert corr > 0.8, f"Native and Python should agree highly (got r={corr:.3f})"


def test_suppress_artifacts_detects_spikes():
    t, raw, true_idx = _make_isotope_pulse(
        n_samples=1000, n_pulses=3, snr=30.0, artifact_prob=0.02, seed=123
    )
    clean, mask = suppress_artifacts(raw, use_native=False)

    n_detected = int(mask.sum())
    n_true = len(true_idx)
    assert n_detected >= 0.5 * n_true, (
        f"Should detect at least half the injected artifacts: "
        f"detected={n_detected}, true={n_true}"
    )

    clean_std = np.std(clean[mask]) if mask.any() else 0.0
    raw_std = np.std(raw[mask]) if mask.any() else np.std(raw)
    assert clean_std < raw_std, "Cleaned signal should be smoother at artifact locations"


def test_filter_config_defaults():
    cfg = FilterConfig()
    assert cfg.process_noise_init > 0
    assert cfg.measurement_noise_init > 0
    assert 0 < cfg.adaptation_rate < 1
    assert cfg.outlier_threshold_sigma > 0


def test_compute_isotope_ratio_basic():
    rng = np.random.default_rng(0)
    n = 500
    base = rng.uniform(100, 200, n)
    ratio_true = 1.0 / 500.0
    heavy = base * ratio_true * (1 + rng.normal(0, 0.01, n))
    light = base

    result = compute_isotope_ratio(heavy, light, reference_ratio=ratio_true, apply_filter=False)
    assert result.shape == (n,)
    assert not np.any(np.isnan(result))

    delta_abs_mean = np.abs(np.nanmean(result))
    assert delta_abs_mean < 50.0, "delta permil should be near zero for matched reference"


def test_compute_isotope_ratio_with_filter():
    rng = np.random.default_rng(0)
    n = 500
    base = np.ones(n) * 100.0
    ratio_true = 0.002
    heavy = base * ratio_true
    light = base

    spike_idx = rng.choice(n, size=10, replace=False)
    heavy[spike_idx] *= 10

    result_no_filter = compute_isotope_ratio(heavy, light, reference_ratio=ratio_true, apply_filter=False)
    result_filtered = compute_isotope_ratio(heavy, light, reference_ratio=ratio_true, apply_filter=True)

    assert np.std(result_filtered) < np.std(result_no_filter)


def test_filter_accepts_isotope_signal_object():
    t, raw, _ = _make_isotope_pulse(n_samples=200)
    sig = IsotopeSignal(isotope="18O", time=t, current=raw, sampling_rate_hz=10.0)
    result = adaptive_kalman_filter(sig, use_native=False)
    assert result.shape == raw.shape


def test_empty_signal_noop():
    empty = np.array([], dtype=np.float64)
    result = adaptive_kalman_filter(empty, use_native=False)
    assert result.size == 0

    clean, mask = suppress_artifacts(empty, use_native=False)
    assert clean.size == 0
    assert mask.size == 0
