"""
Paleoclimate cycle extrapolation and uncertainty quantification module.

Implements a non-linear least-squares spectral analysis (LSSA) / harmonic
inference engine that decomposes isotopic time series into periodic components
rooted in Milankovitch orbital forcing theory.  A Markov-Chain Monte Carlo
(MCMC) sampler returns full posterior distributions so that the output is not
just a single best-fit curve but a **P10 / P50 (median) / P90 confidence
envelope** that reveals abrupt transition risks (glacial / interglacial
tipping points).

Theory
------
The model assumes that the observed isotope signal ``δ(t)`` can be
decomposed as a sum of sinusoidal components with known *a priori* periods
(eccentricity ~100 kyr, obliquity ~41 kyr, precession ~23 kyr) plus an
AR(1)-like noise term and a long-term drift:

    δ(t) = μ + α·t + Σ_i A_i·sin(2π·t / P_i + φ_i) + ε_t

where:
  - ``μ``   – baseline offset
  - ``α``   – long-term linear drift (e.g. cooling / warming trend)
  - ``A_i`` – amplitude of the i-th orbital harmonic
  - ``P_i`` – period of the i-th harmonic (with Gaussian priors around
              the canonical Milankovitch values)
  - ``φ_i`` – phase of the i-th harmonic
  - ``ε_t`` – residual noise (modelled as Gaussian for simplicity)

The posterior ``p(θ | data)`` is sampled via a Metropolis-Hastings MCMC
with adaptive proposal covariance (Robust Adaptive Metropolis algorithm).

API summary
-----------
- :class:`MilankovitchPrior`     – orbital-cycle prior configuration
- :func:`lssa_periodogram`       – Least-Squares Spectral Analysis periodogram
- :func:`fit_harmonic_lsq`       – non-linear least-squares best fit
- :class:`MCMCResult`            – MCMC posterior container
- :func:`sample_harmonic_mcmc`   – run the MCMC sampler
- :func:`build_confidence_band`  – compute P10/P50/P90 envelope
- :func:`detect_tipping_points`  – find abrupt transition candidates
- :func:`extrapolate_climate_cycles` – high-level one-shot entry point
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)


# ===========================================================================
# Milankovitch priors
# ===========================================================================

MILANKOVITCH_CANONICAL = {
    "eccentricity_100kyr":   100_000.0,   # Earth orbital eccentricity
    "obliquity_41kyr":        41_000.0,   # Axial tilt (obliquity)
    "precession_23kyr":       23_000.0,   # Axial precession (wobble)
}

MILANKOVITCH_PRIOR_SD_FRAC = 0.05  # ±5% a-priori uncertainty on periods


@dataclass
class MilankovitchPrior:
    """Configuration of prior knowledge for harmonic decomposition.

    Attributes
    ----------
    periods:
        Mapping ``{name: period_years}`` of expected periodic components.
        Defaults to the three canonical Milankovitch cycles.
    period_sd_frac:
        Fractional Gaussian standard deviation of the period prior.
    amplitude_prior_mean, amplitude_prior_sd:
        Half-normal prior parameters for each harmonic amplitude.
    drift_prior_sd:
        Gaussian prior std-dev on the linear drift coefficient (units of
        signal per year).
    noise_prior_shape, noise_prior_scale:
        Inverse-Gamma prior parameters for the residual noise variance.
    include_drift:
        Whether to include a linear trend term α·t.
    """

    periods: dict[str, float] = field(
        default_factory=lambda: dict(MILANKOVITCH_CANONICAL)
    )
    period_sd_frac: float = MILANKOVITCH_PRIOR_SD_FRAC
    amplitude_prior_mean: float = 1.0
    amplitude_prior_sd: float = 5.0
    drift_prior_sd: float = 1e-4
    noise_prior_shape: float = 3.0
    noise_prior_scale: float = 1.0
    include_drift: bool = True

    @property
    def n_harmonics(self) -> int:
        return len(self.periods)

    def period_names(self) -> list[str]:
        return list(self.periods.keys())


# ===========================================================================
# Harmonic model — forward function
# ===========================================================================

def _harmonic_forward(
    t: np.ndarray,
    mu: float,
    alpha: float,
    amplitudes: np.ndarray,
    periods: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Evaluate the harmonic model at times *t*.

    ``model(t) = μ + α·t + Σ_i A_i·sin(2π t / P_i + φ_i)``
    """
    y = np.full_like(t, mu + alpha * t, dtype=np.float64)
    for A, P, phi in zip(amplitudes, periods, phases):
        y += A * np.sin(2.0 * math.pi * t / P + phi)
    return y


def _param_dict_to_arrays(
    prior: MilankovitchPrior, theta: dict[str, float]
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Unpack a flat parameter dictionary into typed arrays.

    Returns ``(mu, alpha, amplitudes, periods, phases, log_sigma)``.
    """
    mu = theta["mu"]
    alpha = theta.get("alpha", 0.0)
    log_sigma = theta["log_sigma"]
    names = prior.period_names()
    amplitudes = np.array([theta[f"A_{n}"] for n in names])
    periods = np.array([theta[f"P_{n}"] for n in names])
    phases = np.array([theta[f"phi_{n}"] for n in names])
    return mu, alpha, amplitudes, periods, phases, log_sigma


def _pack_param_dict(
    prior: MilankovitchPrior,
    mu: float,
    alpha: float,
    amplitudes: np.ndarray,
    periods: np.ndarray,
    phases: np.ndarray,
    log_sigma: float,
) -> dict[str, float]:
    """Inverse of :func:`_param_dict_to_arrays`."""
    d: dict[str, float] = {"mu": mu, "alpha": alpha, "log_sigma": log_sigma}
    for i, name in enumerate(prior.period_names()):
        d[f"A_{name}"] = float(amplitudes[i])
        d[f"P_{name}"] = float(periods[i])
        d[f"phi_{name}"] = float(phases[i])
    return d


def _param_names(prior: MilankovitchPrior) -> list[str]:
    names = ["mu", "alpha"] if prior.include_drift else ["mu"]
    for n in prior.period_names():
        names.extend([f"A_{n}", f"P_{n}", f"phi_{n}"])
    names.append("log_sigma")
    return names


def _n_params(prior: MilankovitchPrior) -> int:
    return 1 + (1 if prior.include_drift else 0) + prior.n_harmonics * 3 + 1


# ===========================================================================
# Prior / likelihood / posterior
# ===========================================================================

def _log_prior(
    theta: dict[str, float],
    prior: MilankovitchPrior,
) -> float:
    """Log of the prior probability density."""
    lp = 0.0

    lp += stats.norm.logpdf(theta["mu"], 0.0, 10.0)

    if prior.include_drift:
        lp += stats.norm.logpdf(theta["alpha"], 0.0, prior.drift_prior_sd)

    for name, P0 in prior.periods.items():
        A = theta[f"A_{name}"]
        P = theta[f"P_{name}"]

        lp += stats.halfnorm.logpdf(
            abs(A), loc=0.0, scale=prior.amplitude_prior_sd
        )

        lp += stats.norm.logpdf(P, loc=P0, scale=P0 * prior.period_sd_frac)

    sigma = np.exp(theta["log_sigma"])
    lp += stats.invgamma.logpdf(
        sigma ** 2, a=prior.noise_prior_shape, scale=prior.noise_prior_scale
    )

    return float(lp)


def _log_likelihood(
    theta: dict[str, float],
    t: np.ndarray,
    y: np.ndarray,
    prior: MilankovitchPrior,
) -> float:
    mu, alpha, amps, pers, phis, log_sigma = _param_dict_to_arrays(prior, theta)
    sigma = np.exp(log_sigma)
    y_hat = _harmonic_forward(t, mu, alpha, amps, pers, phis)
    resid = y - y_hat
    n = len(t)
    ll = -0.5 * n * np.log(2.0 * math.pi) - n * log_sigma - 0.5 * np.sum(resid ** 2) / (sigma ** 2)
    return float(ll)


def _log_posterior(
    theta: dict[str, float],
    t: np.ndarray,
    y: np.ndarray,
    prior: MilankovitchPrior,
) -> float:
    lp = _log_prior(theta, prior)
    if not np.isfinite(lp):
        return -np.inf
    ll = _log_likelihood(theta, t, y, prior)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ===========================================================================
# Least-Squares Spectral Analysis (LSSA)
# ===========================================================================

def lssa_periodogram(
    t: np.ndarray,
    y: np.ndarray,
    *,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
    n_periods: int = 200,
    include_drift: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Lomb-Scargle-style least-squares periodogram.

    For each trial period *P*, fit ``A·sin(2π t/P + φ) + μ + α·t`` by
    linear least squares (using the sin + cos decomposition so the model
    is linear in ``[A cos φ, A sin φ]``).  Return the fitted amplitude at
    each trial period.

    Parameters
    ----------
    t, y:
        Time and value arrays.  *t* should be in years for Milankovitch
        interpretation.
    period_min, period_max:
        Period range to scan (same units as *t*).  Defaults to 10 % and
        10× the baseline duration.
    n_periods:
        Number of trial periods (log-spaced).
    include_drift:
        Include a linear trend term in the sinusoid + baseline model.

    Returns
    -------
    (periods, amplitudes) : tuple[np.ndarray, np.ndarray]
        Trial periods and the corresponding best-fit sinusoid amplitude.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if t.size != y.size:
        raise ValueError("t and y must have the same length")
    if t.size < 5:
        raise ValueError("Need at least 5 samples for LSSA periodogram")

    duration = float(t[-1] - t[0])
    if period_min is None:
        period_min = max(duration * 0.1, np.median(np.diff(t)) * 2)
    if period_max is None:
        period_max = duration * 10.0

    periods = np.logspace(
        np.log10(period_min), np.log10(period_max), n_periods
    )
    amplitudes = np.zeros(n_periods, dtype=np.float64)

    t_norm = t - t.mean()
    y_demean = y - y.mean()

    for i, P in enumerate(periods):
        omega = 2.0 * math.pi / P
        sin_t = np.sin(omega * t_norm)
        cos_t = np.cos(omega * t_norm)

        if include_drift:
            X = np.column_stack([np.ones_like(t_norm), t_norm, sin_t, cos_t])
        else:
            X = np.column_stack([np.ones_like(t_norm), sin_t, cos_t])

        coef, *_ = np.linalg.lstsq(X, y_demean, rcond=None)
        if include_drift:
            sin_coef = coef[2]
            cos_coef = coef[3]
        else:
            sin_coef = coef[1]
            cos_coef = coef[2]
        amplitudes[i] = math.sqrt(sin_coef ** 2 + cos_coef ** 2)

    return periods, amplitudes


# ===========================================================================
# Non-linear least-squares best fit (initial point for MCMC)
# ===========================================================================

def fit_harmonic_lsq(
    t: np.ndarray,
    y: np.ndarray,
    *,
    prior: Optional[MilankovitchPrior] = None,
    method: str = "trf",
) -> dict[str, float]:
    """Fit the multi-harmonic model via non-linear least squares.

    Uses ``scipy.optimize.curve_fit`` with the period values initialised
    from the prior means and amplitudes initialised from an LSSA scan.
    The result is a good starting point for the MCMC chain.

    Parameters
    ----------
    t, y:
        Time and value arrays.
    prior:
        Prior configuration (also defines which harmonics to include).
    method:
        Optimization method passed to ``scipy.optimize.least_squares``.

    Returns
    -------
    dict[str, float]
        Named parameter dictionary suitable for use as an MCMC start point.
    """
    if prior is None:
        prior = MilankovitchPrior()

    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    mu0 = float(np.mean(y))
    alpha0 = float(np.polyfit(t - t.mean(), y, 1)[0]) if prior.include_drift else 0.0

    per_list = list(prior.periods.values())
    name_list = prior.period_names()

    amp_init = np.ones(prior.n_harmonics, dtype=np.float64)
    period_init = np.array(per_list, dtype=np.float64)
    phase_init = np.zeros(prior.n_harmonics, dtype=np.float64)

    try:
        pers_lssa, amps_lssa = lssa_periodogram(
            t, y,
            period_min=min(per_list) * 0.5,
            period_max=max(per_list) * 2.0,
            n_periods=min(500, max(100, len(per_list) * 50)),
            include_drift=prior.include_drift,
        )
        for i, p0 in enumerate(per_list):
            idx = int(np.argmin(np.abs(pers_lssa - p0)))
            amp_init[i] = max(amps_lssa[idx], 1e-6)
    except Exception as exc:
        logger.warning("LSSA initialisation failed, using default amps: %s", exc)
        amp_init[:] = max(np.std(y) / prior.n_harmonics, 1e-3)

    n_p = prior.n_harmonics

    def _pack_x(mu, alpha, amps, pers, phases) -> np.ndarray:
        return np.concatenate([[mu, alpha], amps, pers, phases])

    def _unpack_x(x: np.ndarray):
        mu = x[0]
        alpha = x[1] if prior.include_drift else 0.0
        off = 2 if prior.include_drift else 1
        amps = x[off:off + n_p]
        pers = x[off + n_p:off + 2 * n_p]
        phases = x[off + 2 * n_p:off + 3 * n_p]
        return mu, alpha, amps, pers, phases

    def residuals(x: np.ndarray) -> np.ndarray:
        mu, alpha, amps, pers, phases = _unpack_x(x)
        yhat = _harmonic_forward(t, mu, alpha, amps, pers, phases)
        return y - yhat

    x0 = _pack_x(mu0, alpha0, amp_init, period_init, phase_init)

    lower = _pack_x(
        -np.inf, -np.inf,
        np.full(n_p, 1e-9),
        np.array(per_list) * (1.0 - 0.5),
        np.full(n_p, -np.pi * 4),
    )
    upper = _pack_x(
        np.inf, np.inf,
        np.full(n_p, np.inf),
        np.array(per_list) * (1.0 + 0.5),
        np.full(n_p, np.pi * 4),
    )

    if not prior.include_drift:
        x0[1] = 0.0
        lower[1] = -1e-9
        upper[1] = 1e-9

    result = optimize.least_squares(
        residuals, x0,
        bounds=(lower, upper),
        method=method,
        max_nfev=5000,
    )

    mu_fit, alpha_fit, amps_fit, pers_fit, phases_fit = _unpack_x(result.x)
    sigma_fit = float(np.std(residuals(result.x)))
    sigma_fit = max(sigma_fit, 1e-9)

    d = _pack_param_dict(
        prior, mu_fit, alpha_fit,
        np.abs(amps_fit), pers_fit, phases_fit,
        math.log(sigma_fit),
    )
    return d
