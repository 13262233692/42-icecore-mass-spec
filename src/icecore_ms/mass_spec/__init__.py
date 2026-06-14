"""
LA-ICP-MS mass spectrometry signal processing module.

Provides adaptive Kalman filtering for isotope pulse signals
(e.g., delta-18O) to suppress surface fracture artifacts and
dust contamination noise in the frequency domain.
"""

from .filtering import (
    IsotopeSignal,
    FilterConfig,
    adaptive_kalman_filter,
    suppress_artifacts,
    compute_isotope_ratio,
)

__all__ = [
    "IsotopeSignal",
    "FilterConfig",
    "adaptive_kalman_filter",
    "suppress_artifacts",
    "compute_isotope_ratio",
]
