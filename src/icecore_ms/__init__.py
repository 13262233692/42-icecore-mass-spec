"""
Ice Core Isotope Abundance Tomography Kernel
============================================

High-throughput processing kernel for polar ice core mass spectrometry data,
combining NetCDF4 climate tensor slicing with frequency-domain adaptive
Kalman filtering for LA-ICP-MS isotope signals.

Modules:
    climate   - Polar climate reconstruction data I/O and tensor slicing
    mass_spec - LA-ICP-MS mass spectrometry signal processing
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
