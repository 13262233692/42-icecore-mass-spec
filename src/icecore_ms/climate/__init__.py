"""
Polar climate reconstruction data I/O and tensor slicing module.

Handles multi-dimensional NetCDF4 tensors containing multi-millennial
polar climate records (snowfall, temperature, trace elements) across
multiple Antarctic stations.
"""

from .netcdf_loader import (
    ClimateTensor,
    VariableSpec,
    compute_anomaly,
    load_climate_tensor,
    slice_depth_range,
    slice_station,
    slice_time_window,
)

__all__ = [
    "ClimateTensor",
    "VariableSpec",
    "compute_anomaly",
    "load_climate_tensor",
    "slice_depth_range",
    "slice_station",
    "slice_time_window",
]
