"""
Polar climate reconstruction data I/O, memory-adaptive Dask tensor slicing,
and distributed correlation matrix computation.

Modules:
    netcdf_loader     – Core 3D tensor I/O (refactored with adaptive chunking)
    adaptive_chunking – Memory-aware chunk planner + distributed GC barriers
    correlation       – Block-wise OOM-safe cross-station / cross-variable correlation
"""

from .adaptive_chunking import (
    ChunkBudget,
    compute_optimal_chunks,
    format_bytes,
    gc_barrier,
    parse_memory_string,
    probe_available_memory,
    validate_chunk_memory,
)
from .correlation import (
    CorrelationResult,
    compute_correlation_matrix,
    rechunk_for_operation,
)
from .netcdf_loader import (
    ClimateTensor,
    VariableSpec,
    compute_anomaly,
    get_dask_client,
    load_climate_tensor,
    slice_depth_range,
    slice_station,
    slice_time_window,
)

__all__ = [
    "ChunkBudget",
    "ClimateTensor",
    "CorrelationResult",
    "VariableSpec",
    "compute_anomaly",
    "compute_correlation_matrix",
    "compute_optimal_chunks",
    "format_bytes",
    "gc_barrier",
    "get_dask_client",
    "load_climate_tensor",
    "parse_memory_string",
    "probe_available_memory",
    "rechunk_for_operation",
    "slice_depth_range",
    "slice_station",
    "slice_time_window",
    "validate_chunk_memory",
]
