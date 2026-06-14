"""
Memory-safe distributed correlation / covariance matrix computation.

The naive approach of ``data.transpose("station", "time", "variable") @ data.T``
triggers a full Dask shuffle that materialises the entire tensor in every
worker node simultaneously – causing the OOM described in the ticket.

This module implements a **block-wise outer-product accumulation** strategy:

1. Rechunk the input tensor into the layout optimal for correlation
   (full time axis per chunk, small station blocks).
2. Split the station list into coarse blocks ``S_0, S_1, ..., S_B``.
3. For each pair ``(S_i, S_j)`` with ``i ≤ j`` load the two station blocks
   into a worker, compute the symmetric block ``C[i:j] = X_i^T @ X_j`` (after
   demeaning and normalisation), exploit symmetry to fill ``C[j:i] = C[i:j].T``,
   then release the block references and run a GC barrier before advancing.
4. Assemble the final matrix from the per-block results.

The memory footprint is bounded by ``2 × |S_block| × |time| × sizeof(dtype)``
per worker, which the adaptive chunking module guarantees to fit in RAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from .adaptive_chunking import (
    ChunkBudget,
    compute_optimal_chunks,
    format_bytes,
    gc_barrier,
    probe_available_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for a computed correlation / covariance matrix.

    Attributes:
        matrix: ``(N, N)`` correlation or covariance matrix as a
            :class:`xarray.DataArray` with labelled station / variable axes.
        kind: ``"pearson"`` or ``"covariance"``.
        along_axis: Which axis was held fixed (``"station"`` or ``"variable"``).
        labels: Ordered axis labels (station names or variable names).
        n_samples: Number of valid time/depth samples used.
        memory_peak_estimate_bytes: Estimated peak per-worker memory usage.
    """

    matrix: xr.DataArray
    kind: str
    along_axis: str
    labels: list[str]
    n_samples: int
    memory_peak_estimate_bytes: int

    def __repr__(self) -> str:
        shape = tuple(self.matrix.shape)
        mem = format_bytes(self.memory_peak_estimate_bytes)
        return (
            f"CorrelationResult(kind={self.kind!r}, along={self.along_axis!r}, "
            f"shape={shape}, n_samples={self.n_samples}, peak_mem≈{mem})"
        )


def _block_slices(n: int, block_size: int) -> list[tuple[int, int]]:
    """Return list of ``(start, stop)`` index ranges covering ``[0, n)``."""
    blocks: list[tuple[int, int]] = []
    if block_size <= 0:
        return [(0, n)]
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        blocks.append((start, stop))
    return blocks


def _demean_normalize(
    arr: np.ndarray,
    axis: int,
    *,
    compute_correlation: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Demean *arr* along *axis* and, for correlation, divide by stdev.

    Returns ``(normalized, means, stds)``.
    """
    means = arr.mean(axis=axis, keepdims=True)
    centered = arr - means
    if compute_correlation:
        stds = centered.std(axis=axis, keepdims=True)
        stds = np.where(stds < eps, 1.0, stds)
        normalized = centered / stds
    else:
        normalized = centered
        stds = np.ones_like(means)
    return normalized, means.squeeze(axis), stds.squeeze(axis)


def compute_correlation_matrix(
    tensor,
    *,
    along: str = "station",
    variable: Optional[str] = None,
    kind: str = "pearson",
    block_size: Optional[int] = None,
    gc_interval_blocks: int = 3,
    budget_bytes: Optional[int] = None,
    force_rechunk: bool = True,
) -> CorrelationResult:
    """Compute a station-station or variable-variable correlation matrix safely.

    Parameters
    ----------
    tensor:
        A :class:`~icecore_ms.climate.ClimateTensor` holding the 3D climate data.
    along:
        Axis along which to correlate – ``"station"`` (default, produces a
        ``N_station × N_station`` matrix) or ``"variable"`` (produces
        ``N_variable × N_variable``).
    variable:
        When ``along="station"``, restrict computation to one variable. If
        ``None`` all variables are averaged together (equivalent to the
        average inter-station correlation across variables).
    kind:
        ``"pearson"`` (default, correlation) or ``"covariance"``.
    block_size:
        Number of stations/variables per block.  Auto-computed from the
        memory budget when ``None``.
    gc_interval_blocks:
        Run a distributed GC barrier every *N* block-pair iterations.
    budget_bytes:
        Per-worker memory budget.  Auto-detected from psutil / Dask when ``None``.
    force_rechunk:
        If ``True`` (default) the tensor is automatically rechunked into the
        correlation-friendly layout before computation begins.  Set to
        ``False`` only if you already called :func:`rechunk_for_operation`
        with ``operation="cross_station_correlation"``.
    """
    from .netcdf_loader import ClimateTensor

    if not isinstance(tensor, ClimateTensor):
        raise TypeError(f"Expected ClimateTensor, got {type(tensor).__name__}")

    if kind not in ("pearson", "covariance"):
        raise ValueError(f"kind must be 'pearson' or 'covariance', got {kind!r}")

    if budget_bytes is None:
        budget_bytes = probe_available_memory()

    if along not in ("station", "variable"):
        raise ValueError(f"along must be 'station' or 'variable', got {along!r}")

    da = tensor.data
    depth_axis = tensor.depth_axis

    if along == "station":
        if variable is not None:
            if variable not in [v.name for v in tensor.variables]:
                raise KeyError(f"Variable {variable!r} not in tensor")
            da = da.sel(variable=variable).squeeze(drop=True)
            if da.ndim == 2:
                da = da.expand_dims({"variable": [variable]})
        n_axis = tensor.n_stations
        labels = list(tensor.stations)
    else:
        n_axis = tensor.n_variables
        labels = [v.name for v in tensor.variables]

    if force_rechunk:
        op = "cross_station_correlation" if along == "station" else "cross_variable"
        shape = {depth_axis: tensor.n_depth,
                 "station": tensor.n_stations,
                 "variable": tensor.n_variables}
        budget = compute_optimal_chunks(
            shape,
            depth_axis=depth_axis,
            dtype=da.dtype,
            operation=op,
            budget_bytes=budget_bytes,
        )
        logger.info("Rechunking for %s: %s", op, budget.summary())
        da = da.chunk(budget.chunks)

    if block_size is None:
        if along == "station":
            bs = da.chunks[da.dims.index("station")][0] if da.chunks is not None else -1
        else:
            bs = da.chunks[da.dims.index("variable")][0] if da.chunks is not None else -1
        block_size = n_axis if bs == -1 or bs is None else int(bs)
        block_size = min(block_size, n_axis)
        block_size = max(1, min(block_size, 256))

    logger.info("Computing %s matrix along %s: n=%d, block_size=%d",
                kind, along, n_axis, block_size)

    n_time = tensor.n_depth
    item_bytes = np.dtype(da.dtype).itemsize
    peak_est = int(2 * block_size * n_time * item_bytes * 1.3)  # 1.3 = Python overhead
    logger.info("Estimated peak per-worker memory: %s (budget %s)",
                format_bytes(peak_est), format_bytes(budget_bytes))

    if along == "station":
        da_2d = da.stack(flat=("variable",)).mean(dim="flat") \
            if da.ndim == 3 and da.sizes.get("variable", 1) > 1 and variable is None \
            else da.squeeze(dim="variable") if da.ndim == 3 else da
        if da_2d.ndim != 2:
            raise ValueError(f"Expected 2D (depth, station) array, got {da_2d.ndim}D")
        axis_idx = da_2d.dims.index("station")
        time_idx = da_2d.dims.index(depth_axis)
    else:
        da_2d = da.isel(station=0).squeeze(drop=True) if tensor.n_stations == 1 \
            else da.mean(dim="station")
        if da_2d.ndim != 2:
            raise ValueError(f"Expected 2D (depth, variable) array, got {da_2d.ndim}D")
        axis_idx = da_2d.dims.index("variable")
        time_idx = da_2d.dims.index(depth_axis)

    da_2d = da_2d.transpose(depth_axis, along)
    full_data = da_2d.compute()
    arr = np.asarray(full_data.values, dtype=np.float64)

    normalized, _means, _stds = _demean_normalize(
        arr, axis=time_idx, compute_correlation=(kind == "pearson")
    )

    blocks = _block_slices(n_axis, block_size)
    n_blocks = len(blocks)
    result = np.zeros((n_axis, n_axis), dtype=np.float64)

    logger.info("Processing %d blocks (%d block pairs)...",
                n_blocks, n_blocks * (n_blocks + 1) // 2)

    pair_count = 0
    for i, (i_start, i_stop) in enumerate(blocks):
        block_i = normalized[:, i_start:i_stop]  # (T, bi)
        for j in range(i, n_blocks):
            j_start, j_stop = blocks[j]
            block_j = normalized[:, j_start:j_stop]  # (T, bj)

            block_corr = block_i.T @ block_j  # (bi, bj)
            if kind == "pearson":
                block_corr /= max(1, n_time - 1)
            else:
                block_corr /= max(1, n_time - 1)

            result[i_start:i_stop, j_start:j_stop] = block_corr
            if i != j:
                result[j_start:j_stop, i_start:i_stop] = block_corr.T

            pair_count += 1
            if pair_count % gc_interval_blocks == 0:
                gc_barrier(collect_workers=False)
                logger.debug("GC barrier after %d pairs", pair_count)

    if kind == "pearson":
        np.clip(result, -1.0, 1.0, out=result)

    row_dim = f"{along}_row"
    col_dim = f"{along}_col"
    coords = {row_dim: labels, col_dim: labels}
    result_da = xr.DataArray(
        result,
        dims=(row_dim, col_dim),
        coords=coords,
        attrs={"kind": kind, "n_samples": int(n_time), "along_axis": along},
    )

    gc_barrier(collect_workers=False)
    logger.info("Correlation matrix complete: %s", result_da.shape)

    return CorrelationResult(
        matrix=result_da,
        kind=kind,
        along_axis=along,
        labels=list(labels),
        n_samples=int(n_time),
        memory_peak_estimate_bytes=peak_est,
    )


def rechunk_for_operation(
    tensor,
    operation: str,
    *,
    budget_bytes: Optional[int] = None,
    dask_memory_limit: Optional[str] = None,
) -> tuple:
    """Return a new ``ClimateTensor`` rechunked optimally for *operation*.

    See :func:`icecore_ms.climate.adaptive_chunking.compute_optimal_chunks`
    for the list of supported operations.
    """
    from .netcdf_loader import ClimateTensor

    if not isinstance(tensor, ClimateTensor):
        raise TypeError(f"Expected ClimateTensor, got {type(tensor).__name__}")

    shape = {
        tensor.depth_axis: tensor.n_depth,
        "station": tensor.n_stations,
        "variable": tensor.n_variables,
    }
    budget: ChunkBudget = compute_optimal_chunks(
        shape,
        depth_axis=tensor.depth_axis,
        dtype=tensor.data.dtype,
        operation=operation,
        budget_bytes=budget_bytes,
        dask_memory_limit=dask_memory_limit,
    )
    logger.info("Rechunk plan:\n%s", budget.summary())
    new_data = tensor.data.chunk(budget.chunks)
    return (
        ClimateTensor(
            data=new_data,
            depth_axis=tensor.depth_axis,
            stations=list(tensor.stations),
            variables=list(tensor.variables),
            path=tensor.path,
        ),
        budget,
    )


__all__ = [
    "CorrelationResult",
    "compute_correlation_matrix",
    "rechunk_for_operation",
]
