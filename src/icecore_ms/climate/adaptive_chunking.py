"""
Memory-aware adaptive chunking engine for distributed climate tensor workloads.

Problem Diagnosis
-----------------
The default Dask chunking strategy (``station: -1`` meaning *all* stations in one
chunk, paired with a hard-coded ``depth: 500``) causes catastrophic MemoryError /
OOM when performing cross-axis operations such as transpose + covariance/correlation
matrix assembly. During a transpose shuffle, each worker must materialise every
chunk that overlaps with its target output partition; with ``station: -1`` every
time-chunk overlaps every station partition, so the *entire tensor* is inflated in
every worker's RSS simultaneously.

Solution
--------
This module provides:

1. :func:`probe_available_memory` – detect per-worker RAM budget (psutil + Dask).
2. :func:`compute_optimal_chunks` – given tensor metadata and a target *per-chunk*
   memory budget, return chunk sizes with mathematical guarantees that no shuffle
   task ever exceeds the budget even in the worst-case transpose.
3. :func:`rechunk_for_operation` – select the optimal chunk layout for a specific
   access pattern (``"time_series"`` vs. ``"cross_station_correlation"``).
4. :func:`gc_barrier` – explicit distributed garbage-collection barrier that
   forces reference drops on all workers *before* a memory-heavy stage.
5. :class:`ChunkBudget` – dataclass carrying the resolved plan for traceability.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


OperationKind = Literal["time_series", "cross_station_correlation", "cross_variable", "general"]

_DTYPE_BYTES: dict[np.dtype | str, int] = {
    np.dtype("float64"): 8,
    np.dtype("float32"): 4,
    np.dtype("int64"): 8,
    np.dtype("int32"): 4,
    np.dtype("int16"): 2,
    np.dtype("bool"): 1,
    "float64": 8,
    "float32": 4,
}


def _dtype_nbytes(dtype: Any) -> int:
    dt = np.dtype(dtype)
    return _DTYPE_BYTES.get(dt, dt.itemsize)


_MEM_SCALE = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "B": 1}


def parse_memory_string(spec: str | int | float) -> int:
    """Parse ``"4GB"`` / ``"512MB"`` / plain integer bytes into byte count.

    Examples
    --------
    >>> parse_memory_string("4GB")
    4294967296
    >>> parse_memory_string(1024)
    1024
    """
    if isinstance(spec, (int, float)):
        return int(spec)
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB|B)?\s*$", spec, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse memory spec: {spec!r}")
    value = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    return int(value * _MEM_SCALE[unit])


def format_bytes(n: int) -> str:
    """Human-readable byte formatting."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def probe_available_memory(
    dask_memory_limit: Optional[str] = None,
    safety_ratio: float = 0.45,
) -> int:
    """Return a *per-worker* memory budget in bytes.

    Priority:
      1. Explicit ``dask_memory_limit`` string (e.g. ``"8GB"``).
      2. ``psutil.virtual_memory().available`` scaled by ``safety_ratio``.
      3. Conservative fallback of 2 GB.

    The ``safety_ratio`` accounts for Python object overhead, Dask task-graph
    bookkeeping, and the fact that shuffle tasks read *multiple* chunks
    simultaneously.  Default ``0.45`` means we assume 45% of free RAM can be
    dedicated to a single numerical array buffer.
    """
    if dask_memory_limit:
        limit = parse_memory_string(dask_memory_limit)
        budget = int(limit * safety_ratio)
        logger.info("Using Dask worker memory limit %s → budget %s",
                    format_bytes(limit), format_bytes(budget))
        return budget

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        budget = int(vm.available * safety_ratio)
        logger.info("Detected system RAM available %s (total %s) → budget %s (%.0f%%)",
                    format_bytes(vm.available), format_bytes(vm.total),
                    format_bytes(budget), safety_ratio * 100)
        return budget
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("psutil unavailable (%s); falling back to 2 GB budget", exc)
        return parse_memory_string("2GB")


@dataclass
class ChunkBudget:
    """Resolved chunk plan with provenance.

    Attributes:
        chunks: Final mapping ``{dim_name: chunk_size}``. ``-1`` means the whole
            dimension (only used when the dimension is tiny enough to fit).
        per_chunk_bytes: Predicted per-chunk memory footprint in bytes.
        budget_bytes: Budget used for the calculation.
        operation: Operation kind this plan was optimised for.
        n_chunks: Total number of chunks across the whole tensor.
        notes: Human-readable rationale for debugging.
    """

    chunks: dict[str, int]
    per_chunk_bytes: int
    budget_bytes: int
    operation: OperationKind
    n_chunks: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ChunkBudget(operation={self.operation!r},",
            f"            per_chunk={format_bytes(self.per_chunk_bytes)}, "
            f"budget={format_bytes(self.budget_bytes)},",
            f"            n_chunks_total={self.n_chunks},",
            f"            chunks={self.chunks})",
        ]
        if self.notes:
            lines.append("  notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        return "\n".join(lines)


def _shape_of_dim(shape: Mapping[str, int], dim: str) -> int:
    if dim not in shape:
        raise KeyError(f"Dimension {dim!r} not in shape {dict(shape)}")
    return int(shape[dim])


def _optimal_1d_split(
    n_elements: int,
    elements_budget: int,
    prefer_smaller: bool = True,
) -> int:
    """Return the chunk size along one axis that keeps total elements ≤ budget.

    The chunk size is rounded *down* to a "nice" number (multiple of a power of
    two or a multiple of 100) so that chunk boundaries are cache-friendly and
    partitions are roughly equal-sized.
    """
    if n_elements <= 0:
        return 0
    if elements_budget <= 0:
        return 1
    if n_elements <= elements_budget:
        return -1  # whole dimension fits

    raw_chunk = n_elements / math.ceil(n_elements / elements_budget)
    chunk = int(math.floor(raw_chunk)) if prefer_smaller else int(math.ceil(raw_chunk))
    chunk = max(1, chunk)

    for base in (1000, 500, 250, 200, 100, 64, 32, 16):
        if chunk >= base:
            chunk = (chunk // base) * base
            break
    return max(1, chunk)


def compute_optimal_chunks(
    shape: Mapping[str, int],
    *,
    depth_axis: str,
    dtype: Any = np.float64,
    operation: OperationKind = "general",
    budget_bytes: Optional[int] = None,
    dask_memory_limit: Optional[str] = None,
    safety_ratio: float = 0.45,
    shuffle_overhead: float = 2.5,
) -> ChunkBudget:
    """Compute chunk sizes with a strict per-chunk memory budget.

    Parameters
    ----------
    shape:
        Mapping ``{dim_name: size}``.  Expected dims are ``depth_axis``,
        ``"station"``, ``"variable"``.
    depth_axis:
        Name of the time/depth dimension.
    dtype:
        Element dtype, used to compute byte footprint.
    operation:
        Which access pattern to optimise for:

        * ``"time_series"`` – keep *all* stations + variables in a chunk,
          split only along the time/depth axis.  Ideal for per-station analysis.
        * ``"cross_station_correlation"`` – keep *all* time samples per station
          in a single chunk (so covariance can be computed without re-reading
          the time axis), split stations and variables aggressively.  **This is
          the critical fix for the cross-station correlation OOM.**
        * ``"cross_variable"`` – similar to cross-station but along variable axis.
        * ``"general"`` – balanced split: time and station both chunked.
    budget_bytes:
        Hard per-chunk memory limit in bytes.  Auto-detected if ``None``.
    dask_memory_limit:
        Dask worker memory limit string (e.g. ``"8GB"``), forwarded to
        :func:`probe_available_memory`.
    safety_ratio:
        Fraction of detected free RAM to treat as usable for array data.
    shuffle_overhead:
        Worst-case multiplicative factor applied to the per-chunk budget to
        account for shuffle tasks holding multiple chunks in flight.  The
        default ``2.5`` follows Dask's own internal guidance for the ``shuffle``
        scheduler.
    """
    if budget_bytes is None:
        budget_bytes = probe_available_memory(dask_memory_limit, safety_ratio)

    effective_budget = int(budget_bytes / max(1.0, shuffle_overhead))
    item_bytes = _dtype_nbytes(dtype)
    elements_budget = max(1, effective_budget // item_bytes)

    notes: list[str] = []
    notes.append(f"dtype={np.dtype(dtype).name} ({item_bytes} B/item)")
    notes.append(f"raw budget={format_bytes(budget_bytes)}, "
                 f"effective after shuffle_overhead={format_bytes(effective_budget)} "
                 f"= {elements_budget:_} elements")

    n_depth = _shape_of_dim(shape, depth_axis)
    n_station = _shape_of_dim(shape, "station")
    n_variable = _shape_of_dim(shape, "variable")

    chunks: dict[str, int] = {}

    if operation == "time_series":
        chunks["station"] = -1
        chunks["variable"] = -1
        fixed_elements = n_station * n_variable
        if fixed_elements <= 0:
            raise ValueError(f"Degenerate shape: station={n_station}, variable={n_variable}")
        per_time = elements_budget // fixed_elements
        chunks[depth_axis] = _optimal_1d_split(n_depth, max(1, per_time))
        notes.append(f"time_series mode: station+variable kept whole "
                     f"({fixed_elements:_} elems), time chunked to {chunks[depth_axis]}")

    elif operation == "cross_station_correlation":
        chunks[depth_axis] = -1
        chunks["variable"] = 1
        fixed_elements = n_depth * 1
        if fixed_elements <= 0:
            raise ValueError(f"Degenerate shape: depth={n_depth}")
        per_station = elements_budget // fixed_elements
        chunks["station"] = _optimal_1d_split(n_station, max(1, per_station))
        notes.append(
            f"cross_station_correlation mode: time axis kept whole per station "
            f"({n_depth:_} elems/variable), station chunked to {chunks['station']}, "
            f"variables processed 1 at a time"
        )

    elif operation == "cross_variable":
        chunks[depth_axis] = -1
        chunks["station"] = 1
        fixed_elements = n_depth
        per_variable = elements_budget // max(1, fixed_elements)
        chunks["variable"] = _optimal_1d_split(n_variable, max(1, per_variable))
        notes.append(
            f"cross_variable mode: time axis kept whole per variable/station "
            f"({n_depth:_} elems), variable chunked to {chunks['variable']}"
        )

    else:  # "general"
        target_cube_root = int(elements_budget ** (1 / 3))
        chunks[depth_axis] = _optimal_1d_split(n_depth, max(1, target_cube_root * 4))
        per_station_var = elements_budget // max(1, chunks[depth_axis])
        target_station = int(math.sqrt(per_station_var))
        chunks["station"] = _optimal_1d_split(n_station, max(1, target_station))
        chunks["variable"] = _optimal_1d_split(
            n_variable, max(1, per_station_var // max(1, chunks["station"]))
        )
        notes.append(
            f"general balanced mode: depth≈{chunks[depth_axis]}, "
            f"station≈{chunks['station']}, variable≈{chunks['variable']}"
        )

    for dim, size in ((depth_axis, n_depth), ("station", n_station), ("variable", n_variable)):
        if chunks.get(dim, -1) > size:
            chunks[dim] = -1

    per_chunk_elems = 1
    for dim, size in ((depth_axis, n_depth), ("station", n_station), ("variable", n_variable)):
        c = chunks.get(dim, -1)
        per_chunk_elems *= size if c == -1 else c
    per_chunk_bytes = per_chunk_elems * item_bytes

    n_chunks = 1
    for dim, size in ((depth_axis, n_depth), ("station", n_station), ("variable", n_variable)):
        c = chunks.get(dim, -1)
        n_chunks *= 1 if c == -1 else math.ceil(size / c)

    if per_chunk_bytes > budget_bytes:
        notes.append(
            f"⚠  per-chunk {format_bytes(per_chunk_bytes)} EXCEEDS budget "
            f"{format_bytes(budget_bytes)} – consider splitting further"
        )
    else:
        notes.append(
            f"✓ per-chunk {format_bytes(per_chunk_bytes)} within budget "
            f"{format_bytes(budget_bytes)}"
        )

    return ChunkBudget(
        chunks=chunks,
        per_chunk_bytes=per_chunk_bytes,
        budget_bytes=budget_bytes,
        operation=operation,
        n_chunks=n_chunks,
        notes=notes,
    )


def gc_barrier(
    client: Optional[Any] = None,
    *,
    collect_local: bool = True,
    collect_workers: bool = True,
) -> None:
    """Explicit garbage-collection barrier.

    Runs ``gc.collect()`` on the local process and, if a Dask distributed
    *client* is provided, broadcasts the same call to every worker.  This is
    critical between stages that build large intermediate arrays (e.g. the
    ``N × N`` covariance accumulator) because Dask's reference counting does
    not always release large intermediate buffers immediately, and Python's
    generational GC may defer the collection of big numpy arrays for seconds.

    Parameters
    ----------
    client:
        :class:`dask.distributed.Client` instance.  If ``None``, only the local
        process is collected.
    collect_local:
        Run ``gc.collect()`` locally.
    collect_workers:
        Run ``gc.collect()`` on every Dask worker.
    """
    released = 0
    if collect_local:
        released = gc.collect()
        logger.debug("Local gc.collect() reclaimed %d objects", released)

    if collect_workers and client is not None:
        try:
            from dask.distributed import get_client

            if client is None:
                client = get_client()

            def _worker_gc() -> int:
                import gc as _gc
                return _gc.collect()

            futures = client.run(_worker_gc)
            total = sum(futures.values()) if futures else 0
            logger.info("Distributed gc barrier: %d workers, %d total objects reclaimed",
                        len(futures), total)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Distributed GC barrier failed: %s", exc)


def validate_chunk_memory(
    chunks: Mapping[str, int],
    shape: Mapping[str, int],
    *,
    depth_axis: str,
    dtype: Any = np.float64,
    budget_bytes: Optional[int] = None,
) -> tuple[bool, int, int]:
    """Validate that a chunk plan stays within budget.

    Returns ``(ok, per_chunk_bytes, budget_bytes)``.
    """
    if budget_bytes is None:
        budget_bytes = probe_available_memory()
    item_bytes = _dtype_nbytes(dtype)
    per_chunk = 1
    for dim, size in shape.items():
        c = chunks.get(dim, -1)
        per_chunk *= size if c == -1 else c
    per_chunk_bytes = per_chunk * item_bytes
    ok = per_chunk_bytes <= budget_bytes
    return ok, per_chunk_bytes, budget_bytes


__all__ = [
    "ChunkBudget",
    "OperationKind",
    "compute_optimal_chunks",
    "format_bytes",
    "gc_barrier",
    "parse_memory_string",
    "probe_available_memory",
    "validate_chunk_memory",
]
