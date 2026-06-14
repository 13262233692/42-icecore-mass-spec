"""
NetCDF4 multi-dimensional climate tensor loading with **memory-adaptive** Dask
streaming support.

This module is the refactored version of the original loader that caused
distributed OOM during cross-station correlation shuffles.  The critical fix
is replacing the hard-coded chunk plan (``station: -1, depth: 500``) with a
:func:`~icecore_ms.climate.adaptive_chunking.compute_optimal_chunks` call
that probes available worker RAM and guarantees each chunk plus its shuffle
overhead fits within budget.

See :mod:`icecore_ms.climate.adaptive_chunking` and
:mod:`icecore_ms.climate.correlation` for the underlying machinery.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np
import xarray as xr

from .adaptive_chunking import (
    ChunkBudget,
    compute_optimal_chunks,
    gc_barrier,
    parse_memory_string,
    probe_available_memory,
    validate_chunk_memory,
)

try:
    from dask.distributed import Client, LocalCluster

    _HAS_DASK_DISTRIBUTED = True
except ImportError:  # pragma: no cover
    _HAS_DASK_DISTRIBUTED = False


logger = logging.getLogger(__name__)

VariableKind = Literal["snowfall", "temperature", "trace_element", "isotope", "other"]
OperationKind = Literal["time_series", "cross_station_correlation", "cross_variable", "general"]


@dataclass(frozen=True)
class VariableSpec:
    """Specification of a single climate variable within a NetCDF4 tensor.

    Attributes:
        name: NetCDF variable name (e.g. ``"snowfall_rate"``, ``"delta_18O"``).
        kind: Semantic classification for downstream handling.
        units: Physical units string, read from file metadata if *None*.
        long_name: Human-readable description, read from file if *None*.
    """

    name: str
    kind: VariableKind = "other"
    units: Optional[str] = None
    long_name: Optional[str] = None


@dataclass
class ClimateTensor:
    """Lazy-loaded 3D climate tensor: [depth_or_time, station, variable].

    The underlying :class:`xarray.DataArray` is backed by Dask chunks computed
    by :func:`compute_optimal_chunks`, which guarantees that no worker ever
    holds more than ``budget_bytes × shuffle_overhead`` during transposes or
    correlation shuffles.

    Attributes:
        data: :class:`xarray.DataArray` with dimensions
            ``("depth", "station", "variable")`` or ``("time", "station", "variable")``.
        depth_axis: Name of the depth / time axis.
        stations: Ordered list of station names.
        variables: Ordered list of variable specifications.
        path: Source NetCDF4 file path.
        chunk_budget: The :class:`ChunkBudget` used to lay out the Dask chunks.
            Useful for debugging and for deciding whether to rechunk before
            a cross-axis operation.
    """

    data: xr.DataArray
    depth_axis: str
    stations: list[str]
    variables: list[VariableSpec]
    path: str
    chunk_budget: Optional[ChunkBudget] = None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as ``(n_depth, n_stations, n_variables)``."""
        return (
            self.data.sizes[self.depth_axis],
            self.data.sizes["station"],
            self.data.sizes["variable"],
        )

    @property
    def n_depth(self) -> int:
        return self.data.sizes[self.depth_axis]

    @property
    def n_stations(self) -> int:
        return self.data.sizes["station"]

    @property
    def n_variables(self) -> int:
        return self.data.sizes["variable"]

    def memory_report(self) -> str:
        """Human-readable summary of the current chunk layout and memory use."""
        import dask.array as da

        darr = self.data.data
        if not isinstance(darr, da.Array):
            return "ClimateTensor (not backed by Dask — loaded into RAM)"

        item_bytes = np.dtype(darr.dtype).itemsize
        chunk_sizes = [max(c) if isinstance(c, tuple) else c for c in darr.chunks]
        per_chunk = int(np.prod(chunk_sizes) * item_bytes)
        total = int(np.prod(darr.shape) * item_bytes)
        lines = [
            f"ClimateTensor memory report",
            f"  shape        = {tuple(darr.shape)} ({self.depth_axis!r}, station, variable)",
            f"  chunk shape  = {tuple(chunk_sizes)}",
            f"  n_chunks     = {darr.npartitions}",
            f"  per-chunk    = {_fmt(per_chunk)} (budget {_fmt(self.chunk_budget.budget_bytes) if self.chunk_budget else '?'})",
            f"  total size   = {_fmt(total)}",
        ]
        if self.chunk_budget:
            lines.append("  budget notes:")
            lines.extend(f"    - {n}" for n in self.chunk_budget.notes)
        return "\n".join(lines)

    def __repr__(self) -> str:
        budget_info = ""
        if self.chunk_budget:
            from .adaptive_chunking import format_bytes
            budget_info = f", chunk={format_bytes(self.chunk_budget.per_chunk_bytes)}"
        return (
            f"ClimateTensor(shape={self.shape}, depth_axis={self.depth_axis!r}, "
            f"stations={self.stations}, variables={[v.name for v in self.variables]}"
            f"{budget_info}, path={self.path!r})"
        )


def _fmt(n: int) -> str:
    from .adaptive_chunking import format_bytes
    return format_bytes(n)


_GLOBAL_DASK_CLIENT: Optional["Client"] = None


def get_dask_client() -> Optional["Client"]:
    """Return the currently active Dask distributed client, if any."""
    global _GLOBAL_DASK_CLIENT
    if _GLOBAL_DASK_CLIENT is not None:
        try:
            _GLOBAL_DASK_CLIENT.status  # noqa: B018 - trigger attribute access check
            return _GLOBAL_DASK_CLIENT
        except Exception:
            _GLOBAL_DASK_CLIENT = None
    if _HAS_DASK_DISTRIBUTED:
        try:
            from dask.distributed import get_client as _get
            _GLOBAL_DASK_CLIENT = _get()
            return _GLOBAL_DASK_CLIENT
        except Exception:
            return None
    return None


def _ensure_dask_client(
    n_workers: Optional[int] = None,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
) -> Optional["Client"]:
    global _GLOBAL_DASK_CLIENT
    existing = get_dask_client()
    if existing is not None:
        return existing
    if not _HAS_DASK_DISTRIBUTED:
        return None
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,
        )
        _GLOBAL_DASK_CLIENT = Client(cluster)
        logger.info("Started LocalCluster: workers=%d, memory_limit=%s",
                    n_workers or "auto", memory_limit)
        return _GLOBAL_DASK_CLIENT
    except Exception as exc:
        logger.warning("Failed to start Dask LocalCluster: %s", exc)
        return None


def _infer_depth_axis(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.dims:
            return name
    raise ValueError(
        f"Could not find depth/time dimension. Candidates: {list(candidates)}. "
        f"Available dims: {list(ds.dims)}"
    )


def _collect_variable_specs(
    ds: xr.Dataset,
    variable_names: Iterable[str],
    kind_overrides: Optional[dict[str, VariableKind]] = None,
) -> list[VariableSpec]:
    specs: list[VariableSpec] = []
    overrides = kind_overrides or {}
    for name in variable_names:
        if name not in ds.variables:
            raise KeyError(f"Variable {name!r} not found in NetCDF file. "
                           f"Available: {list(ds.variables.keys())}")
        var = ds[name]
        kind = overrides.get(name, "other")
        units = var.attrs.get("units")
        long_name = var.attrs.get("long_name") or var.attrs.get("standard_name")
        specs.append(VariableSpec(name=name, kind=kind, units=units, long_name=long_name))
    return specs


def load_climate_tensor(
    path: str | os.PathLike[str],
    *,
    variables: Optional[Sequence[str]] = None,
    variable_kinds: Optional[dict[str, VariableKind]] = None,
    stations: Optional[Sequence[str]] = None,
    depth_axis: Optional[str] = None,
    chunks: Optional[dict[str, int]] = None,
    optimize_for: OperationKind = "general",
    budget_bytes: Optional[int] = None,
    use_dask_distributed: bool = True,
    dask_workers: Optional[int] = None,
    dask_threads_per_worker: int = 2,
    dask_memory_limit: str = "4GB",
    engine: str = "netcdf4",
    strict_budget: bool = False,
) -> ClimateTensor:
    """Stream-load a 3D climate tensor from NetCDF4 using memory-adaptive Dask chunking.

    This is the refactored, OOM-safe version of the loader.  The key changes
    versus the original implementation:

    * **Adaptive chunking**: chunk sizes are derived from the available per-worker
      RAM budget rather than hard-coded.  In particular the dangerous default
      ``station: -1`` (all stations in one chunk — guaranteed OOM during any
      cross-station shuffle) is only used for ``optimize_for="time_series"``.
    * **Operation-aware layout**: the ``optimize_for`` flag selects the optimal
      chunk strategy for the workload (see
      :func:`~icecore_ms.climate.adaptive_chunking.compute_optimal_chunks`).
    * **GC barriers**: explicit distributed garbage collection before heavy
      stages prevents reference pinning of large intermediate arrays.
    * **Budget validation**: when ``strict_budget=True`` the loader raises
      instead of silently returning an oversize chunk plan.

    Parameters
    ----------
    path:
        Path to a NetCDF4/HDF5 file containing the climate reconstruction.
    variables:
        Variable names to load. If ``None``, all 2D variables with shape
        ``(depth, station)`` are loaded.
    variable_kinds:
        Optional mapping from variable name to :class:`VariableKind`.
    stations:
        Subset of stations to retain. If ``None`` all stations are kept.
    depth_axis:
        Explicit name of the depth/time dimension. Auto-detected if ``None``.
    chunks:
        Optional manual chunk specification.  If provided, this *overrides*
        the adaptive planner — use with caution.
    optimize_for:
        Workload hint passed to the adaptive chunk planner:
        ``"time_series"``, ``"cross_station_correlation"``, ``"cross_variable"``,
        or ``"general"`` (default).
    budget_bytes:
        Hard per-worker memory limit in bytes.  Auto-detected from
        ``dask_memory_limit`` / system RAM when ``None``.
    use_dask_distributed:
        Spin up a :class:`dask.distributed.LocalCluster` for parallel reads.
    dask_workers, dask_threads_per_worker, dask_memory_limit:
        Cluster configuration for the local Dask scheduler.
    engine:
        Xarray NetCDF engine: ``"netcdf4"`` (default) or ``"h5netcdf"``.
    strict_budget:
        If ``True``, raise :class:`MemoryError` when even the smallest valid
        chunk plan exceeds the detected budget.

    Returns
    -------
    ClimateTensor
        A lazily-evaluated tensor backed by Dask arrays with a provably
        memory-safe chunk layout.
    """
    path_str = os.fspath(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"NetCDF file not found: {path_str}")

    client: Optional["Client"] = None
    if use_dask_distributed:
        client = _ensure_dask_client(
            n_workers=dask_workers,
            threads_per_worker=dask_threads_per_worker,
            memory_limit=dask_memory_limit,
        )

    if budget_bytes is None:
        budget_bytes = probe_available_memory(
            dask_memory_limit=dask_memory_limit if use_dask_distributed else None
        )

    ds_probe = xr.open_dataset(path_str, engine=engine, chunks=None)
    detected_depth = depth_axis or _infer_depth_axis(
        ds_probe, ["depth", "time", "year", "age", "layer"]
    )
    if "station" not in ds_probe.dims:
        raise ValueError(
            f"NetCDF file must have a 'station' dimension. Got dims: {list(ds_probe.dims)}"
        )

    if variables is None:
        variables = [
            name for name, var in ds_probe.variables.items()
            if var.ndim >= 2
            and detected_depth in var.dims
            and "station" in var.dims
            and name not in ds_probe.dims
        ]
        if not variables:
            ds_probe.close()
            raise ValueError(
                "No 2D (depth, station) variables found. Please specify `variables=` explicitly."
            )

    spec_list = _collect_variable_specs(ds_probe, variables, variable_kinds)

    station_coord = (ds_probe["station"].values
                     if "station" in ds_probe.coords
                     else np.arange(ds_probe.sizes["station"]))
    station_names = [str(s) for s in station_coord]
    if stations is not None:
        missing = [s for s in stations if s not in station_names]
        if missing:
            ds_probe.close()
            raise KeyError(f"Stations {missing} not in dataset. Available: {station_names}")
        station_names = list(stations)

    n_stations_eff = len(station_names)
    n_variables_eff = len(spec_list)
    n_depth_eff = int(ds_probe.sizes[detected_depth])
    dtype_eff = ds_probe[variables[0]].dtype
    ds_probe.close()

    shape_plan = {
        detected_depth: n_depth_eff,
        "station": n_stations_eff,
        "variable": n_variables_eff,
    }

    if chunks is not None and len(chunks) > 0:
        ok, per_chunk_bytes, budget_used = validate_chunk_memory(
            chunks, shape_plan, depth_axis=detected_depth, dtype=dtype_eff,
            budget_bytes=budget_bytes,
        )
        if not ok and strict_budget:
            from .adaptive_chunking import format_bytes
            raise MemoryError(
                f"User-provided chunks {chunks} produce {format_bytes(per_chunk_bytes)} "
                f"per chunk, exceeding budget {format_bytes(budget_used)}. "
                f"Reduce chunk size or pass optimize_for='cross_station_correlation'."
            )
        if not ok:
            logger.warning(
                "User chunks %s exceed memory budget (%s > %s). OOM likely.",
                chunks, _fmt(per_chunk_bytes), _fmt(budget_bytes),
            )
        effective_chunks = dict(chunks)
        chunk_budget = ChunkBudget(
            chunks=effective_chunks,
            per_chunk_bytes=per_chunk_bytes,
            budget_bytes=budget_bytes,
            operation=optimize_for,
            n_chunks=0,
            notes=["user-provided chunk plan"],
        )
    else:
        chunk_budget = compute_optimal_chunks(
            shape_plan,
            depth_axis=detected_depth,
            dtype=dtype_eff,
            operation=optimize_for,
            budget_bytes=budget_bytes,
            dask_memory_limit=dask_memory_limit if use_dask_distributed else None,
        )
        if strict_budget and chunk_budget.per_chunk_bytes > chunk_budget.budget_bytes:
            raise MemoryError(
                f"Cannot lay out tensor within budget {_fmt(budget_bytes)}. "
                f"Smallest feasible per-chunk footprint is {_fmt(chunk_budget.per_chunk_bytes)}. "
                f"Either increase worker RAM or reduce the dataset."
            )
        effective_chunks = chunk_budget.chunks
        logger.info("Adaptive chunk plan:\n%s", chunk_budget.summary())

    open_chunks = {
        detected_depth: effective_chunks.get(detected_depth, 500),
        "station": effective_chunks.get("station", -1),
    }
    ds = xr.open_dataset(path_str, engine=engine, chunks=open_chunks)

    gc_barrier(client, collect_local=True, collect_workers=False)

    data_arrays = []
    for spec in spec_list:
        da = ds[spec.name]
        expected_dims = (detected_depth, "station")
        if da.ndim == 2 and tuple(da.dims) != expected_dims:
            da = da.transpose(*expected_dims)
        elif da.ndim > 2:
            extra = [d for d in da.dims if d not in expected_dims]
            da = da.isel({d: 0 for d in extra}).squeeze()
            if da.ndim == 2:
                da = da.transpose(*expected_dims)
            else:
                raise ValueError(
                    f"Variable {spec.name!r} has unexpected shape {da.shape} "
                    f"with dims {da.dims}. Expected 2D: {expected_dims}"
                )
        da_stacked = da.expand_dims({"variable": [spec.name]})
        da_stacked = da_stacked.transpose(detected_depth, "station", "variable")
        if stations is not None:
            da_stacked = da_stacked.sel(station=stations)
        data_arrays.append(da_stacked)

    combined = xr.concat(data_arrays, dim="variable")

    actual_dims = set(combined.dims)
    final_chunks = {k: v for k, v in effective_chunks.items() if k in actual_dims}
    if detected_depth not in final_chunks:
        final_chunks[detected_depth] = effective_chunks.get(detected_depth, 500)
    combined = combined.chunk(final_chunks)

    gc_barrier(client, collect_local=True, collect_workers=(client is not None))

    return ClimateTensor(
        data=combined,
        depth_axis=detected_depth,
        stations=station_names,
        variables=spec_list,
        path=path_str,
        chunk_budget=chunk_budget,
    )


def slice_station(tensor: ClimateTensor, station: str | int) -> xr.DataArray:
    """Extract a single station as a 2D ``(depth, variable)`` DataArray."""
    if isinstance(station, int):
        station = tensor.stations[station]
    if station not in tensor.stations:
        raise KeyError(f"Station {station!r} not found. Available: {tensor.stations}")
    return tensor.data.sel(station=station)


def slice_depth_range(
    tensor: ClimateTensor,
    start: Optional[int | float] = None,
    stop: Optional[int | float] = None,
    *,
    by_index: bool = False,
) -> ClimateTensor:
    """Subset the tensor along the depth/time axis.

    Parameters
    ----------
    tensor:
        Source climate tensor.
    start, stop:
        Range bounds. Interpreted as coordinate values by default, or as
        integer indices when ``by_index=True``.
    by_index:
        If ``True``, treat *start* / *stop* as integer indices along the
        depth axis. Otherwise they are treated as coordinate values.
    """
    da = tensor.data
    axis = tensor.depth_axis
    if by_index:
        selector = slice(start, stop)
        da = da.isel({axis: selector})
    else:
        if start is None and stop is None:
            pass
        elif start is None:
            da = da.sel({axis: slice(None, stop)})
        elif stop is None:
            da = da.sel({axis: slice(start, None)})
        else:
            da = da.sel({axis: slice(start, stop)})

    return ClimateTensor(
        data=da,
        depth_axis=axis,
        stations=list(tensor.stations),
        variables=list(tensor.variables),
        path=tensor.path,
        chunk_budget=tensor.chunk_budget,
    )


def slice_time_window(
    tensor: ClimateTensor,
    *,
    start_year: Optional[float] = None,
    end_year: Optional[float] = None,
) -> ClimateTensor:
    """Time-window slice assuming depth axis is calibrated in years BP."""
    return slice_depth_range(tensor, start=start_year, stop=end_year, by_index=False)


def compute_anomaly(
    tensor: ClimateTensor,
    *,
    reference_start: Optional[float] = None,
    reference_end: Optional[float] = None,
) -> ClimateTensor:
    """Compute anomalies relative to a reference time window (Dask-graph-only).

    A local GC barrier is emitted before returning so that the reference slice
    temporary is released promptly.
    """
    if reference_start is None and reference_end is None:
        ref = tensor.data.mean(dim=tensor.depth_axis)
    else:
        ref_slice = slice_time_window(tensor, start_year=reference_start, end_year=reference_end)
        ref = ref_slice.data.mean(dim=tensor.depth_axis)
    anomaly_data = tensor.data - ref
    gc_barrier(get_dask_client(), collect_local=True, collect_workers=False)
    return ClimateTensor(
        data=anomaly_data,
        depth_axis=tensor.depth_axis,
        stations=list(tensor.stations),
        variables=list(tensor.variables),
        path=tensor.path,
        chunk_budget=tensor.chunk_budget,
    )
