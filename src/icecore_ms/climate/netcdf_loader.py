"""
NetCDF4 multi-dimensional climate tensor loading with Dask streaming support.

This module provides high-throughput, memory-efficient access to multi-millennial
polar climate reconstruction datasets stored as NetCDF4/HDF5 tensors. The tensors
typically have three dimensions: depth/time (thousands of annual layers),
station (multiple Antarctic drilling sites), and variable (snowfall, temperature,
trace elements like dust, Na+, Ca2+).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np
import xarray as xr

try:
    from dask.distributed import Client, LocalCluster

    _HAS_DASK_DISTRIBUTED = True
except ImportError:  # pragma: no cover
    _HAS_DASK_DISTRIBUTED = False


VariableKind = Literal["snowfall", "temperature", "trace_element", "isotope", "other"]


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

    The underlying :class:`xarray.DataArray` is backed by Dask chunks so that
    slices can be computed on demand without loading the full multi-GB dataset
    into memory.

    Attributes:
        data: :class:`xarray.DataArray` with dimensions
            ``("depth", "station", "variable")`` or ``("time", "station", "variable")``.
        depth_axis: Name of the depth / time axis.
        stations: Ordered list of station names.
        variables: Ordered list of variable specifications.
        path: Source NetCDF4 file path.
    """

    data: xr.DataArray
    depth_axis: str
    stations: list[str]
    variables: list[VariableSpec]
    path: str

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

    def __repr__(self) -> str:
        return (
            f"ClimateTensor(shape={self.shape}, depth_axis={self.depth_axis!r}, "
            f"stations={self.stations}, variables={[v.name for v in self.variables]}, "
            f"path={self.path!r})"
        )


def _ensure_dask_client(
    n_workers: Optional[int] = None,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
) -> Optional["Client"]:
    if not _HAS_DASK_DISTRIBUTED:
        return None
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,
        )
        return Client(cluster)
    except Exception:
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
    use_dask_distributed: bool = True,
    dask_workers: Optional[int] = None,
    dask_threads_per_worker: int = 2,
    dask_memory_limit: str = "4GB",
    engine: str = "netcdf4",
) -> ClimateTensor:
    """Stream-load a 3D climate tensor from NetCDF4 using Dask.

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
        Dask chunk specification (``{dim_name: chunk_size}``). The default
        chunks along the depth axis are 500 layers so that slices fit in RAM.
    use_dask_distributed:
        Spin up a :class:`dask.distributed.LocalCluster` for parallel reads.
    dask_workers, dask_threads_per_worker, dask_memory_limit:
        Cluster configuration for the local Dask scheduler.
    engine:
        Xarray NetCDF engine: ``"netcdf4"`` (default) or ``"h5netcdf"``.

    Returns
    -------
    ClimateTensor
        A lazily-evaluated tensor backed by Dask arrays.
    """
    path_str = os.fspath(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"NetCDF file not found: {path_str}")

    if use_dask_distributed:
        _ensure_dask_client(
            n_workers=dask_workers,
            threads_per_worker=dask_threads_per_worker,
            memory_limit=dask_memory_limit,
        )

    default_chunks = {"depth": 500, "time": 500, "station": -1, "variable": 1}
    effective_chunks = dict(default_chunks)
    if chunks:
        effective_chunks.update(chunks)

    ds = xr.open_dataset(path_str, engine=engine, chunks=effective_chunks)

    detected_depth = depth_axis or _infer_depth_axis(
        ds, ["depth", "time", "year", "age", "layer"]
    )
    if "station" not in ds.dims:
        raise ValueError(
            f"NetCDF file must have a 'station' dimension. Got dims: {list(ds.dims)}"
        )

    if variables is None:
        variables = [
            name for name, var in ds.variables.items()
            if var.ndim >= 2
            and detected_depth in var.dims
            and "station" in var.dims
            and name not in ds.dims
        ]
        if not variables:
            raise ValueError(
                "No 2D (depth, station) variables found. Please specify `variables=` explicitly."
            )

    spec_list = _collect_variable_specs(ds, variables, variable_kinds)

    station_coord = ds["station"].values if "station" in ds.coords else np.arange(ds.sizes["station"])
    station_names = [str(s) for s in station_coord]
    if stations is not None:
        missing = [s for s in stations if s not in station_names]
        if missing:
            raise KeyError(f"Stations {missing} not in dataset. Available: {station_names}")
        station_names = list(stations)

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
    filtered_chunks = {k: v for k, v in effective_chunks.items() if k in actual_dims}
    if detected_depth not in filtered_chunks:
        filtered_chunks[detected_depth] = 500
    combined = combined.chunk(filtered_chunks)

    return ClimateTensor(
        data=combined,
        depth_axis=detected_depth,
        stations=station_names,
        variables=spec_list,
        path=path_str,
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
    """Compute anomalies relative to a reference time window (in-place on Dask graph)."""
    if reference_start is None and reference_end is None:
        ref = tensor.data.mean(dim=tensor.depth_axis)
    else:
        ref_slice = slice_time_window(tensor, start_year=reference_start, end_year=reference_end)
        ref = ref_slice.data.mean(dim=tensor.depth_axis)
    anomaly_data = tensor.data - ref
    return ClimateTensor(
        data=anomaly_data,
        depth_axis=tensor.depth_axis,
        stations=list(tensor.stations),
        variables=list(tensor.variables),
        path=tensor.path,
    )
