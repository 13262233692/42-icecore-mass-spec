"""Tests for the climate NetCDF4 tensor loading module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from icecore_ms.climate import (
    ClimateTensor,
    VariableSpec,
    compute_anomaly,
    load_climate_tensor,
    slice_depth_range,
    slice_station,
    slice_time_window,
)


def _make_synthetic_netcdf(path: str, n_depth: int = 1000, n_stations: int = 5) -> None:
    rng = np.random.default_rng(42)
    depth = np.arange(n_depth, dtype=np.float64)
    stations = np.array([f"ST{i:02d}" for i in range(n_stations)])

    snowfall = rng.normal(200.0, 30.0, size=(n_depth, n_stations)) + np.linspace(0, 50, n_depth)[:, None]
    temperature = rng.normal(-40.0, 5.0, size=(n_depth, n_stations)) + np.sin(np.linspace(0, 10, n_depth))[:, None] * 3
    dust = rng.gamma(2.0, 1.0, size=(n_depth, n_stations))
    delta_18O = rng.normal(-40.0, 2.0, size=(n_depth, n_stations)) + np.linspace(0, 5, n_depth)[:, None]

    ds = xr.Dataset(
        {
            "snowfall_rate": (("depth", "station"), snowfall, {"units": "mm/yr", "long_name": "Annual snowfall rate"}),
            "temperature": (("depth", "station"), temperature, {"units": "degC", "long_name": "Reconstructed temperature"}),
            "dust_concentration": (("depth", "station"), dust, {"units": "ppb", "long_name": "Dust concentration"}),
            "delta_18O": (("depth", "station"), delta_18O, {"units": "permil", "long_name": "delta 18O isotope ratio"}),
        },
        coords={
            "depth": ("depth", depth, {"units": "m", "long_name": "Ice depth below surface"}),
            "station": ("station", stations, {"long_name": "Antarctic drilling station"}),
        },
        attrs={"source": "synthetic test data", "n_years": n_depth},
    )
    ds.to_netcdf(path, engine="netcdf4")


@pytest.fixture(scope="module")
def synthetic_netcdf(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("climate")
    path = tmp / "synthetic_climate.nc"
    _make_synthetic_netcdf(str(path))
    return str(path)


def test_load_climate_tensor_full(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    assert isinstance(tensor, ClimateTensor)
    assert tensor.n_depth == 1000
    assert tensor.n_stations == 5
    assert tensor.n_variables == 4
    assert tensor.depth_axis == "depth"
    assert len(tensor.stations) == 5
    assert tensor.stations[0] == "ST00"
    assert tensor.shape == (1000, 5, 4)


def test_load_climate_tensor_variable_subset(synthetic_netcdf):
    tensor = load_climate_tensor(
        synthetic_netcdf,
        variables=["temperature", "delta_18O"],
        variable_kinds={"temperature": "temperature", "delta_18O": "isotope"},
        use_dask_distributed=False,
    )
    assert tensor.n_variables == 2
    names = [v.name for v in tensor.variables]
    assert names == ["temperature", "delta_18O"]
    kinds = [v.kind for v in tensor.variables]
    assert kinds == ["temperature", "isotope"]


def test_load_climate_tensor_station_subset(synthetic_netcdf):
    tensor = load_climate_tensor(
        synthetic_netcdf,
        stations=["ST01", "ST03"],
        use_dask_distributed=False,
    )
    assert tensor.n_stations == 2
    assert tensor.stations == ["ST01", "ST03"]


def test_slice_station(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    da = slice_station(tensor, "ST02")
    assert isinstance(da, xr.DataArray)
    assert da.sizes["depth"] == 1000
    assert da.sizes["variable"] == 4
    assert "station" not in da.dims or da.sizes.get("station", 1) == 1


def test_slice_station_by_index(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    da = slice_station(tensor, 2)
    assert da.sizes["depth"] == 1000


def test_slice_depth_range_by_index(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    sub = slice_depth_range(tensor, 100, 300, by_index=True)
    assert sub.n_depth == 200
    assert sub.n_stations == tensor.n_stations
    assert sub.n_variables == tensor.n_variables


def test_slice_time_window(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    sub = slice_time_window(tensor, start_year=100.0, end_year=500.0)
    assert 100 <= sub.n_depth <= 401


def test_compute_anomaly(synthetic_netcdf):
    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    anomaly = compute_anomaly(tensor)
    assert anomaly.shape == tensor.shape
    values = anomaly.data.values
    global_mean = np.nanmean(values, axis=0)
    assert np.allclose(global_mean, 0.0, atol=1e-6)


def test_variablespec_metadata(synthetic_netcdf):
    tensor = load_climate_tensor(
        synthetic_netcdf,
        variables=["snowfall_rate", "temperature"],
        use_dask_distributed=False,
    )
    specs = {v.name: v for v in tensor.variables}
    assert specs["snowfall_rate"].units == "mm/yr"
    assert specs["temperature"].units == "degC"
    assert specs["snowfall_rate"].long_name is not None


def test_dask_backing_preserves_laziness(synthetic_netcdf):
    import dask.array as da

    tensor = load_climate_tensor(synthetic_netcdf, use_dask_distributed=False)
    assert isinstance(tensor.data.data, da.Array)
    computed = tensor.data.isel(depth=slice(0, 10)).compute()
    assert computed.shape == (10, 5, 4)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_climate_tensor("/nonexistent/path.nc", use_dask_distributed=False)


def test_missing_variable_raises(synthetic_netcdf):
    with pytest.raises(KeyError):
        load_climate_tensor(synthetic_netcdf, variables=["not_a_var"], use_dask_distributed=False)


def test_missing_station_raises(synthetic_netcdf):
    with pytest.raises(KeyError):
        load_climate_tensor(synthetic_netcdf, stations=["NOPE"], use_dask_distributed=False)
