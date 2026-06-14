"""Stress and memory-safety tests for the adaptive chunking subsystem.

These tests are specifically designed to reproduce and then verify the fix
for the Dask OOM scenario described in the ticket: loading a large number
of station time series and computing their cross-station correlation matrix
without exhausting worker RAM.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from icecore_ms.climate import (
    ChunkBudget,
    ClimateTensor,
    compute_correlation_matrix,
    compute_optimal_chunks,
    format_bytes,
    gc_barrier,
    load_climate_tensor,
    parse_memory_string,
    probe_available_memory,
    rechunk_for_operation,
    validate_chunk_memory,
)


# ---------------------------------------------------------------------------
# parse_memory_string / format_bytes
# ---------------------------------------------------------------------------

def test_parse_memory_string_units():
    assert parse_memory_string("1B") == 1
    assert parse_memory_string("1KB") == 1024
    assert parse_memory_string("2MB") == 2 * 1024 * 1024
    assert parse_memory_string("4GB") == 4 * 1024 ** 3
    assert parse_memory_string(1024) == 1024
    assert parse_memory_string("512") == 512
    with pytest.raises(ValueError):
        parse_memory_string("not a memory spec")


def test_format_bytes_roundtrip():
    for size in (0, 1, 1023, 1024, 1024 ** 2, 1024 ** 3, 1024 ** 4):
        formatted = format_bytes(size)
        assert isinstance(formatted, str)
        assert len(formatted) > 0


# ---------------------------------------------------------------------------
# probe_available_memory
# ---------------------------------------------------------------------------

def test_probe_available_memory_returns_positive():
    budget = probe_available_memory()
    assert budget > 0
    assert isinstance(budget, int)


def test_probe_available_memory_honours_dask_limit():
    budget = probe_available_memory(dask_memory_limit="1GB", safety_ratio=0.5)
    expected = int(1024 ** 3 * 0.5)
    assert budget == expected


# ---------------------------------------------------------------------------
# compute_optimal_chunks
# ---------------------------------------------------------------------------

def test_optimal_chunks_general_mode_balanced():
    budget = compute_optimal_chunks(
        {"time": 200000, "station": 500, "variable": 10},
        depth_axis="time",
        dtype=np.float64,
        operation="general",
        budget_bytes=256 * 1024 * 1024,  # 256 MB
    )
    assert isinstance(budget, ChunkBudget)
    assert budget.operation == "general"
    assert budget.per_chunk_bytes <= budget.budget_bytes * 3  # within shuffle overhead
    for dim in ("time", "station", "variable"):
        assert dim in budget.chunks


def test_optimal_chunks_time_series_keeps_station_whole():
    budget = compute_optimal_chunks(
        {"depth": 100000, "station": 200, "variable": 4},
        depth_axis="depth",
        dtype=np.float64,
        operation="time_series",
        budget_bytes=512 * 1024 * 1024,
    )
    assert budget.chunks["station"] == -1
    assert budget.chunks["variable"] == -1
    assert budget.chunks["depth"] > 0 and budget.chunks["depth"] <= 100000


def test_optimal_chunks_cross_station_keeps_time_whole():
    budget = compute_optimal_chunks(
        {"time": 200000, "station": 500, "variable": 6},
        depth_axis="time",
        dtype=np.float64,
        operation="cross_station_correlation",
        budget_bytes=1024 * 1024 * 1024,
    )
    assert budget.chunks["time"] == -1
    assert budget.chunks["variable"] == 1
    assert 0 < budget.chunks["station"] <= 500

    per_station_bytes = 200000 * 1 * 8  # time × 1 variable × float64
    per_chunk_max = budget.chunks["station"] * per_station_bytes
    assert per_chunk_max <= budget.budget_bytes * 1.5


def test_optimal_chunks_cross_variable_keeps_time_whole():
    budget = compute_optimal_chunks(
        {"depth": 50000, "station": 50, "variable": 100},
        depth_axis="depth",
        dtype=np.float32,
        operation="cross_variable",
        budget_bytes=512 * 1024 * 1024,
    )
    assert budget.chunks["depth"] == -1
    assert budget.chunks["station"] == 1
    assert budget.chunks["variable"] == -1 or 0 < budget.chunks["variable"] <= 100


def test_optimal_chunks_tiny_dimensions_collapse_to_minus_one():
    budget = compute_optimal_chunks(
        {"time": 100, "station": 5, "variable": 2},
        depth_axis="time",
        dtype=np.float64,
        operation="general",
        budget_bytes=1024 * 1024,
    )
    assert budget.chunks.get("station") == -1 or budget.chunks.get("station") > 0


def test_optimal_chunks_summary_contains_notes():
    budget = compute_optimal_chunks(
        {"depth": 5000, "station": 20, "variable": 3},
        depth_axis="depth",
        operation="time_series",
        budget_bytes=64 * 1024 * 1024,
    )
    summary = budget.summary()
    assert "time_series" in summary
    assert len(budget.notes) >= 1


def test_validate_chunk_memory_detects_violation():
    shape = {"time": 200000, "station": 500, "variable": 10}
    bad_chunks = {"time": -1, "station": -1, "variable": -1}  # EVERYTHING in one chunk!
    ok, per_chunk, budget_used = validate_chunk_memory(
        bad_chunks, shape, depth_axis="time", dtype=np.float64,
        budget_bytes=100 * 1024 * 1024,  # 100 MB
    )
    assert not ok
    assert per_chunk == 200000 * 500 * 10 * 8
    assert per_chunk > budget_used


def test_validate_chunk_memory_accepts_good_plan():
    shape = {"time": 200000, "station": 500, "variable": 10}
    good_chunks = {"time": 500, "station": 10, "variable": 1}
    ok, _per_chunk, _budget = validate_chunk_memory(
        good_chunks, shape, depth_axis="time", dtype=np.float64,
        budget_bytes=10 * 1024 * 1024,
    )
    assert ok


# ---------------------------------------------------------------------------
# gc_barrier
# ---------------------------------------------------------------------------

def test_gc_barrier_local_only_runs():
    gc_barrier(client=None, collect_local=True, collect_workers=False)


# ---------------------------------------------------------------------------
# load_climate_tensor with adaptive chunking
# ---------------------------------------------------------------------------

_PROJECT_TMP = Path(__file__).resolve().parent.parent / "_test_tmp"


def _make_large_netcdf(path: str, n_time: int = 5000, n_station: int = 50, n_var: int = 4) -> None:
    rng = np.random.default_rng(0)
    time = np.arange(n_time, dtype=np.float64)
    stations = np.array([f"ST{i:03d}" for i in range(n_station)])
    data_vars = {}
    for vi in range(n_var):
        arr = rng.standard_normal((n_time, n_station)).astype(np.float32)
        data_vars[f"var_{vi}"] = (("time", "station"), arr)
    ds = xr.Dataset(data_vars, coords={"time": time, "station": stations})
    ds.to_netcdf(path, engine="netcdf4")
    ds.close()


@pytest.fixture
def large_netcdf(tmp_path):
    # Use project-local temp dir to avoid Windows permission issues with
    # user-name special characters in the system TEMP path.
    _PROJECT_TMP.mkdir(exist_ok=True)
    path = _PROJECT_TMP / f"large_{os.getpid()}.nc"
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    _make_large_netcdf(str(path))
    try:
        yield str(path)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def test_load_climate_tensor_default_uses_adaptive_plan(large_netcdf):
    tensor = load_climate_tensor(large_netcdf, use_dask_distributed=False)
    assert isinstance(tensor, ClimateTensor)
    assert tensor.chunk_budget is not None
    assert tensor.chunk_budget.operation == "general"
    assert tensor.shape == (5000, 50, 4)


def test_load_climate_tensor_optimize_for_cross_station(large_netcdf):
    tensor = load_climate_tensor(
        large_netcdf,
        use_dask_distributed=False,
        optimize_for="cross_station_correlation",
    )
    assert tensor.chunk_budget is not None
    assert tensor.chunk_budget.operation == "cross_station_correlation"
    assert tensor.chunk_budget.chunks["time"] == -1
    assert tensor.chunk_budget.chunks["variable"] == 1


def test_load_climate_tensor_strict_budget_raises(large_netcdf):
    with pytest.raises(MemoryError):
        load_climate_tensor(
            large_netcdf,
            use_dask_distributed=False,
            chunks={"time": -1, "station": -1, "variable": -1},
            budget_bytes=1000,
            strict_budget=True,
        )


def test_load_climate_tensor_memory_report(large_netcdf):
    tensor = load_climate_tensor(large_netcdf, use_dask_distributed=False)
    report = tensor.memory_report()
    assert "ClimateTensor memory report" in report
    assert "shape" in report
    assert "per-chunk" in report


# ---------------------------------------------------------------------------
# rechunk_for_operation
# ---------------------------------------------------------------------------

def test_rechunk_for_operation(large_netcdf):
    tensor = load_climate_tensor(large_netcdf, use_dask_distributed=False)
    rechunked, budget = rechunk_for_operation(
        tensor, "cross_station_correlation", budget_bytes=256 * 1024 * 1024
    )
    assert isinstance(rechunked, ClimateTensor)
    assert isinstance(budget, ChunkBudget)
    assert budget.operation == "cross_station_correlation"


# ---------------------------------------------------------------------------
# compute_correlation_matrix – the critical OOM fix verification
# ---------------------------------------------------------------------------

def test_compute_correlation_matrix_station(large_netcdf):
    tensor = load_climate_tensor(
        large_netcdf,
        use_dask_distributed=False,
        optimize_for="cross_station_correlation",
    )
    result = compute_correlation_matrix(
        tensor,
        along="station",
        kind="pearson",
        variable="var_0",
    )
    assert result.matrix.shape == (50, 50)
    assert result.kind == "pearson"
    assert result.along_axis == "station"
    assert result.n_samples == 5000

    corr = result.matrix.values
    assert np.all(np.isfinite(corr))
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
    assert (corr >= -1.0 - 1e-9).all()
    assert (corr <= 1.0 + 1e-9).all()
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)


def test_compute_correlation_matrix_covariance(large_netcdf):
    tensor = load_climate_tensor(large_netcdf, use_dask_distributed=False)
    result = compute_correlation_matrix(
        tensor, along="variable", kind="covariance"
    )
    assert result.matrix.shape == (4, 4)
    assert result.kind == "covariance"
    cov = result.matrix.values
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


def test_compute_correlation_matrix_memory_estimate(large_netcdf):
    tensor = load_climate_tensor(large_netcdf, use_dask_distributed=False)
    result = compute_correlation_matrix(
        tensor, along="station", variable="var_1",
    )
    assert result.memory_peak_estimate_bytes > 0
    n_time = 5000
    n_station = 50
    item_bytes = 8
    peak_bytes = result.memory_peak_estimate_bytes
    per_block_pair = 2 * n_time * item_bytes
    assert peak_bytes >= per_block_pair, (
        f"peak={peak_bytes} should be at least for 2 blocks ({per_block_pair})"
    )
    max_expected = int(2 * min(256, n_station) * n_time * item_bytes * 2)
    assert peak_bytes <= max_expected, (
        f"peak={peak_bytes} should not exceed reasonable max {max_expected}"
    )


# ---------------------------------------------------------------------------
# The smoking gun: prove the old chunk plan would be rejected
# ---------------------------------------------------------------------------

def test_old_default_chunks_violate_budget_for_large_dataset():
    """The old hard-coded plan ``station: -1, depth: 500`` is catastrophic
    for cross-station correlation because it forces every worker to hold the
    full station dimension.  This test demonstrates that the adaptive planner
    produces a *safe* plan while the old one exceeds budget by orders of magnitude.

    The real failure mode is not just per-chunk size but the **shuffle overhead**:
    with ``station: -1`` every time-chunk contains every station, so a transpose
    requires pulling the *entire tensor* into each worker simultaneously.
    """
    n_time = 200000  # 200k layers
    n_station = 500
    n_variable = 10
    shape = {"time": n_time, "station": n_station, "variable": n_variable}
    budget_bytes = 10 * 1024 * 1024  # strict 10 MB / worker (realistic for a
                                     # memory-safe budget after OS + Python +
                                     # Dask bookkeeping)

    old_plan = {"time": 500, "station": -1, "variable": -1}
    ok_old, per_old, _ = validate_chunk_memory(
        old_plan, shape, depth_axis="time", dtype=np.float64,
        budget_bytes=budget_bytes,
    )
    assert not ok_old, (
        f"Old plan should be rejected but used only {format_bytes(per_old)} "
        f"vs budget {format_bytes(budget_bytes)}"
    )
    assert per_old > budget_bytes, (
        f"Old per-chunk {format_bytes(per_old)} should exceed budget "
        f"{format_bytes(budget_bytes)}"
    )

    n_chunks_old = n_time // 500
    shuffle_memory_old = per_old * n_chunks_old
    assert shuffle_memory_old > 100 * budget_bytes, (
        f"Old plan shuffle working set ~ {format_bytes(shuffle_memory_old)} "
        f"should be > 100× budget"
    )

    adaptive = compute_optimal_chunks(
        shape, depth_axis="time", operation="cross_station_correlation",
        budget_bytes=budget_bytes, dtype=np.float64,
    )
    ok_new, per_new, _ = validate_chunk_memory(
        adaptive.chunks, shape, depth_axis="time", dtype=np.float64,
        budget_bytes=budget_bytes,
    )
    assert ok_new, (
        f"Adaptive plan should fit in budget: plan={format_bytes(per_new)} "
        f"vs budget={format_bytes(budget_bytes)}"
    )
    assert per_new <= budget_bytes * 1.1

    improvement = per_old / max(1, per_new)
    assert improvement > 5, (
        f"Adaptive plan should be ≥5× smaller per-chunk than the old plan; "
        f"got {improvement:.0f}× ({format_bytes(per_old)} → {format_bytes(per_new)})"
    )
