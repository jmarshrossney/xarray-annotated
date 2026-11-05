"""Dask-backed arrays: nothing is computed until the caller asks."""

from typing import Annotated

import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Coords, Dims, Dtype, Freq, Unit, units

dask = pytest.importorskip("dask")
dask_array = pytest.importorskip("dask.array")


@pytest.fixture
def computes():
    """Count the times the array from `lazy` is computed."""
    return []


@pytest.fixture
def lazy(computes):
    """Build a dask-backed DataArray that records every computation."""

    def build(unit="hPa"):
        def read():
            computes.append(1)
            return np.array([1000.0, 1010.0, 1020.0])

        data = dask_array.from_delayed(dask.delayed(read)(), shape=(3,), dtype=float)
        time = pd.date_range("2020-01-01", periods=3, freq="D")
        return xr.DataArray(
            data, dims="time", coords={"time": time}, attrs={"units": unit}
        )

    return build


class TestNothingIsComputed:
    def test_convert(self, lazy, computes):
        out = units.convert(lazy(), "Pa")
        assert isinstance(out.data, dask_array.Array)
        assert computes == []
        np.testing.assert_allclose(out.values, [100000.0, 101000.0, 102000.0])
        assert computes == [1]
        assert out.attrs["units"] == "Pa"

    def test_check_units(self, lazy, computes):
        out = units.check_units(lazy(), "hectopascal", "p")
        assert isinstance(out.data, dask_array.Array)
        assert computes == []

    def test_enforce_with_every_marker(self, lazy, computes):
        @xra.enforce
        def f(
            p: Annotated[
                xr.DataArray,
                Unit("hPa"),
                Dims("time"),
                Coords("time"),
                Dtype("float64"),
                Freq("D"),
            ],
        ) -> Annotated[xr.DataArray, Unit("hPa"), Dims("time"), Freq("D")]:
            return p * 2

        out = f(lazy())
        assert isinstance(out.data, dask_array.Array)
        assert computes == []
        np.testing.assert_allclose(out.values, [2000.0, 2020.0, 2040.0])


class TestQuantified:
    """A quantified dask array holds a `Quantity` wrapping a dask array."""

    def test_check_units_leaves_it_uncomputed(self, lazy, computes):
        da = lazy().pint.quantify()
        assert units.check_units(da, "hPa", "p") is da
        assert computes == []

    def test_dequantify_leaves_it_uncomputed(self, lazy, computes):
        out = units.dequantify(units.quantify(lazy()))
        assert isinstance(out.data, dask_array.Array)
        assert computes == []

    def test_quantify_leaves_it_uncomputed(self, lazy, computes):
        out = units.quantify(lazy())
        assert isinstance(out.data.magnitude, dask_array.Array)
        assert computes == []

    def test_convert_keeps_it_quantified_and_uncomputed(self, lazy, computes):
        out = units.convert(lazy().pint.quantify(), "Pa")
        assert isinstance(out.data.magnitude, dask_array.Array)
        assert computes == []
        np.testing.assert_allclose(
            out.data.magnitude.compute(), [100000.0, 101000.0, 102000.0]
        )

    def test_output_stays_quantified_and_uncomputed(self, lazy, computes):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return lazy().pint.quantify().pint.to("Pa")

        out = f()
        assert isinstance(out.data.magnitude, dask_array.Array)
        assert computes == []


class TestSwitchInWorkers:
    """`enabled` is a `ContextVar`, and dask's in-process schedulers carry it over."""

    @staticmethod
    def _checked() -> bool:
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        try:
            f(xr.DataArray([1.0], attrs={"units": "hPa"}))
        except ValueError:
            return True
        return False

    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    def test_a_block_around_compute_reaches_the_tasks(self, scheduler):
        task = dask.delayed(self._checked)()
        assert task.compute(scheduler=scheduler) is True
        with xra.enabled(False):
            assert task.compute(scheduler=scheduler) is False

    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    def test_a_block_around_the_graph_alone_does_not(self, scheduler):
        with xra.enabled(False):
            task = dask.delayed(self._checked)()
        assert task.compute(scheduler=scheduler) is True
