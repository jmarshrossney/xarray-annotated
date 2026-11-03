"""Tests for composing multiple facets under one ``@enforce``.

1. Several structural markers on one hint.
2. Units alongside structure — units converts the input first, schema validates
   the converted array; on the way out units stamps (or converts) first, schema
   validates what the caller receives.
3. All three domains on one hint: each reads only its own markers, so one hint
   can declare a unit, a structure, and a frequency at once.

The markers of every domain coexist in one ``Annotated``, and each domain reads only
its own.
"""

from typing import Annotated

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import schema, temporal
from xarray_annotated.schema import Coords, Dims, Dtype
from xarray_annotated.temporal import Freq
from xarray_annotated.units import Unit


def _da(dims=("time", "x"), dtype="float64", coords=(), unit=None, fill=0.0):
    """Build a DataArray with dims/dtype/coords and an optional units attr.

    ``fill`` sets the constant array value so multiplicative unit conversions are
    observable (zeros would hide a ``hPa`` -> ``Pa`` conversion).
    """
    shape = tuple(range(2, 2 + len(dims)))
    arr = np.full(shape, fill, dtype=dtype)
    coord_map = {
        d: np.arange(s) for d, s in zip(dims, shape, strict=True) if d in coords
    }
    da = xr.DataArray(arr, dims=dims, coords=coord_map)
    if unit is not None:
        da.attrs["units"] = unit
    return da


#: One hint declaring all three facets at once, aliased so the same declaration can be
#: reused for a parameter and a return.
Weekly = Annotated[xr.DataArray, Unit("degC"), Dims("time"), Freq("7D")]
WeeklySun = Annotated[xr.DataArray, Unit("degC"), Dims("time"), Freq("W-SUN")]

#: The lazy counterpart, which `get_type_hints` hands back unresolved.
type LazyWeekly = Annotated[xr.DataArray, Unit("degC"), Dims("time"), Freq("7D")]


def _timeseries(freq="7D", periods=5, unit=None, fill=0.0):
    """A 1-d DataArray over a datetime axis of the given frequency."""
    times = pd.date_range("2020-01-01", periods=periods, freq=freq)  # a Wednesday
    da = xr.DataArray(
        np.full(periods, fill, dtype="float64"), dims=("time",), coords={"time": times}
    )
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestComposeSchemaMarkers:
    """Multiple structural markers on one hint."""

    def test_dims_and_coords_pass(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time")],
        ) -> xr.DataArray:
            return x

        da = _da(("time", "x"), coords=("time",))
        assert f(da) is da  # never mutates

    def test_dims_ok_but_coord_missing_fails(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time")],
        ) -> xr.DataArray:
            return x

        with pytest.raises(schema.SchemaError, match="missing coords"):
            f(_da(("time", "x")))  # dims satisfied, coordinate label absent

    def test_dims_coords_dtype_each_violation_fails(self):
        @xra.enforce
        def f(
            x: Annotated[
                xr.DataArray, Dims("time", "x"), Coords("time"), Dtype("float64")
            ],
        ) -> xr.DataArray:
            return x

        # all three satisfied
        f(_da(("time", "x"), coords=("time",)))
        # dims wrong, coords/dtype fine
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(_da(("time",), coords=("time",)))
        # dtype wrong, dims/coords fine
        with pytest.raises(schema.SchemaError, match="dtype kind mismatch"):
            f(_da(("time", "x"), coords=("time",), dtype="int64"))


class TestComposeDecorators:
    """Units and structure declared together."""

    def test_dims_and_units_converts_and_validates(self):
        # recommended form: the Unit marker alongside a schema marker
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa", convert=True)],
        ) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]:
            return x

        out = f(_da(("time", "x"), unit="hPa", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 1000.0))

    def test_same_unit_respelling_needs_no_conversion(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")],
        ) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]:
            return x

        # With no opt-in to conversion only a same-unit spelling passes; it is
        # relabelled with the values untouched.
        out = f(_da(("time", "x"), unit="pascal", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 10.0))
        with pytest.raises(ValueError, match="convert=True"):
            f(_da(("time", "x"), unit="hPa", fill=10.0))

    def test_dims_mismatch_raises_with_valid_units(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")],
        ) -> xr.DataArray:
            return x

        # units convert cleanly (Pa -> Pa), then inner schema rejects the dims
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(_da(("time",), unit="Pa", fill=1.0))

    def test_units_dimensional_mismatch_raises_with_valid_dims(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")],
        ) -> xr.DataArray:
            return x

        # outer units runs first: kg vs Pa raises before schema sees the array
        with pytest.raises(pint.DimensionalityError):
            f(_da(("time", "x"), unit="kg", fill=1.0))

    def test_full_composition_dims_coords_dtype_units(self):
        @xra.enforce
        def f(
            x: Annotated[
                xr.DataArray,
                Dims("time", "x"),
                Coords("time"),
                Dtype("float64"),
                Unit("Pa", convert=True),
            ],
        ) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]:
            return x

        out = f(_da(("time", "x"), coords=("time",), unit="hPa", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 1000.0))

    def test_output_stamped_and_validated(self):
        @xra.enforce
        def f(n: int) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]:
            return _da(("time", "x"), fill=5.0)  # raw output, no units attr

        out = f(1)
        assert out.attrs["units"] == "Pa"  # stamped by units
        np.testing.assert_allclose(out.values, np.full(out.shape, 5.0))  # not converted

    def test_output_dims_mismatch_raises(self):
        @xra.enforce
        def f(n: int) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]:
            return _da(("time",), fill=5.0)

        # inner schema validates the output before outer units stamps it
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(1)


class TestComposeAllThreeDomains:
    """One hint, three declarations: ``Unit`` + ``Dims`` + ``Freq``."""

    def test_every_domain_honours_its_own_marker(self):
        @xra.enforce
        def f(x: Weekly) -> Weekly:
            return x

        out = f(_timeseries(unit="degC", fill=10.0))
        assert out.attrs["units"] == "degC"  # units read the Unit marker
        assert out.dims == ("time",)  # schema read the Dims marker
        # ... and temporal read the Freq marker: a wrong-phase axis is rejected.
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f(_timeseries(freq="D", unit="degC"))

    def test_anchored_freq_rejects_wrong_phase_alongside_units(self):
        @xra.enforce
        def f(x: WeeklySun) -> xr.DataArray:
            return x

        # The 7D axis starts on a Wednesday, so W-SUN is violated.
        with pytest.raises(temporal.FreqError, match="expected 'W-SUN', got 'W-WED'"):
            f(_timeseries(unit="degC"))

    def test_pep695_type_alias_declares_like_an_assignment(self):
        def f(x: LazyWeekly) -> None: ...
        def g(x: Weekly) -> None: ...

        assert xra.declarations_from_signature(f) == xra.declarations_from_signature(g)

    def test_bare_string_beside_freq_is_a_description(self):
        # A bare string is a description: never a unit, never a frequency.
        @xra.enforce
        def f(x: Annotated[xr.DataArray, "hPa", Freq("7D")]) -> xr.DataArray:
            return x

        out = f(_timeseries(unit="kg", fill=1.0))
        assert out.attrs["units"] == "kg"

    def test_units_convert_before_freq_check(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Unit("Pa", convert=True), Freq("7D")],
        ) -> Annotated[xr.DataArray, Unit("Pa"), Freq("7D")]:
            return x

        out = f(_timeseries(unit="hPa", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 1000.0))
