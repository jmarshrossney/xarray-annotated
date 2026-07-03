"""Tests for composing multiple checks.

Two documented composition patterns (see docs/usage.md "Combining multiple
checks"):

1. Several structural markers on one hint under a single ``@declare_schema``.
2. Stacking ``@declare_units`` (outer) over ``@declare_schema`` (inner) to check
   structure *and* units together — units converts the input first, schema
   validates the converted array; on the way out schema validates before units
   stamps.

The two markers/unit string coexist in one ``Annotated``: ``annotated_schema``
skips bare strings and ``annotated_unit`` reads the first ``str``, so each reader
ignores the other's metadata. This is test-only; no source changes are involved.
"""

from typing import Annotated

import numpy as np
import pint
import pytest
import xarray as xr

from xarray_annotated import schema, units
from xarray_annotated.schema import Coords, Dims, Dtype


def _da(dims=("time", "x"), dtype="float64", coords=(), unit=None, fill=0.0):
    """Build a DataArray with dims/dtype/coords and an optional units attr.

    ``fill`` sets the constant array value so multiplicative unit conversions are
    observable (zeros would hide a ``hPa`` -> ``Pa`` conversion).
    """
    shape = tuple(range(2, 2 + len(dims)))
    arr = np.full(shape, fill, dtype=dtype)
    coord_map = {d: np.arange(s) for d, s in zip(dims, shape) if d in coords}
    da = xr.DataArray(arr, dims=dims, coords=coord_map)
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestComposeSchemaMarkers:
    """Multiple structural markers on one hint, one ``@declare_schema``."""

    def test_dims_and_coords_pass(self):
        @schema.declare_schema
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time")],
        ) -> xr.DataArray:
            return x

        da = _da(("time", "x"), coords=("time",))
        assert f(da) is da  # never mutates

    def test_dims_ok_but_coord_missing_fails(self):
        @schema.declare_schema
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time")],
        ) -> xr.DataArray:
            return x

        with pytest.raises(schema.SchemaError, match="missing coords"):
            f(_da(("time", "x")))  # dims satisfied, coordinate label absent

    def test_dims_coords_dtype_each_violation_fails(self):
        @schema.declare_schema
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
    """Stacked ``@declare_units`` (outer) + ``@declare_schema`` (inner)."""

    def test_dims_and_units_converts_and_validates(self):
        @units.declare_units
        @schema.declare_schema
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x"), "Pa"],
        ) -> Annotated[xr.DataArray, Dims("time", "x"), "Pa"]:
            return x

        out = f(_da(("time", "x"), unit="hPa", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 1000.0))

    def test_dims_mismatch_raises_with_valid_units(self):
        @units.declare_units
        @schema.declare_schema
        def f(x: Annotated[xr.DataArray, Dims("time", "x"), "Pa"]) -> xr.DataArray:
            return x

        # units convert cleanly (Pa -> Pa), then inner schema rejects the dims
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(_da(("time",), unit="Pa", fill=1.0))

    def test_units_dimensional_mismatch_raises_with_valid_dims(self):
        @units.declare_units
        @schema.declare_schema
        def f(x: Annotated[xr.DataArray, Dims("time", "x"), "Pa"]) -> xr.DataArray:
            return x

        # outer units runs first: kg vs Pa raises before schema sees the array
        with pytest.raises(pint.DimensionalityError):
            f(_da(("time", "x"), unit="kg", fill=1.0))

    def test_full_composition_dims_coords_dtype_units(self):
        @units.declare_units
        @schema.declare_schema
        def f(
            x: Annotated[
                xr.DataArray,
                Dims("time", "x"),
                Coords("time"),
                Dtype("float64"),
                "Pa",
            ],
        ) -> Annotated[xr.DataArray, Dims("time", "x"), "Pa"]:
            return x

        out = f(_da(("time", "x"), coords=("time",), unit="hPa", fill=10.0))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, np.full(out.shape, 1000.0))

    def test_output_stamped_and_validated(self):
        @units.declare_units
        @schema.declare_schema
        def f(n: int) -> Annotated[xr.DataArray, Dims("time", "x"), "Pa"]:
            return _da(("time", "x"), fill=5.0)  # raw output, no units attr

        out = f(1)
        assert out.attrs["units"] == "Pa"  # stamped by units
        np.testing.assert_allclose(out.values, np.full(out.shape, 5.0))  # not converted

    def test_output_dims_mismatch_raises(self):
        @units.declare_units
        @schema.declare_schema
        def f(n: int) -> Annotated[xr.DataArray, Dims("time", "x"), "Pa"]:
            return _da(("time",), fill=5.0)

        # inner schema validates the output before outer units stamps it
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(1)
