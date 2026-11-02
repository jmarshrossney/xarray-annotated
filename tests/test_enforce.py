"""Tests specific to ``enforce``: facet ordering, prefixed keywords, the switch.

Per-domain behaviour under ``enforce`` is covered in each domain's test module;
this one pins what only the unified decorator can get right — that units are
applied *before* structure and frequency on both sides of the call — and the
surface that is new with it.
"""

import dataclasses
import typing
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
import pint
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import schema, temporal, units
from xarray_annotated.schema import Dims, Dtype
from xarray_annotated.temporal import Freq
from xarray_annotated.units import Unit


def _int_da(unit: str | None = "hPa") -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    da = xr.DataArray(np.arange(4, dtype="int64"), dims="time", coords={"time": times})
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestOrdering:
    def test_input_units_converted_before_schema_check(self):
        # An int64 hPa array converts to float64 Pa; Dtype must see the result.
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa"), Dtype("float64")]):
            return p

        out = f(_int_da())
        assert out.dtype == np.float64
        np.testing.assert_allclose(out.values, [0.0, 100.0, 200.0, 300.0])

    def test_output_units_converted_before_schema_check(self):
        # A quantified int64 hPa output converts to float64 Pa on the way out;
        # Dtype must see what the caller receives.
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa"), Dtype("float64")]:
            return _int_da().pint.quantify()

        out = f()
        assert out.dtype == np.float64
        assert units.units_equal(f"{out.pint.units:~P}", "Pa")

    def test_output_schema_mismatch_reported_after_units(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa"), Dims("x")]:
            return _int_da()

        with pytest.raises(schema.SchemaError, match="'return'"):
            f()

    def test_all_facets_on_typeddict_return(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]

        @xra.enforce
        def f(x: xr.DataArray) -> Out:
            return {"p": x}

        out = f(_int_da(unit=None))
        assert out["p"].attrs["units"] == "Pa"

    def test_all_facets_on_dataclass_return_converts_quantified_field(self):
        @dataclass
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa"), Dtype("float64"), Freq("D")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da().pint.quantify())

        out = f()
        assert out.p.dtype == np.float64
        assert units.units_equal(f"{out.p.pint.units:~P}", "Pa")

    def test_frozen_dataclass_conversion_still_explained(self):
        @dataclass(frozen=True)
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("time")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da().pint.quantify())

        with pytest.raises(dataclasses.FrozenInstanceError, match="frozen"):
            f()


class TestKeywords:
    @pytest.mark.parametrize(
        "kwarg",
        [
            "units_on_missing",
            "units_on_inexact",
            "units_on_output",
            "schema_on_mismatch",
            "temporal_on_mismatch",
            "temporal_on_uninferable",
        ],
    )
    def test_bad_value_rejected_at_decoration(self, kwarg):
        axis = kwarg.split("_", 1)[1]
        with pytest.raises(ValueError, match=f"Invalid {axis} 'bogus'"):
            xra.enforce(**{kwarg: "bogus"})  # type: ignore[arg-type]

    def test_units_on_missing_overrides_policy(self):
        @xra.enforce(units_on_missing="error")
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        with (
            units.policy(on_missing="ignore"),
            pytest.raises(ValueError, match="no 'units' attribute"),
        ):
            f(_int_da(unit=None))

    def test_schema_on_mismatch_overrides_policy(self):
        @xra.enforce(schema_on_mismatch="warn")
        def f(p: Annotated[xr.DataArray, Dims("x")]): ...

        with schema.policy(on_mismatch="error"), pytest.warns(schema.SchemaWarning):
            f(_int_da())

    def test_marker_override_beats_keyword(self):
        @xra.enforce(schema_on_mismatch="error")
        def f(p: Annotated[xr.DataArray, Dims("x", on_mismatch="ignore")]): ...

        f(_int_da())  # marker says ignore

    def test_temporal_keywords_override_policy(self):
        @xra.enforce(temporal_on_mismatch="warn", temporal_on_uninferable="error")
        def f(p: Annotated[xr.DataArray, Freq("7D")]): ...

        with pytest.warns(temporal.FreqWarning):
            f(_int_da())
        with pytest.raises(temporal.FreqError):
            f(_int_da().isel(time=slice(0, 2)))  # too short to infer

    def test_keywords_are_independent_per_domain(self):
        # schema_on_mismatch must not leak into the temporal domain.
        @xra.enforce(schema_on_mismatch="ignore")
        def f(p: Annotated[xr.DataArray, Dims("x"), Freq("7D")]): ...

        with pytest.raises(temporal.FreqError):
            f(_int_da())


class TestDecorationTime:
    def test_bad_unit_raises(self):
        with pytest.raises(ValueError, match=r"input 'p'.*not a recognised unit"):

            @xra.enforce
            def f(p: Annotated[xr.DataArray, Unit("not_a_unit")]): ...

    def test_bad_freq_raises(self):
        with pytest.raises(ValueError, match="output: invalid frequency"):

            @xra.enforce
            def f() -> Annotated[xr.DataArray, Freq("not_an_offset")]: ...

    def test_bad_output_field_raises(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("not_a_unit")]

        with pytest.raises(ValueError, match=r"output 'p'.*not a recognised unit"):

            @xra.enforce
            def f() -> Out: ...


class TestWrapper:
    def test_disabled_is_total_noop(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("x"), Freq("7D")],
        ) -> Annotated[xr.DataArray, Unit("kg"), Dims("y")]:
            return p

        da = _int_da()
        with units.policy(enabled=False):
            out = f(da)
        assert out is da
        assert out.attrs["units"] == "hPa"

    def test_annotations_preserved(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa"), Dims("time")]) -> None: ...

        hint = typing.get_type_hints(f, include_extras=True)["p"]
        assert typing.get_args(hint)[1:] == (Unit("Pa"), Dims("time"))

    def test_non_dataarray_arguments_untouched(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")], n: int = 3) -> int:
            return n

        assert f(_int_da(), n=5) == 5

    def test_dimensional_mismatch_always_raises(self):
        @xra.enforce(units_on_inexact="warn", units_on_missing="ignore")
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        with pytest.raises(pint.DimensionalityError):
            f(_int_da(unit="kg"))

    def test_declared_output_field_that_is_not_a_dataarray_passes_through(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray | None, Unit("Pa"), Dims("x")]

        @xra.enforce
        def f() -> Out:
            return {"p": None}

        assert f() == {"p": None}
