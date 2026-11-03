"""Tests specific to ``enforce``: facet ordering, prefixed keywords, the switch.

Per-domain behaviour under ``enforce`` is covered in each domain's test module;
this one pins what only the unified decorator can get right — that units are
applied *before* structure and frequency on both sides of the call — and the
surface that is new with it.
"""

import asyncio
import inspect
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
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True), Dtype("float64")]):
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

    def test_frozen_dataclass_field_is_converted_too(self):
        @dataclass(frozen=True)
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("time")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da().pint.quantify())

        assert units.units_equal(f"{f().p.pint.units:~P}", "Pa")


class TestKeywords:
    @pytest.mark.parametrize(
        "kwarg",
        [
            "units_on_output",
            "temporal_on_uninferable",
        ],
    )
    def test_bad_value_rejected_at_decoration(self, kwarg):
        with pytest.raises(ValueError, match=f"^Invalid {kwarg} 'bogus'"):
            xra.enforce(**{kwarg: "bogus"})  # type: ignore[arg-type]

    def test_missing_unit_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        with pytest.raises(ValueError, match="no 'units' attribute"):
            f(_int_da(unit=None))

    @pytest.mark.parametrize("kwarg", ["schema_on_mismatch", "temporal_on_mismatch"])
    def test_mismatch_severity_has_no_keyword(self, kwarg):
        # Loosening a check is stated on the marker it applies to.
        with pytest.raises(TypeError, match=kwarg):
            xra.enforce(**{kwarg: "warn"})  # type: ignore[call-arg]

    def test_temporal_on_uninferable_keyword(self):
        @xra.enforce(temporal_on_uninferable="error")
        def f(p: Annotated[xr.DataArray, Freq("7D")]): ...

        with pytest.raises(temporal.FreqError):
            f(_int_da().isel(time=slice(0, 2)))  # too short to infer

    def test_marker_overrides_are_independent_per_domain(self):
        # A schema marker's on_mismatch must not leak into the temporal domain.
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Dims("x", on_mismatch="ignore"), Freq("7D")],
        ): ...

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
        with xra.enabled(False):
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
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)], n: int = 3) -> int:
            return n

        assert f(_int_da(), n=5) == 5

    def test_dimensional_mismatch_always_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]): ...

        with pytest.raises(pint.DimensionalityError):
            f(_int_da(unit="kg"))

    def test_declared_output_field_that_is_not_a_dataarray_passes_through(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray | None, Unit("Pa"), Dims("x")]

        @xra.enforce
        def f() -> Out:
            return {"p": None}

        assert f() == {"p": None}


class TestAsync:
    def test_output_is_applied_to_the_awaited_result(self):
        @xra.enforce
        async def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)],
        ) -> Annotated[xr.DataArray, Unit("kPa"), Dims("time")]:
            return p / 1000.0

        out = asyncio.run(f(_int_da().astype(float)))
        assert out.attrs["units"] == "kPa"

    def test_output_checks_run(self):
        @xra.enforce
        async def f() -> Annotated[xr.DataArray, Dims("nope")]:
            return _int_da()

        with pytest.raises(schema.SchemaError):
            asyncio.run(f())

    def test_require_sees_the_awaited_result(self):
        @xra.enforce
        async def f() -> Annotated[xr.DataArray, Unit("Pa", on_output="require")]:
            return _int_da("Pa")

        assert asyncio.run(f()).attrs["units"] == "Pa"

    def test_input_checks_run(self):
        @xra.enforce
        async def f(p: Annotated[xr.DataArray, Unit("kg")]) -> None: ...

        with pytest.raises(pint.DimensionalityError):
            asyncio.run(f(_int_da("Pa")))

    def test_disabled_is_a_noop(self):
        @xra.enforce
        async def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _int_da(None)

        with xra.enabled(False):
            assert "units" not in asyncio.run(f()).attrs

    def test_stays_a_coroutine_function(self):
        @xra.enforce
        async def f() -> None: ...

        assert inspect.iscoroutinefunction(f)


class TestOutputStampLeavesTheInputAlone:
    def _passthrough(self):
        @xra.enforce
        def f(x: xr.DataArray) -> Annotated[xr.DataArray, Unit("Pa")]:
            return x

        return f

    def test_caller_array_keeps_its_label(self):
        da = _int_da("hPa")
        out = self._passthrough()(da)
        assert out.attrs["units"] == "Pa"
        assert da.attrs["units"] == "hPa"

    def test_data_is_not_copied(self):
        da = xr.DataArray(np.zeros(3), dims="x", attrs={"units": "hPa"})
        assert np.shares_memory(self._passthrough()(da).values, da.values)

    def test_later_check_failure_leaves_it_alone(self):
        @xra.enforce
        def f(x: xr.DataArray) -> Annotated[xr.DataArray, Unit("Pa"), Dims("nope")]:
            return x

        da = _int_da("hPa")
        with pytest.raises(schema.SchemaError):
            f(da)
        assert da.attrs["units"] == "hPa"

    def test_unlabelled_array_stays_unlabelled(self):
        da = xr.DataArray([1.0])
        assert self._passthrough()(da).attrs["units"] == "Pa"
        assert "units" not in da.attrs

    @pytest.mark.parametrize("frozen", [True, False])
    def test_dataclass_field(self, frozen):
        @dataclass(frozen=frozen)
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f(x: xr.DataArray) -> Out:
            return Out(p=x)

        da = _int_da("hPa")
        assert f(da).p.attrs["units"] == "Pa"
        assert da.attrs["units"] == "hPa"

    def test_frozen_dataclass_takes_a_conversion(self):
        @dataclass(frozen=True)
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da("hPa").astype(float).pint.quantify())

        out = f().p
        assert units.units_equal(f"{out.pint.units:~P}", "Pa")
        np.testing.assert_allclose(out.pint.magnitude, _int_da().values * 100.0)

    def test_dict_field(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f(x: xr.DataArray) -> Out:
            return {"p": x}

        da = _int_da("hPa")
        assert f(da)["p"].attrs["units"] == "Pa"
        assert da.attrs["units"] == "hPa"
