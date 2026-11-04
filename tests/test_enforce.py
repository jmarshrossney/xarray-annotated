"""Tests specific to ``enforce``: facet ordering, prefixed keywords, the switch.

Per-domain behaviour under ``enforce`` is covered in each domain's test module;
this one pins what only the unified decorator can get right — that units are
applied *before* structure and frequency on both sides of the call — and the
surface that is new with it.
"""

import asyncio
import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, NotRequired, TypedDict

import numpy as np
import pandas as pd
import pint
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Declared, schema, temporal, units
from xarray_annotated.schema import Dims, Dtype, SchemaError
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


class TestVariadics:
    def test_declared_var_positional_is_rejected(self):
        with pytest.raises(ValueError, match=r"'xs'.*variadic"):

            @xra.enforce
            def f(*xs: Annotated[xr.DataArray, Unit("Pa")]) -> None: ...

    def test_declared_var_keyword_is_rejected(self):
        with pytest.raises(ValueError, match=r"'xs'.*variadic"):

            @xra.enforce
            def f(**xs: Annotated[xr.DataArray, Dims("x")]) -> None: ...

    def test_undeclared_variadics_are_fine(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")], *a: int, **k: str) -> int:
            return len(a) + len(k)

        assert f(_int_da("Pa"), 1, 2, z="3") == 3


_PA = Annotated[xr.DataArray, Unit("Pa")]

type _Alias = _PA
type _OptionalAlias = _PA | None
type _StrayAlias = list[_PA]
type _StrayGeneric[T] = Annotated[T, Unit("Pa")]
type _StrayGenericAlias = _StrayGeneric[xr.DataArray]
type _Generic[T] = list[T]
type _Recursive = dict[str, _Recursive] | list[_Recursive] | str
type _RecursiveGeneric[T] = list[_RecursiveGeneric[T]] | T
type _StrayRecursive = list[_StrayRecursive] | list[_PA]
type _DeclaredRecursive = _PA | list[_DeclaredRecursive]
type _AliasOfAlias = _Alias
type _DA = xr.DataArray
type _OptionalDA = xr.DataArray | None
type _Extended = Annotated[_Alias, Dims("x")]
type _FloatAlias = Annotated[float, Unit("Pa")]


class _Fields(TypedDict):
    pa: _PA


type _FieldsAlias = _Fields


class _StrayField(TypedDict):
    pa: list[_PA]


def _annotated(*, param=None, ret=None):
    def f(x=None):
        return x

    f.__annotations__ = {"x": param, "return": ret}
    return f


class TestStrayMarkers:
    """A marker that `enforce` could not apply raises rather than going unchecked."""

    @pytest.mark.parametrize(
        "hint",
        [
            tuple[_PA, _PA],
            list[_PA],
            dict[str, _PA],
            Annotated[xr.Dataset, Unit("Pa")],
            Annotated[float, Unit("Pa")],
            Annotated[int, Unit("Pa")] | None,  # pyright: ignore[reportOperatorIssue]
            Callable[[_PA], _PA],
            _StrayAlias,
            _StrayGeneric[xr.DataArray],
            _StrayGeneric[xr.DataArray] | None,
            _StrayGenericAlias,
            _Generic[_PA],
            _StrayRecursive,
            Annotated[_FloatAlias, Dims("x")],
            Annotated[_StrayGeneric[xr.DataArray], Dims("x")],
            Annotated[_OptionalAlias, Dims("x")],
        ],
    )
    @pytest.mark.parametrize("side", ["param", "ret"])
    def test_raises_at_decoration(self, hint, side):
        with pytest.raises(ValueError, match="enforce cannot apply"):
            xra.enforce(_annotated(**{side: hint}))

    def test_raises_on_a_typeddict_field(self):
        with pytest.raises(ValueError, match="enforce cannot apply") as info:
            xra.enforce(_annotated(ret=_StrayField))
        assert "output 'pa'" in info.value.__notes__[-1]

    def test_raises_when_switched_off(self):
        with (
            xra.enabled(False),
            pytest.raises(ValueError, match="enforce cannot apply"),
        ):
            xra.enforce(_annotated(ret=list[_PA]))

    @pytest.mark.parametrize(
        "hint",
        [
            xr.DataArray,
            int,
            Annotated[xr.DataArray, "Pa"],
            Annotated[int, "Pa"],
            _PA,
            _Generic[int],
            _Recursive,
            _RecursiveGeneric[int],
            _DeclaredRecursive,
        ],
    )
    @pytest.mark.parametrize("side", ["param", "ret"])
    def test_hints_without_stray_markers_pass(self, hint, side):
        xra.enforce(_annotated(**{side: hint}))

    def test_the_readers_still_report_nothing(self):
        def f(x: tuple[_PA, _PA]): ...

        assert xra.declarations_from_signature(f) == ({}, None)


class TestTypeAliases:
    @pytest.mark.parametrize("hint", [_Alias, _OptionalAlias, _Alias | None])
    def test_parameter_is_enforced(self, hint):
        f = xra.enforce(_annotated(param=hint))
        with pytest.raises(ValueError, match="differs from declared"):
            f(xr.DataArray([1.0], attrs={"units": "hPa"}))

    def test_return_is_stamped(self):
        f = xra.enforce(_annotated(ret=_Alias))
        assert f(xr.DataArray([1.0])).attrs["units"] == "Pa"

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (Annotated[_Alias, Dims("x")], Declared(Unit("Pa"), Dims("x"))),
            (Annotated[_AliasOfAlias, Dims("x")], Declared(Unit("Pa"), Dims("x"))),
            (_Extended, Declared(Unit("Pa"), Dims("x"))),
            (
                Annotated[_Extended, Dtype("float64")],
                Declared(Unit("Pa"), Dims("x"), Dtype("float64")),
            ),
            (Annotated[_Alias, "a description"], Declared(Unit("Pa"))),
            (Annotated[_DA, Unit("Pa")], Declared(Unit("Pa"))),
            (Annotated[_OptionalDA, Unit("Pa")], Declared(Unit("Pa"))),
            (
                NotRequired[Annotated[_Alias, Dims("x")]],
                Declared(Unit("Pa"), Dims("x")),
            ),
        ],
    )
    def test_markers_merge_through_an_alias(self, hint, expected):
        assert xra.declaration_from_hint(hint) == expected

    def test_merged_markers_match_an_assignment_alias(self):
        assert xra.declaration_from_hint(
            Annotated[_Alias, Dims("x")]
        ) == xra.declaration_from_hint(Annotated[_PA, Dims("x")])

    @pytest.mark.parametrize(
        "hint", [Annotated[_Alias, Unit("K")], Annotated[_PA, Unit("K")]]
    )
    def test_a_marker_cannot_be_overridden_through_an_alias(self, hint):
        with pytest.raises(ValueError, match="declares 2 Unit"):
            xra.declaration_from_hint(hint)

    def test_merged_markers_are_all_enforced(self):
        f = xra.enforce(_annotated(param=Annotated[_Alias, Dims("x")]))
        f(xr.DataArray([1.0], dims="x", attrs={"units": "Pa"}))
        with pytest.raises(ValueError, match="differs from declared"):
            f(xr.DataArray([1.0], dims="x", attrs={"units": "hPa"}))
        with pytest.raises(SchemaError):
            f(xr.DataArray([1.0], dims="y", attrs={"units": "Pa"}))

    def test_annotate_extends_an_alias(self):
        assert xra.declaration_from_hint(xra.annotate(_Alias, dims="x")) == Declared(
            Unit("Pa"), Dims("x")
        )

    def test_aliased_typeddict_return_is_read_per_field(self):
        f = xra.enforce(_annotated(ret=_FieldsAlias))
        out = f({"pa": xr.DataArray([1.0])})
        assert out["pa"].attrs["units"] == "Pa"
