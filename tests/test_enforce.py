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
from dataclasses import InitVar, dataclass, field
from typing import (
    Annotated,
    Concatenate,
    NamedTuple,
    NotRequired,
    ParamSpec,
    TypedDict,
)

import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Declared, schema, units
from xarray_annotated.schema import Coords, Dims, Dtype, SchemaError
from xarray_annotated.temporal import Freq
from xarray_annotated.units import Unit, UnitError


def _int_da(unit: str | None = "hPa") -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    da = xr.DataArray(np.arange(4, dtype="int64"), dims="time", coords={"time": times})
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestOrdering:
    def test_input_units_checked_before_schema_check(self):
        # Both are wrong, and the unit is reported.
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa"), Dtype("float64")]):
            return p

        with pytest.raises(UnitError, match="'hPa' differs"):
            f(_int_da())

    def test_output_units_checked_before_schema_check(self):
        # Both are wrong, and the unit is reported.
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa"), Dtype("float64")]:
            return _int_da()

        with pytest.raises(UnitError, match="'hPa' differs"):
            f()

    def test_output_schema_mismatch_reported_after_units(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa"), Dims("x")]:
            return _int_da("Pa")

        with pytest.raises(schema.SchemaError, match="'return'"):
            f()

    def test_all_facets_on_typeddict_return(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]

        @xra.enforce
        def f(x: xr.DataArray) -> Out:
            return {"p": x}

        out = f(_int_da("pascal"))
        assert out["p"].attrs["units"] == "pascal"

    def test_all_facets_on_dataclass_return_checks_quantified_field(self):
        @dataclass
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa"), Dtype("float64"), Freq("D")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da("Pa").astype(float).pint.quantify())

        out = f()
        assert out.p.dtype == np.float64
        assert units.units_equal(f"{out.p.pint.units:~P}", "Pa")

    def test_frozen_dataclass_quantified_field_is_checked_too(self):
        @dataclass(frozen=True)
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa"), Dims("time")]

        @xra.enforce
        def f() -> Out:
            return Out(p=_int_da("Pa").pint.quantify())

        assert units.units_equal(f"{f().p.pint.units:~P}", "Pa")


class TestNoOptions:
    def test_missing_unit_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        with pytest.raises(UnitError, match="no 'units' attribute"):
            f(_int_da(unit=None))

    @pytest.mark.parametrize("kwarg", ["temporal_on_uninferable", "units_on_output"])
    def test_enforce_has_no_keywords(self, kwarg):
        with pytest.raises(TypeError, match=kwarg):
            xra.enforce(**{kwarg: "warn"})  # type: ignore[call-arg]


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
        def f(p: Annotated[xr.DataArray, Unit("Pa")], n: int = 3) -> int:
            return n

        assert f(_int_da("Pa"), n=5) == 5

    def test_dimensional_mismatch_always_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]): ...

        with pytest.raises(UnitError):
            f(_int_da(unit="kg"))

    def test_optional_output_field_of_none_passes(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray | None, Unit("Pa"), Dims("x")]

        @xra.enforce
        def f() -> Out:
            return {"p": None}

        assert f() == {"p": None}


class TestAsync:
    def test_output_is_checked_on_the_awaited_result(self):
        @xra.enforce
        async def f(
            p: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("kPa"), Dims("time")]:
            return (p / 1000.0).assign_attrs(units="kilopascal")

        out = asyncio.run(f(_int_da("Pa").astype(float)))
        assert out.attrs["units"] == "kilopascal"

    def test_output_checks_run(self):
        @xra.enforce
        async def f() -> Annotated[xr.DataArray, Dims("nope")]:
            return _int_da()

        with pytest.raises(schema.SchemaError):
            asyncio.run(f())

    def test_output_unit_is_checked_on_the_awaited_result(self):
        @xra.enforce
        async def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _int_da("hPa")

        with pytest.raises(UnitError, match="'hPa' differs"):
            asyncio.run(f())

    def test_input_checks_run(self):
        @xra.enforce
        async def f(p: Annotated[xr.DataArray, Unit("kg")]) -> None: ...

        with pytest.raises(UnitError):
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


class TestChecksChangeNothing:
    """The body gets what the caller passed, the caller what the body returned."""

    def _passthrough(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return x

        return f

    @pytest.mark.parametrize(
        "make",
        [
            lambda: _int_da("pascal"),
            lambda: _int_da("Pa"),
            lambda: xr.DataArray([1.0], attrs={"units": "pascal"}).pint.quantify(),
        ],
        ids=["respelled", "same", "quantified"],
    )
    def test_same_object_in_and_out(self, make):
        da = make()
        assert self._passthrough()(da) is da

    def test_the_body_sees_a_quantified_input(self):
        seen = {}

        @xra.enforce
        def f(x: Annotated[xr.DataArray, Unit("Pa")]) -> None:
            seen["x"] = x

        da = xr.DataArray([1.0], attrs={"units": "Pa"}).pint.quantify()
        f(da)
        assert seen["x"] is da

    def test_the_label_keeps_its_spelling(self):
        da = _int_da("pascal")
        assert self._passthrough()(da).attrs["units"] == "pascal"

    def test_encoding_is_left_alone(self):
        da = xr.DataArray([1.0], attrs={"units": "Pa"}).pint.quantify()
        da.encoding = {"dtype": "int16"}
        assert self._passthrough()(da).encoding == {"dtype": "int16"}

    def test_same_result_when_switched_off(self):
        @xra.enforce
        def dry(rh: Annotated[xr.DataArray, Unit("percent")]) -> xr.DataArray:
            return 1 - rh / 100

        rh = xr.DataArray([60.0], attrs={"units": "percent"}).pint.quantify()
        on = dry(rh)
        with xra.enabled(False):
            off = dry(rh)
        xr.testing.assert_identical(on, off)

    def test_two_spellings_in_a_plain_body_lose_the_label(self):
        # xarray drops conflicting attrs, so the output has no label to check.
        @xra.enforce
        def add(
            a: Annotated[xr.DataArray, Unit("Pa")],
            b: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return a + b

        with pytest.raises(UnitError, match="no 'units' attribute"):
            add(_int_da("pascal"), _int_da("Pa"))


class TestValidatesOnce:
    def test_nothing_is_validated_when_called(self, monkeypatch):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        def fail(*_):
            raise AssertionError("validated again")

        monkeypatch.setattr("xarray_annotated._contract.assert_valid_unit", fail)
        monkeypatch.setattr("xarray_annotated.units._check.assert_valid_unit", fail)
        f(_int_da("Pa"))

    def test_nothing_is_validated_when_called_on_any_facet(self, monkeypatch):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Dtype("int"), Freq("D")],
        ) -> Annotated[xr.DataArray, Dtype("int"), Freq("D")]:
            return p

        def fail(*_):
            raise AssertionError("validated again")

        for module in ("_contract", "schema._check"):
            monkeypatch.setattr(f"xarray_annotated.{module}.assert_valid_schema", fail)
        for module in ("_contract", "temporal._check"):
            monkeypatch.setattr(f"xarray_annotated.{module}.assert_valid_freq", fail)
        time = pd.date_range("2000-01-01", periods=3, freq="D")
        f(xr.DataArray(np.arange(3), dims="time", coords={"time": time}))


class TestNothingDeclared:
    def test_returns_the_function_itself(self):
        def f(x: xr.DataArray, y: Annotated[int, "a description"]) -> xr.DataArray:
            return x

        assert xra.enforce(f) is f

    def test_returns_a_coroutine_function_itself(self):
        async def f(x: xr.DataArray) -> None:
            pass

        assert xra.enforce(f) is f

    def test_returns_the_function_for_a_record_with_no_declared_fields(self):
        class Plain(TypedDict):
            x: xr.DataArray

        def f() -> Plain:
            return {"x": xr.DataArray([1.0])}

        assert xra.enforce(f) is f

    def test_a_stray_marker_still_raises(self):
        def f(x: tuple[Annotated[xr.DataArray, Unit("Pa")], int]) -> None:
            pass

        with pytest.raises(ValueError, match="enforce cannot apply"):
            xra.enforce(f)

    @pytest.mark.parametrize("side", ["param", "ret"])
    def test_a_declaration_on_either_side_is_wrapped(self, side):
        func = _annotated(**{side: Annotated[xr.DataArray, Unit("Pa")]})
        assert xra.enforce(func) is not func


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


_P = ParamSpec("_P")
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
            Callable[[_PA], int],
            Callable[[int, _PA], int],
            list[Callable[[_PA], int]],
            Callable[Concatenate[_PA, _P], int],  # pyright: ignore[reportGeneralTypeIssues]
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
        with pytest.raises(UnitError, match="differs from declared"):
            f(xr.DataArray([1.0], attrs={"units": "hPa"}))

    def test_return_is_checked(self):
        f = xra.enforce(_annotated(ret=_Alias))
        with pytest.raises(UnitError, match="'hPa' differs"):
            f(xr.DataArray([1.0], attrs={"units": "hPa"}))

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
        with pytest.raises(UnitError, match="differs from declared"):
            f(xr.DataArray([1.0], dims="x", attrs={"units": "hPa"}))
        with pytest.raises(SchemaError):
            f(xr.DataArray([1.0], dims="y", attrs={"units": "Pa"}))

    def test_annotate_extends_an_alias(self):
        assert xra.declaration_from_hint(xra.annotate(_Alias, dims="x")) == Declared(
            Unit("Pa"), Dims("x")
        )

    def test_aliased_typeddict_return_is_read_per_field(self):
        f = xra.enforce(_annotated(ret=_FieldsAlias))
        with pytest.raises(UnitError, match="'hPa' differs"):
            f({"pa": xr.DataArray([1.0], attrs={"units": "hPa"})})


@dataclass
class _Record:
    pa: _PA


@dataclass
class _Record2:
    pa: _PA


@dataclass
class _GenericRecord[T]:
    pa: _PA
    other: T


class _Tuple(NamedTuple):
    pa: _PA


@dataclass
class _Nested:
    inner: _Record


@dataclass
class _OptionalNested:
    inner: _Record | None


@dataclass
class _ListNested:
    inners: list[_Record]


class _NestedInTypedDict(TypedDict):
    inner: NotRequired[_Record]


class _Plain(TypedDict):
    x: int


@dataclass
class _HoldsPlain:
    pa: _PA
    config: _Plain


@dataclass
class _SelfReferring:
    pa: _PA
    next: "_SelfReferring | None" = None


@dataclass
class _Unresolvable:
    x: "NotDefinedAnywhere"  # type: ignore[name-defined]  # noqa: F821


type _OptionalRecord = _Record | None
type _GenericAlias[T] = _GenericRecord[T] | None


def _kg() -> xr.DataArray:
    return xr.DataArray([1.0], attrs={"units": "kg"})


class TestReturnContainers:
    """A declared return container is read, or refused, but never left unchecked."""

    @pytest.mark.parametrize(
        ("hint", "value"),
        [
            (_Record, lambda: _Record(_kg())),
            (_Record | None, lambda: _Record(_kg())),
            (_OptionalRecord, lambda: _Record(_kg())),
            (_Fields | None, lambda: {"pa": _kg()}),
            (_FieldsAlias, lambda: {"pa": _kg()}),
            (Annotated[_Record, "described"], lambda: _Record(_kg())),
            (_GenericRecord[int], lambda: _GenericRecord(_kg(), 1)),
            (_GenericRecord[int] | None, lambda: _GenericRecord(_kg(), 1)),
            (_GenericAlias[int], lambda: _GenericRecord(_kg(), 1)),
        ],
    )
    def test_read_field_by_field(self, hint, value):
        f = xra.enforce(_annotated(ret=hint))
        with pytest.raises(UnitError):
            f(value())

    @pytest.mark.parametrize(
        "hint",
        [
            _Tuple,
            _Tuple | None,
            list[_Record],
            tuple[_Record, _Record],
            _Record | _Record2,
            _Record | _Fields,
            _PA | _Record,
            list[_Nested],
            list[_SelfReferring],
            dict[str, _OptionalRecord],
            _Nested,
            _Nested | None,
            _OptionalNested,
            _ListNested,
            _SelfReferring,
            _NestedInTypedDict,
        ],
    )
    def test_refused_at_decoration(self, hint):
        with pytest.raises(ValueError, match=r"enforce\s+cannot apply"):
            xra.enforce(_annotated(ret=hint))

    @pytest.mark.parametrize(
        "hint",
        [
            Annotated[_Record, Unit("Pa")],
            _GenericRecord[_PA],
        ],
    )
    def test_marker_on_the_container_itself_is_refused(self, hint):
        with pytest.raises(ValueError, match="enforce cannot apply"):
            xra.enforce(_annotated(ret=hint))

    @pytest.mark.parametrize(
        "hint",
        [
            list[_Plain],
            list[_Unresolvable],
            _HoldsPlain,
            _Plain | int,
            _PA,
            _PA | None,
        ],
    )
    def test_nothing_unread_is_not_refused(self, hint):
        xra.enforce(_annotated(ret=hint))

    def test_optional_container_may_return_none(self):
        f = xra.enforce(_annotated(ret=_Record | None))
        assert f(None) is None

    def test_reader_reports_the_fields(self):
        expected = {"pa": Declared(unit=Unit("Pa"))}
        for hint in (_Record | None, _OptionalRecord, _GenericRecord[int]):
            _, output = xra.declarations_from_signature(_annotated(ret=hint))
            assert output == expected

    def test_reader_returns_nothing_for_what_enforce_refuses(self):
        for hint in (_Tuple, list[_Record], _Record | _Record2):
            _, output = xra.declarations_from_signature(_annotated(ret=hint))
            assert output is None


class _Respelled(TypedDict):
    a: _PA
    b: _PA


@dataclass
class _RespelledRecord:
    a: _PA
    b: _PA


def _pascal() -> xr.DataArray:
    return xr.DataArray([1.0], attrs={"units": "pascal"})


def _pa() -> xr.DataArray:
    return xr.DataArray([1.0], attrs={"units": "Pa"})


class TestRecordReturnsAreNotTouched:
    """A record return comes back as the same object, once every field has passed."""

    def _identity(self, hint):
        return xra.enforce(_annotated(ret=hint))

    def test_failed_dict_check_raises(self):
        with pytest.raises(UnitError):
            self._identity(_Respelled)({"a": _pascal(), "b": _kg()})

    def test_failed_dataclass_check_raises(self):
        with pytest.raises(UnitError):
            self._identity(_RespelledRecord)(_RespelledRecord(_pascal(), _kg()))

    def test_dict_comes_back_as_itself(self):
        d = {"a": _pascal(), "b": _pa(), "extra": 1}
        before = d["a"]
        assert self._identity(_Respelled)(d) is d
        assert d["a"] is before
        assert d["a"].attrs["units"] == "pascal"

    def test_dataclass_comes_back_as_itself(self):
        r = _RespelledRecord(_pascal(), _pa())
        assert self._identity(_RespelledRecord)(r) is r

    def test_undeclared_value_comes_back_as_itself(self):
        @dataclass
        class Plain:
            n: Annotated[xr.DataArray | int, Unit("Pa")]

        r = Plain(1)
        assert self._identity(Plain)(r) is r

    @pytest.mark.parametrize(
        "options", [{"frozen": True}, {"frozen": True, "slots": True}], ids=str
    )
    def test_frozen_dataclass(self, options):
        @dataclass(**options)
        class Out:
            a: _PA

        r = Out(_pascal())
        assert self._identity(Out)(r) is r

    def test_frozen_dataclass_with_an_initvar(self):
        @dataclass(frozen=True)
        class Out:
            a: _PA
            scale: InitVar[float]

        r = Out(_pascal(), 2.0)
        assert self._identity(Out)(r) is r

    def test_frozen_dataclass_with_an_init_false_field(self):
        @dataclass(frozen=True)
        class Out:
            a: _PA = field(init=False)

            def __post_init__(self):
                object.__setattr__(self, "a", _pascal())

        r = Out()
        assert self._identity(Out)(r) is r

    def test_post_init_does_not_run_again(self):
        calls = []

        @dataclass(frozen=True)
        class Out:
            a: _PA

            def __post_init__(self):
                calls.append(1)

        self._identity(Out)(Out(_pascal()))
        assert len(calls) == 1


class TestOtherValues:
    """A declared value is a `DataArray`, or whatever else its hint allows."""

    @pytest.mark.parametrize(
        "value",
        [None, np.array([1.0]), xr.Dataset(), 1.0],
        ids=lambda v: type(v).__name__,
    )
    def test_input_raises(self, value):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Dims("x")]): ...

        with pytest.raises(TypeError, match="input 'p': declared a DataArray"):
            f(value)

    def test_none_raises_at_a_default_the_hint_does_not_allow(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Dims("x")] = None): ...  # type: ignore[assignment]

        with pytest.raises(TypeError, match=r"got None.*\| None"):
            f()

    @pytest.mark.parametrize(
        "hint",
        [
            Annotated[xr.DataArray | None, Dims("x")],
            Annotated[xr.DataArray, Dims("x")] | None,  # pyright: ignore[reportOperatorIssue]
        ],
    )
    def test_optional_allows_none_only(self, hint):
        @xra.enforce
        def f(p: hint): ...  # pyright: ignore[reportInvalidTypeForm]

        f(None)
        with pytest.raises(TypeError, match="got float"):
            f(1.0)

    @pytest.mark.parametrize(
        "hint",
        [
            Annotated[xr.DataArray | float, Dims("x")],
            Annotated[xr.DataArray, Dims("x")] | float,
        ],
    )
    def test_another_type_passes_but_does_not_allow_none(self, hint):
        @xra.enforce
        def f(p: hint): ...  # pyright: ignore[reportInvalidTypeForm]

        f(1.0)
        with pytest.raises(TypeError, match="got None"):
            f(None)

    def test_a_dataarray_under_a_union_is_still_checked(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray | None, Dims("x")]): ...

        with pytest.raises(schema.SchemaError):
            f(xr.DataArray([1.0], dims="y"))

    def test_undeclared_parameters_are_not_looked_at(self):
        @xra.enforce
        def f(p: xr.DataArray, q: Annotated[int, "a description"]): ...

        f(None, None)  # type: ignore[arg-type]

    def test_switched_off_checks_nothing(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Dims("x")]): ...

        with xra.enabled(False):
            f(None)  # type: ignore[arg-type]


class TestRecordParameters:
    """A parameter's fields are never read, so a marker on one is refused."""

    class _Fields(TypedDict):
        p: Annotated[xr.DataArray, Unit("Pa")]

    @dataclass
    class _Data:
        p: Annotated[xr.DataArray, Unit("Pa")]

    @pytest.mark.parametrize("record", [_Fields, _Data], ids=["TypedDict", "dataclass"])
    @pytest.mark.parametrize(
        "wrap",
        [
            lambda r: r,
            lambda r: r | None,
            lambda r: list[r],
            lambda r: Annotated[r, "a description"],
        ],
        ids=["bare", "optional", "list", "described"],
    )
    def test_refused_at_decoration(self, record, wrap):
        hint = wrap(record)

        def f(x: hint): ...  # pyright: ignore[reportInvalidTypeForm]

        with pytest.raises(ValueError, match="Declare a parameter's arrays"):
            xra.enforce(f)

    def test_a_record_with_no_markers_is_fine(self):
        @dataclass
        class Plain:
            p: xr.DataArray

        @xra.enforce
        def f(x: Plain):
            return x

        record = Plain(p=xr.DataArray([1.0]))
        assert f(record) is record

    def test_the_reader_still_reports_nothing(self):
        def f(x: TestRecordParameters._Data): ...

        assert xra.declarations_from_signature(f) == ({}, None)


class TestMalformedMarkers:
    """A malformed declaration raises `ValueError`, naming its site, at decoration."""

    @pytest.mark.parametrize(
        "marker",
        [
            Unit(5),  # type: ignore[arg-type]
            Unit(None),  # type: ignore[arg-type]
            Freq(5),  # type: ignore[arg-type]
            Freq(None),  # type: ignore[arg-type]
            Dims(5),  # type: ignore[arg-type]
            Coords(5),  # type: ignore[arg-type]
            Dtype(5),  # type: ignore[arg-type]
        ],
        ids=repr,
    )
    @pytest.mark.parametrize("side", ["input", "output"])
    def test_a_non_string_argument_raises(self, marker, side):
        hint = Annotated[xr.DataArray, marker]

        def takes(p: hint): ...  # pyright: ignore[reportInvalidTypeForm]

        def returns() -> hint: ...  # pyright: ignore[reportInvalidTypeForm]

        f, match = (takes, "input 'p'") if side == "input" else (returns, "output")
        with pytest.raises(ValueError, match=match):
            xra.enforce(f)
