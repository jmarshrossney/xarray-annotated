"""Tests for the unified ``declarations_from_signature`` reader.

The all-facets reader is the read-side counterpart to ``annotate``: where the
writer turns facet values into an ``Annotated`` hint, this turns a whole signature
back into one homogeneous ``Declared`` value per declared DataArray.  The tests
below pin the uniform shape (one marker, or ``None``, per facet slot), the
inherited ``TypedDict`` / dataclass / single-return handling, and the exact
round-trip with ``annotate`` that is the reader's whole point.
"""

import sys
import typing
from dataclasses import dataclass
from typing import Annotated, TypedDict

import pint
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import (
    Declared,
    annotate,
    declaration_from_hint,
    declarations_from_signature,
)
from xarray_annotated.schema import Coords, Dims, Dtype
from xarray_annotated.temporal import Freq
from xarray_annotated.units import (
    Unit,
    units_compatible,
    units_equal,
)


def _return_hint(hint):
    """A zero-arg function whose *return* annotation is ``hint``."""

    def fn(): ...

    fn.__annotations__["return"] = hint
    return fn


class TestInputs:
    def test_each_facet_and_combination(self):
        def node(
            u: Annotated[xr.DataArray, Unit("Pa")],
            s: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")],
            both: Annotated[xr.DataArray, Unit("degC"), Coords("time")],
            plain: xr.DataArray,
            flag: bool,
        ) -> None: ...

        inputs, output = declarations_from_signature(node)
        assert output is None
        # Only the declaration-bearing params contribute; `plain`/`flag` drop out.
        assert set(inputs) == {"u", "s", "both"}
        assert inputs["u"] == Declared(unit=Unit("Pa"))
        assert inputs["s"] == Declared(dims=Dims("time", "x"), dtype=Dtype("float64"))
        assert inputs["both"] == Declared(unit=Unit("degC"), coords=Coords("time"))

    def test_freq_only_declaration(self):
        def node(a: Annotated[xr.DataArray, Freq("7D")]) -> None: ...

        inputs, _ = declarations_from_signature(node)
        assert inputs["a"] == Declared(freq=Freq("7D"))

    def test_freq_alongside_other_facets(self):
        def node(
            a: Annotated[xr.DataArray, Unit("degC"), Dims("time"), Freq("W-SUN")],
        ): ...

        inputs, _ = declarations_from_signature(node)
        assert inputs["a"] == Declared(
            unit=Unit("degC"), dims=Dims("time"), freq=Freq("W-SUN")
        )

    def test_bare_string_declares_nothing(self):
        # A string is a description, whatever it happens to spell.
        def node(
            a: Annotated[xr.DataArray, "Pa"], b: Annotated[xr.DataArray, "7D"]
        ) -> None: ...

        inputs, _ = declarations_from_signature(node)
        assert inputs == {}


class TestSingleHint:
    def test_reads_every_facet(self):
        hint = Annotated[
            xr.DataArray,
            Unit("Pa"),
            Dims("time"),
            Dtype("f8"),
            Coords("time"),
            Freq("D"),
        ]
        assert declaration_from_hint(hint) == Declared(
            unit=Unit("Pa"),
            dims=Dims("time"),
            dtype=Dtype("f8"),
            coords=Coords("time"),
            freq=Freq("D"),
        )

    @pytest.mark.parametrize(
        ("hint", "others"),
        [
            (Annotated[xr.DataArray, Unit("Pa")], ()),
            (Annotated[xr.DataArray | None, Unit("Pa")], (type(None),)),
            (Annotated[xr.DataArray, Unit("Pa")] | None, (type(None),)),  # pyright: ignore[reportOperatorIssue]
            (Annotated[xr.DataArray | float, Unit("Pa")], (float,)),
            (Annotated[xr.DataArray | None, Unit("Pa")] | int, (type(None), int)),
        ],
    )
    def test_others_records_what_else_a_hint_allows(self, hint, others):
        declared = declaration_from_hint(hint)
        assert declared is not None
        assert declared.others == others
        assert declared == Declared(unit=Unit("Pa"))  # not a marker, so not compared

    def test_optional_dataarray_is_read(self):
        hint = Annotated[xr.DataArray | None, Unit("Pa"), Freq("D")]
        assert declaration_from_hint(hint) == Declared(unit=Unit("Pa"), freq=Freq("D"))

    @pytest.mark.parametrize(
        "hint",
        [
            xr.DataArray,
            None,
            Annotated[bool, Unit("Pa")],
            Annotated[bool, "toggles X"],
            Annotated[xr.DataArray, "Pa"],
            Annotated[xr.DataArray, 42],
        ],
    )
    def test_declares_nothing(self, hint):
        assert declaration_from_hint(hint) is None

    def test_drives_walk_signature(self):
        def node(a: Annotated[xr.DataArray, Unit("Pa")]) -> None: ...

        assert xra.walk_signature(node, declaration_from_hint) == (
            declarations_from_signature(node)
        )


class TestSchemaProperty:
    def test_collects_the_schema_slots(self):
        declared = Declared(
            unit=Unit("Pa"), dims=Dims("time"), coords=Coords("time"), freq=Freq("D")
        )
        assert declared.schema == [Dims("time"), Coords("time")]

    def test_empty_when_no_schema_facet(self):
        assert Declared(unit=Unit("Pa")).schema == []


class TestOutputShapes:
    def test_undeclared_typeddict_field_is_omitted(self):
        class Out(TypedDict):
            a: Annotated[xr.DataArray, Unit("Pa")]
            b: xr.DataArray

        def node() -> Out: ...

        assert declarations_from_signature(node)[1] == {"a": Declared(unit=Unit("Pa"))}

    def test_undeclared_dataclass_field_is_omitted(self):
        @dataclass
        class Out:
            a: Annotated[xr.DataArray, Unit("Pa")]
            b: int

        def node() -> Out: ...

        assert declarations_from_signature(node)[1] == {"a": Declared(unit=Unit("Pa"))}

    def test_dataclass_parameter_is_not_read(self):
        @dataclass
        class Params:
            a: Annotated[xr.DataArray, Unit("Pa")]

        def node(p: Params) -> None: ...

        assert declarations_from_signature(node) == ({}, None)

    def test_single_declared_return(self):
        def node() -> Annotated[xr.DataArray, Unit("Pa"), Dims("t")]: ...

        _, output = declarations_from_signature(node)
        assert output == Declared(unit=Unit("Pa"), dims=Dims("t"))

    def test_typeddict_return(self):
        class Out(TypedDict):
            hi: Annotated[xr.DataArray, Unit("Pa")]
            lo: Annotated[xr.DataArray, Dtype("f8")]

        def node() -> Out: ...

        _, output = declarations_from_signature(node)
        assert output == {
            "hi": Declared(unit=Unit("Pa")),
            "lo": Declared(dtype=Dtype("f8")),
        }

    def test_dataclass_return(self):
        @dataclass
        class Out:
            hi: Annotated[xr.DataArray, Dims("time")]
            plain: int

        def node() -> Out: ...

        _, output = declarations_from_signature(node)
        assert output == {"hi": Declared(dims=Dims("time"))}

    def test_undeclared_return_is_none(self):
        def node() -> xr.DataArray: ...

        _, output = declarations_from_signature(node)
        assert output is None


class TestDeclared:
    def test_defaults_are_all_none(self):
        assert Declared() == Declared(None, None, None, None, None)

    def test_field_order_matches_annotate_kwargs(self):
        # Positional construction follows (unit, dims, dtype, coords, freq).
        d = Declared(Unit("Pa"), Dims("t"), Dtype("f8"), Coords("t"), Freq("D"))
        assert (d.unit, d.dims, d.dtype, d.coords, d.freq) == (
            Unit("Pa"),
            Dims("t"),
            Dtype("f8"),
            Coords("t"),
            Freq("D"),
        )


class TestRoundTripWithAnnotate:
    def test_reader_is_inverse_of_writer(self):
        hint = annotate(
            unit="Pa",
            dims=("time",),
            dtype="float64",
            coords=("time",),
            freq="7D",
        )
        _, output = declarations_from_signature(_return_hint(hint))
        assert output == Declared(
            unit=Unit("Pa"),
            dims=Dims("time"),
            dtype=Dtype("float64"),
            coords=Coords("time"),
            freq=Freq("7D"),
        )

    def test_declared_rebuilds_its_own_hint(self):
        d = Declared(
            Unit("hPa"),
            Dims("x", ordered=True),
            Dtype("i4"),
            Coords("x"),
            Freq("W-SUN", anchored=True),
        )
        hint = annotate(
            unit=d.unit, dims=d.dims, dtype=d.dtype, coords=d.coords, freq=d.freq
        )
        _, output = declarations_from_signature(_return_hint(hint))
        assert output == d

    def test_undeclared_hint_still_yields_none(self):
        _, output = declarations_from_signature(_return_hint(annotate()))
        assert output is None


class TestUnitStr:
    def test_str_is_bare_unit(self):
        assert str(Unit("Pa")) == "Pa"
        assert f"{Unit('m s-1')}" == "m s-1"

    def test_repr_unchanged(self):
        assert repr(Unit("Pa")) == "Unit('Pa')"


class TestPredicatesAcceptMarkers:
    def test_units_compatible_with_markers(self):
        assert units_compatible(Unit("hPa"), Unit("Pa"))
        assert units_compatible(Unit("hPa"), "Pa")  # mixed
        assert not units_compatible(Unit("Pa"), Unit("kg"))

    def test_units_equal_with_markers(self):
        assert units_equal(Unit("Pa"), Unit("pascal"))
        assert units_equal("Pa", Unit("pascal"))  # mixed
        assert not units_equal(Unit("hPa"), Unit("Pa"))


DUPLICATES = [
    pytest.param((Unit("Pa"), Unit("hPa")), "Unit", id="unit"),
    pytest.param((Dims("time"), Dims("x")), "Dims", id="dims"),
    pytest.param((Dtype("float64"), Dtype("int64")), "Dtype", id="dtype"),
    pytest.param((Coords("time"), Coords("lat")), "Coords", id="coords"),
    pytest.param((Freq("D"), Freq("7D")), "Freq", id="freq"),
]


class TestOneMarkerPerFacet:
    """A hint declares each facet at most once, so ``Declared`` loses nothing."""

    @pytest.mark.parametrize(("markers", "kind"), DUPLICATES)
    def test_hint_reader_rejects_a_duplicate(self, markers, kind):
        with pytest.raises(ValueError, match=f"2 {kind} markers"):
            declaration_from_hint(Annotated[xr.DataArray, *markers])

    @pytest.mark.parametrize(("markers", "kind"), DUPLICATES)
    def test_error_names_the_parameter(self, markers, kind):
        def node(p: Annotated[xr.DataArray, *markers]) -> None: ...  # type: ignore[valid-type]

        with pytest.raises(ValueError, match=f"2 {kind} markers") as info:
            declarations_from_signature(node)
        assert any("input 'p'" in note for note in info.value.__notes__)

    def test_error_names_an_output_field(self):
        class Out(TypedDict):
            q: Annotated[xr.DataArray, Dims("time"), Dims("x")]

        with pytest.raises(ValueError, match="2 Dims markers") as info:
            declarations_from_signature(_return_hint(Out))
        assert any("output 'q'" in note for note in info.value.__notes__)

    def test_enforce_rejects_at_decoration(self):
        with pytest.raises(ValueError, match="2 Freq markers"):

            @xra.enforce
            def f(p: Annotated[xr.DataArray, Freq("D"), Freq("D")]): ...

    def test_bare_string_beside_a_unit_marker_is_a_description(self):
        hint = Annotated[xr.DataArray, "surface pressure", Unit("Pa")]
        assert declaration_from_hint(hint) == Declared(unit=Unit("Pa"))

    def test_different_facets_still_combine(self):
        hint = Annotated[
            xr.DataArray,
            Unit("Pa"),
            Dims("time"),
            Dtype("f8"),
            Coords("time"),
            Freq("D"),
        ]
        declared = declaration_from_hint(hint)
        assert declared is not None
        assert declared.schema == [Dims("time"), Dtype("f8"), Coords("time")]


_PA = Annotated[xr.DataArray, Unit("Pa")]


class TestReadsThroughWrappers:
    @pytest.mark.parametrize(
        "hint",
        [_PA | None, typing.Optional[_PA], None | _PA, _PA | int],  # noqa: UP045
    )
    def test_union_member_declares(self, hint):
        declared = declaration_from_hint(hint)
        assert declared is not None
        assert declared.unit == Unit("Pa")

    def test_union_without_a_declaration_declares_nothing(self):
        assert declaration_from_hint(xr.DataArray | None) is None
        hint = Annotated[int, Unit("Pa")] | None  # pyright: ignore[reportOperatorIssue]
        assert declaration_from_hint(hint) is None

    def test_two_declared_union_members_are_malformed(self):
        hint = _PA | Annotated[xr.DataArray, Unit("kg")]
        with pytest.raises(ValueError, match="union declares 2"):
            declaration_from_hint(hint)

    def test_typeddict_qualifiers_either_way_round(self):
        class Out(typing.TypedDict):
            outside: typing.NotRequired[_PA]
            inside: Annotated[typing.NotRequired[xr.DataArray], Unit("Pa")]
            required: typing.Required[_PA]

        def f() -> Out: ...

        _, output = declarations_from_signature(f)
        assert isinstance(output, dict)
        assert set(output) == {"outside", "inside", "required"}
        assert all(decl.unit == Unit("Pa") for decl in output.values())

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="ReadOnly is 3.13+")
    def test_readonly(self):
        hint = typing.ReadOnly[_PA]  # pyright: ignore[reportAttributeAccessIssue]
        assert declaration_from_hint(hint) == declaration_from_hint(_PA)

    def test_optional_parameter_is_enforced(self):
        @xra.enforce
        def f(x: _PA | None = None) -> None: ...

        f()
        f(xr.DataArray([1.0], attrs={"units": "Pa"}))
        with pytest.raises(pint.DimensionalityError):
            f(xr.DataArray([1.0], attrs={"units": "kg"}))

    def test_not_required_field_is_enforced(self):
        class Out(typing.TypedDict):
            a: typing.NotRequired[Annotated[xr.DataArray, Unit("Pa")]]

        @xra.enforce
        def f() -> Out:
            return {"a": xr.DataArray([1.0])}

        with pytest.raises(ValueError, match="no 'units' attribute"):
            f()


type _ToNone = _ToNone | None  # pyright: ignore[reportGeneralTypeIssues]
type _ToItself = _ToItself  # pyright: ignore[reportGeneralTypeIssues]
type _ToMarker = Annotated[_ToMarker, Dims("x")]
type _ToUnion = Annotated[xr.DataArray, Unit("Pa")] | _ToUnion  # pyright: ignore[reportGeneralTypeIssues]
type _First = _Second | None  # pyright: ignore[reportGeneralTypeIssues]
type _Second = _First | int
type _InContainer = Annotated[xr.DataArray, Unit("Pa")] | list[_InContainer]
type _Shared = Annotated[xr.DataArray, Unit("Pa")]
type _UsesShared = Annotated[_Shared, Dims("x")] | None


def _returning(hint):
    def f():
        return None

    f.__annotations__ = {"return": hint}
    return f


class TestSelfReferringAliases:
    @pytest.mark.parametrize(
        "alias", [_ToNone, _ToItself, _ToMarker, _ToUnion, _First, _Second]
    )
    def test_raises_a_value_error_naming_the_alias(self, alias):
        with pytest.raises(ValueError, match=r"type alias _\w+ refers to itself"):
            declaration_from_hint(alias)

    def test_enforce_raises_it_at_decoration(self):
        with pytest.raises(ValueError, match="refers to itself"):
            xra.enforce(_returning(_ToNone))

    def test_signature_reader_raises_it_with_the_location(self):
        with pytest.raises(ValueError, match="refers to itself") as excinfo:
            declarations_from_signature(_returning(_ToNone))
        assert any("output" in note for note in excinfo.value.__notes__)

    def test_self_reference_in_a_container_is_fine(self):
        assert declaration_from_hint(_InContainer) == Declared(unit=Unit("Pa"))

    def test_an_alias_used_on_two_branches_is_not_a_cycle(self):
        declared = declaration_from_hint(_UsesShared)
        assert declared == Declared(unit=Unit("Pa"), dims=Dims("x"))

    def test_custom_extractor_does_not_recurse(self):
        inputs, output = xra.walk_signature(_returning(_ToNone), lambda hint: None)
        assert inputs == {}
        assert output is None
