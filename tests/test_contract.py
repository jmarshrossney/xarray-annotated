"""Whole-declaration checks, which tools rely on to agree with `check_annotations`."""

import dataclasses
import json
import pickle
from typing import Annotated, TypedDict, get_args

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Coords, Declared, Dims, Dtype, Freq, Unit
from xarray_annotated._errors import Kind
from xarray_annotated.schema import SchemaError
from xarray_annotated.temporal import FreqError
from xarray_annotated.units import UnitError


def _array(unit="Pa", freq="h"):
    time = pd.date_range("2000-01-01", periods=4, freq=freq)
    return xr.DataArray(
        np.ones(4), dims="time", coords={"time": time}, attrs={"units": unit}
    )


type Full = Annotated[
    xr.DataArray, Unit("Pa"), Dims("time"), Dtype("float"), Coords("time"), Freq("h")
]


@xra.check_annotations
def _full(p: Full) -> None:
    pass


@xra.check_annotations
def _optional(p: Annotated[xr.DataArray, Unit("Pa")] | None) -> None:
    pass


@xra.check_annotations
def _or_float(p: Annotated[xr.DataArray, Unit("Pa")] | float) -> None:
    pass


def _input_declaration(func):
    return xra.declarations_from_signature(func)[0]["p"]


class TestAgreesWithEnforce:
    """A pre-flight must give the verdict the run would."""

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (_array(), None),
            (_array(unit="pascal"), None),
            (_array(unit="hPa"), UnitError),
            (_array(unit="K"), UnitError),
            (_array().drop_vars("time"), SchemaError),
            (_array().astype(int), SchemaError),
            (_array().rename(time="t"), SchemaError),
            (_array(freq="D"), FreqError),
            (None, xra.ContractError),
            (1.0, xra.ContractError),
            (xr.Dataset(), xra.ContractError),
        ],
        ids=[
            "match",
            "respelled",
            "other-scale",
            "other-dimension",
            "no-coord",
            "int",
            "renamed",
            "daily",
            "none",
            "float",
            "dataset",
        ],
    )
    def test_on_every_marker(self, value, error):
        declared = _input_declaration(_full)
        if error is None:
            assert xra.check_declared(value, declared, "p") is value
            _full(value)
            return
        with pytest.raises(error):
            xra.check_declared(value, declared, "p")
        with pytest.raises(error):
            _full(value)

    @pytest.mark.parametrize(
        ("func", "value"), [(_optional, None), (_or_float, 1.0), (_or_float, "text")]
    )
    def test_on_a_value_the_hint_allows(self, func, value):
        assert xra.check_declared(value, _input_declaration(func), "p") is value
        func(value)

    @pytest.mark.parametrize(
        ("func", "value", "got"),
        [(_optional, 1.0, "got float"), (_or_float, None, "got None")],
    )
    def test_on_a_value_the_hint_refuses(self, func, value, got):
        with pytest.raises(xra.ContractError, match=got):
            xra.check_declared(value, _input_declaration(func), "p")
        with pytest.raises(xra.ContractError, match=got):
            func(value)

    def test_units_are_checked_first(self):
        wrong_twice = _array(unit="hPa").rename(time="t")
        with pytest.raises(UnitError, match="hPa"):
            xra.check_declared(wrong_twice, _input_declaration(_full), "p")


class TestCheckDeclared:
    @pytest.mark.parametrize("value", [_array(unit="hPa"), None, 1.0])
    def test_switched_off_checks_nothing(self, value):
        with xra.enabled(False):
            result = xra.check_declared(value, _input_declaration(_full), "p")
        assert result is value

    def test_nothing_declared_passes_anything(self):
        value = _array()
        assert xra.check_declared(value, Declared(), "p") is value

    def test_names_the_function_when_given_one(self):
        with pytest.raises(xra.ContractError, match=r"\[f\] input 'p'"):
            xra.check_declared(None, Declared(unit=Unit("Pa")), "p", "f")

    def test_names_only_the_input_otherwise(self):
        with pytest.raises(xra.ContractError, match=r"^input 'p'"):
            xra.check_declared(None, Declared(unit=Unit("Pa")), "p")


class TestDeclarationConflicts:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Pa", "Pa"),
            ("Pa", "pascal"),
            ("Pa", "hPa"),
            ("Pa", "K"),
            ("degC", "K"),
        ],
    )
    def test_units_conflict_exactly_when_check_annotations_would_refuse(self, a, b):
        conflicts = xra.declaration_conflicts(
            Declared(unit=Unit(a)), Declared(unit=Unit(b))
        )

        @xra.check_annotations
        def consumer(p: Annotated[xr.DataArray, Unit(b)]) -> None:
            pass

        try:
            consumer(_array(unit=a))
        except UnitError:
            assert conflicts
        else:
            assert conflicts == []

    def test_dimensionally_incompatible_units_say_so(self):
        [conflict] = xra.declaration_conflicts(
            Declared(unit=Unit("Pa")), Declared(unit=Unit("K"))
        )
        assert "dimensionally incompatible" in str(conflict)

    @pytest.mark.parametrize(
        ("a", "b", "facet"),
        [
            (Declared(dims=Dims("x")), Declared(dims=Dims("x", "y")), "dims"),
            (Declared(dtype=Dtype("float")), Declared(dtype=Dtype("int")), "dtype"),
            (Declared(freq=Freq("h")), Declared(freq=Freq("D")), "freq"),
        ],
    )
    def test_each_facet_is_compared(self, a, b, facet):
        [conflict] = xra.declaration_conflicts(a, b)
        assert conflict.facet == facet
        assert conflict.message.startswith(f"{facet}:")
        assert xra.declaration_conflicts(b, a)

    def test_records_the_markers_in_the_order_given(self):
        a, b = Declared(unit=Unit("Pa")), Declared(unit=Unit("hPa"))
        [forward] = xra.declaration_conflicts(a, b)
        [backward] = xra.declaration_conflicts(b, a)
        assert (forward.a, forward.b) == (Unit("Pa"), Unit("hPa"))
        assert (backward.a, backward.b) == (Unit("hPa"), Unit("Pa"))

    def test_a_record_reads_as_its_message(self):
        [conflict] = xra.declaration_conflicts(
            Declared(unit=Unit("Pa")), Declared(unit=Unit("hPa"))
        )
        assert str(conflict) == f"{conflict}" == conflict.message
        assert (
            conflict.message == "unit: Unit('Pa') and Unit('hPa') are different units"
        )

    def test_every_conflicting_facet_is_reported(self):
        a = Declared(unit=Unit("Pa"), dims=Dims("x"), dtype=Dtype("float"))
        b = Declared(unit=Unit("K"), dims=Dims("y"), dtype=Dtype("int"))
        assert len(xra.declaration_conflicts(a, b)) == 3

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (Declared(unit=Unit("Pa")), Declared()),
            (Declared(unit=Unit("Pa")), Declared(dims=Dims("x"))),
            (Declared(coords=Coords("x")), Declared(coords=Coords("y"))),
            (Declared(dims=Dims("x", "y")), Declared(dims=Dims("y", "x"))),
            (Declared(dtype=Dtype("float")), Declared(dtype=Dtype("float32"))),
        ],
        ids=["one-side", "different-facets", "coords", "loose-dims", "kind-and-width"],
    )
    def test_what_is_not_provably_inconsistent_does_not_conflict(self, a, b):
        assert xra.declaration_conflicts(a, b) == []

    @pytest.mark.parametrize("value", [True, False])
    def test_ignores_the_switch(self, value):
        a, b = Declared(unit=Unit("Pa")), Declared(unit=Unit("hPa"))
        with xra.enabled(value):
            assert xra.declaration_conflicts(a, b)

    def test_malformed_declaration_raises_value_error(self):
        malformed = Declared(unit=Unit("not_a_unit"))
        with pytest.raises(ValueError, match="not_a_unit"):
            xra.declaration_conflicts(malformed, Declared(unit=Unit("Pa")))


class TestDeclarationConflictsOnTwoAxes:
    def test_different_dims_do_not_conflict_as_one_array_can_pass_both(self):
        a = Declared(freq=Freq("D", dim="a"))
        b = Declared(freq=Freq("7D", dim="b"))
        assert xra.declaration_conflicts(a, b) == []
        times = {
            "a": pd.date_range("2000-01-01", periods=4, freq="D"),
            "b": pd.date_range("2000-01-01", periods=4, freq="7D"),
        }
        both = xr.DataArray(np.ones((4, 4)), dims=("a", "b"), coords=times)
        for declared in (a, b):
            assert xra.check_declared(both, declared, "p") is both


class TestMalformedDeclaration:
    """Malformed means `ValueError`, before any array is looked at."""

    @pytest.mark.parametrize(
        "declared",
        [
            Declared(dtype=Dtype("notdtype")),
            Declared(unit=Unit("not_a_unit")),
            Declared(freq=Freq("bogus")),
        ],
        ids=["dtype", "unit", "freq"],
    )
    @pytest.mark.parametrize("value", [_array(), None], ids=["array", "none"])
    def test_raises_value_error(self, declared, value):
        with pytest.raises(ValueError, match="'p'"):
            xra.check_declared(value, declared, "p")

    def test_switched_off_checks_nothing(self):
        with xra.enabled(False):
            assert xra.check_declared(None, Declared(freq=Freq("bogus")), "p") is None


class TestAssertValidDeclaration:
    def test_accepts_a_valid_declaration(self):
        xra.assert_valid_declaration(Declared(unit=Unit("Pa"), dims=Dims("x")), "node")

    @pytest.mark.parametrize(
        "declared",
        [
            Declared(unit=Unit("not_a_unit")),
            Declared(dims=Dims("x", "x")),
            Declared(dtype=Dtype("notdtype")),
            Declared(freq=Freq("bogus")),
        ],
        ids=["unit", "dims", "dtype", "freq"],
    )
    def test_names_the_context_on_every_facet(self, declared):
        with pytest.raises(ValueError, match="node 'x'"):
            xra.assert_valid_declaration(declared, "node 'x'")

    def test_ignores_the_switch(self):
        with xra.enabled(False), pytest.raises(ValueError, match="not_a_unit"):
            xra.assert_valid_declaration(Declared(unit=Unit("not_a_unit")), "node")


class TestDeclarationViolations:
    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (_array(), None),
            (_array(unit="hPa"), UnitError),
            (_array(unit="K"), UnitError),
            (_array().rename(time="x"), SchemaError),
            (_array(freq="D"), FreqError),
            (None, xra.ContractError),
            (1.0, xra.ContractError),
        ],
        ids=["ok", "scale", "dimension", "dims", "freq", "none", "float"],
    )
    def test_agrees_with_check_declared(self, value, error):
        declared = _input_declaration(_full)
        violations = xra.declaration_violations(value, declared, "p")
        if error is None:
            assert violations == []
            assert xra.check_declared(value, declared, "p") is value
        else:
            with pytest.raises(error) as info:
                xra.check_declared(value, declared, "p")
            exc = info.value
            assert violations[0] == xra.Violation(
                name="p",
                qualname=None,
                role="input",
                facet=exc.facet,
                kind=exc.kind,
                declared=exc.declared,
                found=exc.found,
                hint=exc.hint,
                message=str(exc),
            )

    def test_reports_every_failing_marker(self):
        declared = Declared(
            unit=Unit("Pa"), dims=Dims("x"), dtype=Dtype("int"), coords=Coords("lat")
        )
        violations = xra.declaration_violations(_array(unit="kg"), declared, "p")
        assert len(violations) == 4

    def test_reports_units_first_and_names_the_array_once(self):
        declared = Declared(unit=Unit("Pa"), dims=Dims("x"))
        unit, dims = xra.declaration_violations(_array(unit="kg"), declared, "p")
        assert (unit.facet, dims.facet) == ("unit", "dims")
        assert "[mass]" in unit.message
        for violation in (unit, dims):
            assert violation.name == "p"
            assert violation.message.count("'p'") == 1

    def test_prefixes_the_qualname(self):
        declared = Declared(unit=Unit("Pa"))
        [violation] = xra.declaration_violations(_array(unit="kg"), declared, "p", "f")
        assert violation.message.startswith("[f] input 'p'")
        assert violation.name == "p"

    @pytest.mark.parametrize("value", [_array(), None], ids=["array", "none"])
    def test_a_malformed_declaration_raises(self, value):
        with pytest.raises(ValueError, match="not_a_unit"):
            xra.declaration_violations(value, Declared(unit=Unit("not_a_unit")), "p")

    def test_allows_what_the_hint_allows(self):
        declared = _input_declaration(_optional)
        assert xra.declaration_violations(None, declared, "p") == []

    def test_switched_off_reports_nothing(self):
        with xra.enabled(False):
            assert (
                xra.declaration_violations(None, Declared(unit=Unit("Pa")), "p") == []
            )


class TestResolveEnabled:
    def test_follows_the_switch(self):
        with xra.enabled(False):
            assert xra.resolve_enabled() is False
        with xra.enabled(True):
            assert xra.resolve_enabled() is True


def _two_axes():
    times = pd.date_range("2000-01-01", periods=4, freq="h")
    return xr.DataArray(
        np.ones((4, 4)), dims=("a", "b"), coords={"a": times, "b": times}
    )


def _irregular():
    times = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"])
    return xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": times})


def _wednesdays():
    times = pd.date_range("2025-01-01", periods=2, freq="7D")
    return xr.DataArray([1.0, 2.0], dims="time", coords={"time": times})


def _time_by_x():
    times = pd.date_range("2000-01-01", periods=4, freq="h")
    return xr.DataArray(np.ones((4, 2)), dims=("time", "x"), coords={"time": times})


#: (value, marker, kind, found), one or more per kind a violation can have.
VIOLATIONS = {
    "unit-missing": (_array().drop_attrs(), Unit("Pa"), "unit_missing", None),
    "unit-too-long": (_array(unit="x" * 300), Unit("Pa"), "unit_too_long", None),
    "unit-unparseable": (
        _array(unit="not_a_unit"),
        Unit("Pa"),
        "unit_unparseable",
        "not_a_unit",
    ),
    "unit-different": (_array(unit="hPa"), Unit("Pa"), "unit_different", "hPa"),
    "unit-incompatible": (_array(unit="K"), Unit("Pa"), "unit_incompatible", "K"),
    "dims-mismatch": (_array(), Dims("x", "y"), "dims_mismatch", ("time",)),
    "dims-mismatch-ordered": (
        _array(),
        Dims("x", ordered=True),
        "dims_mismatch",
        ("time",),
    ),
    "dims-order": (
        _time_by_x(),
        Dims("x", "time", ordered=True),
        "dims_order",
        ("time", "x"),
    ),
    "coords-missing": (_array(), Coords("lat"), "coords_missing", ("time",)),
    "dtype-kind": (_array(), Dtype("int"), "dtype_kind", "float64"),
    "dtype-kind-exact": (_array(), Dtype("int64"), "dtype_kind", "float64"),
    "dtype-width": (_array().astype("f4"), Dtype("float64"), "dtype_width", "float32"),
    "freq-no-axis": (_array().drop_vars("time"), Freq("h"), "freq_no_axis", None),
    "freq-no-named-axis": (_array(), Freq("h", dim="t"), "freq_no_axis", None),
    "freq-ambiguous-axis": (_two_axes(), Freq("h"), "freq_ambiguous_axis", None),
    "freq-irregular": (_irregular(), Freq("D"), "freq_irregular", None),
    "freq-mismatch": (_array(freq="D"), Freq("h"), "freq_mismatch", "D"),
    "freq-mismatch-stepwise": (
        _array(freq="2h"),
        Freq("h", gaps_ok=True),
        "freq_mismatch",
        None,
    ),
    "freq-phase": (_wednesdays(), Freq("W-SUN"), "freq_phase", None),
}

#: (a, b, kind), one or more per kind a conflict can have.
CONFLICTS = {
    "unit-different": (Unit("Pa"), Unit("hPa"), "unit_different"),
    "unit-incompatible": (Unit("Pa"), Unit("K"), "unit_incompatible"),
    "dims-mismatch": (Dims("x"), Dims("x", "y"), "dims_mismatch"),
    "dims-order": (
        Dims("x", "y", ordered=True),
        Dims("y", "x", ordered=True),
        "dims_order",
    ),
    "dtype-kind": (Dtype("float"), Dtype("int"), "dtype_kind"),
    "dtype-width": (Dtype("float64"), Dtype("float32"), "dtype_width"),
    "freq-mismatch": (Freq("h"), Freq("D"), "freq_mismatch"),
}

_SLOTS = {Unit: "unit", Dims: "dims", Coords: "coords", Dtype: "dtype", Freq: "freq"}

#: Kinds only `check_annotations` raises, tested in `TestErrorFields`.
_RECORD_KINDS = {"record_class", "record_field_missing"}


def _violation_of(value, marker):
    declared = Declared(**{_SLOTS[type(marker)]: marker})
    [violation] = xra.declaration_violations(value, declared, "p", "f")
    return violation


def test_every_kind_is_reached():
    reached = (
        {kind for _, _, kind, _ in VIOLATIONS.values()}
        | {kind for _, _, kind in CONFLICTS.values()}
        | _RECORD_KINDS
        | {"not_array"}
    )
    assert reached == set(get_args(Kind.__value__))


def test_every_kind_is_documented():
    for kind in get_args(Kind.__value__):
        assert f'`"{kind}"`' in (xra.Violation.__doc__ or "")
    for _, _, kind in CONFLICTS.values():
        assert f'`"{kind}"`' in (xra.Conflict.__doc__ or "")


class TestViolationFields:
    """What a tool reads instead of the message."""

    @pytest.mark.parametrize(
        ("value", "marker", "kind", "found"), VIOLATIONS.values(), ids=VIOLATIONS
    )
    def test_says_what_failed_why_and_what_was_found(self, value, marker, kind, found):
        violation = _violation_of(value, marker)
        assert violation.name == "p"
        assert violation.qualname == "f"
        assert violation.role == "input"
        assert violation.facet == _SLOTS[type(marker)]
        assert violation.kind == kind
        assert violation.declared == marker
        assert violation.found == found

    @pytest.mark.parametrize(
        ("value", "marker"),
        [(value, marker) for value, marker, _, _ in VIOLATIONS.values()],
        ids=VIOLATIONS,
    )
    def test_a_hint_is_part_of_the_message(self, value, marker):
        violation = _violation_of(value, marker)
        assert violation.hint is None or violation.hint in violation.message

    @pytest.mark.parametrize(
        ("case", "expected"),
        [
            ("unit-missing", "Attach the unit"),
            ("unit-different", "xra.units.convert(da, 'Pa')"),
            ("unit-incompatible", "Fix the data or the declaration."),
            ("dtype-width", "Dtype('float') would accept any width"),
            ("freq-ambiguous-axis", "name one with Freq(..., dim=...)"),
        ],
    )
    def test_says_what_to_do_where_the_message_does(self, case, expected):
        value, marker, _, _ = VIOLATIONS[case]
        hint = _violation_of(value, marker).hint
        assert hint is not None
        assert expected in hint

    def test_names_how_to_make_a_time_axis(self):
        value = _array().drop_indexes("time")
        hint = _violation_of(value, Freq("h")).hint
        assert hint is not None
        assert ".set_xindex('time')" in hint

    @pytest.mark.parametrize(
        ("value", "found", "hinted"),
        [(None, "NoneType", True), ("hPa", "str", False), (1.0, "float", False)],
    )
    def test_a_value_that_is_not_an_array_names_its_type(self, value, found, hinted):
        [violation] = xra.declaration_violations(value, Declared(unit=Unit("Pa")), "p")
        assert (violation.facet, violation.declared) == (None, None)
        assert violation.kind == "not_array"
        assert violation.found == found
        assert (violation.hint is not None) == hinted

    def test_records_hash_so_consumers_can_be_deduplicated(self):
        value = _array(unit="kg")
        found = [
            v
            for d in (Declared(unit=Unit("Pa")), Declared(unit=Unit("Pa")))
            for v in xra.declaration_violations(value, d, "p")
        ]
        assert len(dict.fromkeys(found)) == 1

    def test_a_record_reads_as_its_message(self):
        [violation] = xra.declaration_violations(
            _array(unit="kg"), Declared(unit=Unit("Pa")), "p"
        )
        assert str(violation) == f"{violation}" == violation.message


class TestToDict:
    @pytest.mark.parametrize(
        ("value", "marker"),
        [(value, marker) for value, marker, _, _ in VIOLATIONS.values()],
        ids=VIOLATIONS,
    )
    def test_a_violation_round_trips_through_json(self, value, marker):
        violation = _violation_of(value, marker)
        record = violation.to_dict()
        assert json.loads(json.dumps(record)) == record
        assert record["kind"] == violation.kind
        assert record["message"] == violation.message

    def test_a_violation_has_every_field(self):
        violation = _violation_of(_array(unit="hPa"), Unit("Pa"))
        assert set(violation.to_dict()) == {
            f.name for f in dataclasses.fields(violation)
        }

    def test_a_value_that_is_not_an_array_has_no_marker(self):
        [violation] = xra.declaration_violations(None, Declared(unit=Unit("Pa")), "p")
        assert violation.to_dict()["declared"] is None

    @pytest.mark.parametrize(
        ("marker", "expected"),
        [
            (Unit("Pa"), {"unit": "Pa"}),
            (Dims("x", ordered=True), {"names": ["x"], "ordered": True}),
            (Coords("lat", "lon"), {"names": ["lat", "lon"]}),
            (Dtype("float32"), {"dtype": "float32"}),
            (
                Freq("D", dim="t", gaps_ok=True),
                {"freq": "D", "dim": "t", "gaps_ok": True},
            ),
        ],
        ids=lambda v: type(v).__name__ if not isinstance(v, dict) else "",
    )
    def test_a_marker_becomes_what_it_declares(self, marker, expected):
        violation = xra.Violation(
            name="p",
            qualname=None,
            role="input",
            facet=_SLOTS[type(marker)],
            kind=None,
            declared=marker,
            found=None,
            hint=None,
            message="",
        )
        assert violation.to_dict()["declared"] == expected

    @pytest.mark.parametrize(("a", "b", "kind"), CONFLICTS.values(), ids=CONFLICTS)
    def test_a_conflict_round_trips_through_json(self, a, b, kind):
        slot = _SLOTS[type(a)]
        [conflict] = xra.declaration_conflicts(
            Declared(**{slot: a}), Declared(**{slot: b})
        )
        record = conflict.to_dict()
        assert json.loads(json.dumps(record)) == record
        assert set(record) == {f.name for f in dataclasses.fields(conflict)}


class TestConflictKinds:
    @pytest.mark.parametrize(("a", "b", "kind"), CONFLICTS.values(), ids=CONFLICTS)
    def test_says_why_either_way_round(self, a, b, kind):
        slot = _SLOTS[type(a)]
        for x, y in ((a, b), (b, a)):
            [conflict] = xra.declaration_conflicts(
                Declared(**{slot: x}), Declared(**{slot: y})
            )
            assert (conflict.facet, conflict.kind) == (slot, kind)
            assert (conflict.a, conflict.b) == (x, y)


class TestErrorFields:
    """`check_annotations` users get the same fields on the error it raises."""

    def test_an_input_error_carries_them(self):
        @xra.check_annotations
        def f(p: Annotated[xr.DataArray, Unit("Pa")]) -> None:
            pass

        with pytest.raises(UnitError) as info:
            f(_array(unit="hPa"))
        exc = info.value
        assert (exc.name, exc.role, exc.facet, exc.kind) == (
            "p",
            "input",
            "unit",
            "unit_different",
        )
        assert exc.qualname is not None
        assert exc.qualname.endswith("f")
        assert (exc.declared, exc.found) == (Unit("Pa"), "hPa")
        assert exc.hint is not None
        assert exc.hint in str(exc)

    @pytest.mark.parametrize(
        ("marker", "error", "kind"),
        [
            (Unit("K"), UnitError, "unit_incompatible"),
            (Dims("x"), SchemaError, "dims_mismatch"),
            (Freq("D"), FreqError, "freq_mismatch"),
        ],
        ids=["unit", "schema", "freq"],
    )
    def test_an_output_error_says_it_is_an_output(self, marker, error, kind):
        def f():
            return _array()

        f.__annotations__["return"] = xra.annotate(**{_SLOTS[type(marker)]: marker})
        with pytest.raises(error) as info:
            xra.check_annotations(f)()
        exc = info.value
        assert (exc.name, exc.role, exc.kind, exc.declared) == (
            "return",
            "output",
            kind,
            marker,
        )

    def test_an_output_that_is_not_an_array_says_so(self):
        @xra.check_annotations
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return 1.0  # pyright: ignore[reportReturnType]

        with pytest.raises(xra.ContractError) as info:
            f()
        exc = info.value
        assert (exc.facet, exc.declared, exc.kind) == (None, None, "not_array")
        assert (exc.role, exc.found) == ("output", "float")

    def test_a_record_of_the_wrong_class_says_so(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.check_annotations
        def f() -> Out:
            return 1.0  # pyright: ignore[reportReturnType]

        with pytest.raises(xra.ContractError) as info:
            f()
        exc = info.value
        assert (exc.kind, exc.role, exc.found, exc.facet) == (
            "record_class",
            "output",
            "float",
            None,
        )

    def test_a_missing_field_says_which(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.check_annotations
        def f() -> Out:
            return {}  # pyright: ignore[reportReturnType]

        with pytest.raises(xra.ContractError) as info:
            f()
        exc = info.value
        assert (exc.kind, exc.name, exc.role) == ("record_field_missing", "p", "output")

    def test_the_fields_survive_pickling(self):
        with pytest.raises(UnitError) as info:
            xra.check_declared(_array(unit="hPa"), Declared(unit=Unit("Pa")), "p")
        restored = pickle.loads(pickle.dumps(info.value))
        assert type(restored) is UnitError
        assert str(restored) == str(info.value)
        assert vars(restored) == vars(info.value)

    def test_a_bare_contract_error_has_none_of_them(self):
        exc = xra.ContractError("message")
        assert (exc.name, exc.kind, exc.facet, exc.hint) == (None, None, None, None)

    def test_quantify_and_convert_still_raise_value_error(self):
        with pytest.raises(ValueError, match="no 'units'"):
            xra.units.quantify(_array().drop_attrs())
