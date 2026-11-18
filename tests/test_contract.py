"""Whole-declaration checks, which tools rely on to agree with `check_annotations`."""

from typing import Annotated

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Coords, Declared, Dims, Dtype, Freq, Unit
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
            first = violations[0]
            assert first == xra.Violation(
                "p", exc.facet, exc.declared, exc.found, str(exc)
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


class TestViolationFields:
    """What a tool reads instead of the message."""

    @pytest.mark.parametrize(
        ("value", "declared", "facet", "found"),
        [
            (_array(unit="hPa"), Unit("Pa"), "unit", "hPa"),
            (_array(unit="K"), Unit("Pa"), "unit", "K"),
            (_array().drop_attrs(), Unit("Pa"), "unit", None),
            (_array(unit="x" * 300), Unit("Pa"), "unit", None),
            (_array(unit="not_a_unit"), Unit("Pa"), "unit", "not_a_unit"),
            (_array(), Dims("x", "y"), "dims", ("time",)),
            (_array(), Dims("x", ordered=True), "dims", ("time",)),
            (_array(), Coords("lat"), "coords", ("time",)),
            (_array().astype("f4"), Dtype("float64"), "dtype", "float32"),
            (_array(), Dtype("int"), "dtype", "float64"),
            (_array(freq="D"), Freq("h"), "freq", "D"),
            (_array().drop_vars("time"), Freq("h"), "freq", None),
        ],
        ids=[
            "unit-scale",
            "unit-dimension",
            "unit-missing",
            "unit-over-long",
            "unit-unparseable",
            "dims",
            "dims-ordered",
            "coords",
            "dtype-width",
            "dtype-kind",
            "freq",
            "freq-no-axis",
        ],
    )
    def test_names_the_facet_the_marker_and_what_was_found(
        self, value, declared, facet, found
    ):
        slot = Declared(**{facet: declared})
        [violation] = xra.declaration_violations(value, slot, "p")
        assert violation.name == "p"
        assert violation.facet == facet
        assert violation.declared == declared
        assert violation.found == found

    def test_irregular_spacing_has_no_frequency(self):
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"])
        value = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": times})
        [violation] = xra.declaration_violations(value, Declared(freq=Freq("D")), "p")
        assert (violation.facet, violation.found) == ("freq", None)

    @pytest.mark.parametrize(
        ("value", "found"), [(None, "None"), ("hPa", "str"), (1.0, "float")]
    )
    def test_a_value_that_is_not_an_array_names_its_type(self, value, found):
        [violation] = xra.declaration_violations(value, Declared(unit=Unit("Pa")), "p")
        assert (violation.facet, violation.declared) == (None, None)
        assert violation.found == found

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


class TestErrorFields:
    """`check_annotations` users get the same fields on the error it raises."""

    def test_an_input_error_carries_them(self):
        @xra.check_annotations
        def f(p: Annotated[xr.DataArray, Unit("Pa")]) -> None:
            pass

        with pytest.raises(UnitError) as info:
            f(_array(unit="hPa"))
        assert (info.value.facet, info.value.declared, info.value.found) == (
            "unit",
            Unit("Pa"),
            "hPa",
        )

    def test_an_output_error_carries_them(self):
        @xra.check_annotations
        def f() -> Annotated[xr.DataArray, Dims("x")]:
            return _array()

        with pytest.raises(SchemaError) as info:
            f()
        assert (info.value.facet, info.value.declared, info.value.found) == (
            "dims",
            Dims("x"),
            ("time",),
        )

    def test_a_record_mismatch_has_no_facet(self):
        @xra.check_annotations
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return 1.0  # pyright: ignore[reportReturnType]

        with pytest.raises(xra.ContractError) as info:
            f()
        assert (info.value.facet, info.value.declared) == (None, None)
        assert info.value.found == "float"

    def test_the_fields_survive_pickling(self):
        import pickle

        with pytest.raises(UnitError) as info:
            xra.check_declared(_array(unit="hPa"), Declared(unit=Unit("Pa")), "p")
        restored = pickle.loads(pickle.dumps(info.value))
        assert type(restored) is UnitError
        assert str(restored) == str(info.value)
        assert (restored.facet, restored.declared, restored.found) == (
            "unit",
            Unit("Pa"),
            "hPa",
        )

    def test_quantify_and_convert_still_raise_value_error(self):
        with pytest.raises(ValueError, match="no 'units'"):
            xra.units.quantify(_array().drop_attrs())
