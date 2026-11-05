"""Whole-declaration checks, which tools rely on to agree with `enforce`."""

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


@xra.enforce
def _full(p: Full) -> None:
    pass


@xra.enforce
def _optional(p: Annotated[xr.DataArray, Unit("Pa")] | None) -> None:
    pass


@xra.enforce
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
            (None, TypeError),
            (1.0, TypeError),
            (xr.Dataset(), TypeError),
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
        with pytest.raises(TypeError, match=got):
            xra.check_declared(value, _input_declaration(func), "p")
        with pytest.raises(TypeError, match=got):
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
        with pytest.raises(TypeError, match=r"\[f\] input 'p'"):
            xra.check_declared(None, Declared(unit=Unit("Pa")), "p", "f")

    def test_names_only_the_input_otherwise(self):
        with pytest.raises(TypeError, match=r"^input 'p'"):
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
    def test_units_conflict_exactly_when_enforce_would_refuse(self, a, b):
        conflicts = xra.declaration_conflicts(
            Declared(unit=Unit(a)), Declared(unit=Unit(b))
        )

        @xra.enforce
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
        assert "dimensionally incompatible" in conflict

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
        assert conflict.startswith(f"{facet}:")
        assert xra.declaration_conflicts(b, a)

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
            (None, TypeError),
            (1.0, TypeError),
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
            assert violations[0] == str(info.value)

    def test_reports_every_failing_marker(self):
        declared = Declared(
            unit=Unit("Pa"), dims=Dims("x"), dtype=Dtype("int"), coords=Coords("lat")
        )
        violations = xra.declaration_violations(_array(unit="kg"), declared, "p")
        assert len(violations) == 4

    def test_reports_units_first_and_names_the_array_once(self):
        declared = Declared(unit=Unit("Pa"), dims=Dims("x"))
        unit, dims = xra.declaration_violations(_array(unit="kg"), declared, "p")
        assert "[mass]" in unit
        assert "Dims" in dims or "dims" in dims
        for message in (unit, dims):
            assert message.count("'p'") == 1

    def test_prefixes_the_qualname(self):
        declared = Declared(unit=Unit("Pa"))
        [message] = xra.declaration_violations(_array(unit="kg"), declared, "p", "f")
        assert message.startswith("[f] input 'p'")

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
