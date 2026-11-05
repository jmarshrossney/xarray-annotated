"""Whole-declaration checks, which tools rely on to agree with `enforce`."""

from typing import Annotated

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Coords, Declared, Dims, Dtype, Freq, Unit
from xarray_annotated.schema import SchemaError
from xarray_annotated.temporal import FreqError


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
            (_array(unit="hPa"), ValueError),
            (_array(unit="K"), pint.DimensionalityError),
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
        with pytest.raises(ValueError, match="hPa"):
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
        except (ValueError, pint.DimensionalityError):
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


class TestResolveEnabled:
    def test_follows_the_switch(self):
        with xra.enabled(False):
            assert xra.resolve_enabled() is False
        with xra.enabled(True):
            assert xra.resolve_enabled() is True
