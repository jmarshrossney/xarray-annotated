"""``on_output="require"``: refuse to stamp an output that has no unit of its own (#16).

A wrapper around a library that returns bare arrays declares units it never
computed.  Stamping makes that claim undetectable, since every consumer then
compares one declaration against another.  ``"require"`` makes the wrapper
attach the unit the values really are in, which is then checked.
"""

import dataclasses
import functools
from typing import Annotated, TypedDict

import numpy as np
import pint
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import units
from xarray_annotated.units import Unit


def _da(unit: str | None = None, value: float = 1.0) -> xr.DataArray:
    da = xr.DataArray(np.array([value]), dims="x")
    if unit is not None:
        da.attrs["units"] = unit
    return da


def _require(da: xr.DataArray, declared: str = "Pa") -> xr.DataArray:
    return units.apply_output_units(da, declared, "r", "require")


class TestUnlabelled:
    def test_bare_array_raises_with_the_fix(self):
        with pytest.raises(ValueError, match=r"carries no unit.*\.pint\.quantify\(\)"):
            _require(_da())

    def test_unparseable_label_raises(self):
        with pytest.raises(ValueError, match="unparseable unit label 'not_a_unit'"):
            _require(_da("not_a_unit"))

    def test_over_long_label_raises_without_echoing_it(self):
        with pytest.raises(ValueError, match="unverifiable unit label") as info:
            _require(_da("m" * 10_000))
        assert "m" * 100 not in str(info.value)

    def test_non_string_label_raises(self):
        with pytest.raises(ValueError, match="unverifiable unit label"):
            _require(_da().assign_attrs(units=3))


class TestLabelled:
    def test_equal_label_passes_and_is_restamped(self):
        assert _require(_da("pascal")).attrs["units"] == "Pa"

    def test_compatible_different_label_raises_as_strict(self):
        with pytest.raises(ValueError, match="on_output='require' requires them"):
            _require(_da("hPa"))

    def test_incompatible_label_raises(self):
        with pytest.raises(pint.DimensionalityError, match="on_output='require'"):
            _require(_da("kg"))


class TestQuantified:
    def test_converted_when_declared(self):
        out = units.apply_output_units(
            _da("hPa", 1.0).pint.quantify(),
            Unit("Pa", convert=True),
            "r",
            "require",
        )
        assert units.units_equal(f"{out.pint.units:~P}", "Pa")
        np.testing.assert_allclose(out.pint.magnitude, [100.0])

    def test_compatible_raises_without_opting_in(self):
        with pytest.raises(ValueError, match="convert=True"):
            _require(_da("hPa", 1.0).pint.quantify())

    def test_incompatible_raises(self):
        with pytest.raises(pint.DimensionalityError):
            _require(_da("kg").pint.quantify())

    def test_foreign_registry_unit_raises(self):
        # Unparseable here, as an unparseable attrs label is.
        foreign = pint.UnitRegistry(force_ndarray_like=True)
        foreign.define("widget = [thing]")
        da = _da().pint.quantify("widget", unit_registry=foreign)
        with pytest.raises(ValueError, match="unparseable unit 'widget'"):
            _require(da)

    @pytest.mark.parametrize("mode", ["stamp", "strict"])
    def test_foreign_registry_unit_left_untouched_otherwise(self, mode):
        foreign = pint.UnitRegistry(force_ndarray_like=True)
        foreign.define("widget = [thing]")
        da = _da().pint.quantify("widget", unit_registry=foreign)
        assert units.apply_output_units(da, "Pa", "r", mode) is da


class TestNotADataArray:
    @pytest.mark.parametrize(
        ("value", "carries"),
        [(np.array([1.0]), "a ndarray"), (None, "no value"), (1.0, "a float")],
    )
    def test_return_raises(self, value, carries):
        @xra.enforce(units_on_output="require")
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return value

        with pytest.raises(ValueError, match=f"carries {carries}, not a DataArray"):
            wrap()

    def test_marker_override_raises(self):
        @xra.enforce
        def wrap() -> Annotated[xr.DataArray, Unit("Pa", on_output="require")]:
            return np.array([1.0])  # type: ignore[return-value]

        with pytest.raises(ValueError, match="not a DataArray"):
            wrap()

    def test_missing_typeddict_key_raises(self):
        class Out(TypedDict, total=False):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce(units_on_output="require")
        def wrap() -> Out:
            return {}

        with pytest.raises(ValueError, match=r"output 'p' carries no value"):
            wrap()

    def test_none_dataclass_field_raises(self):
        @dataclasses.dataclass
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce(units_on_output="require")
        def wrap() -> Out:
            return Out(p=None)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match=r"output 'p' carries no value"):
            wrap()

    @pytest.mark.parametrize("mode", ["stamp", "strict"])
    def test_other_modes_pass_through(self, mode):
        class Out(TypedDict, total=False):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce(units_on_output=mode)
        def wrap() -> Out:
            return {}

        @xra.enforce(units_on_output=mode)
        def bare() -> Annotated[xr.DataArray, Unit("Pa")]:
            return None  # type: ignore[return-value]

        assert wrap() == {}
        assert bare() is None

    def test_disabled_passes_through(self, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_ENABLED", "0")

        @xra.enforce(units_on_output="require")
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return None  # type: ignore[return-value]

        assert wrap() is None


class TestWiring:
    def test_through_enforce_keyword(self):
        @xra.enforce(units_on_output="require")
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da()

        with pytest.raises(ValueError, match="carries no unit"):
            wrap()

    def test_through_marker_override(self):
        @xra.enforce
        def wrap() -> Annotated[xr.DataArray, Unit("Pa", on_output="require")]:
            return _da()

        with pytest.raises(ValueError, match="carries no unit"):
            wrap()

    def test_wrapper_idiom_converts(self):
        # The #16 fix: say what the library returns, and the declaration is checked.
        @xra.enforce(units_on_output="require")
        def wrap() -> Annotated[xr.DataArray, Unit("Pa", convert=True)]:
            return _da().assign_attrs(units="hPa").pint.quantify()

        np.testing.assert_allclose(wrap().pint.magnitude, [100.0])

    def test_per_module_partial(self):
        enforce = functools.partial(xra.enforce, units_on_output="require")

        @enforce
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da()

        with pytest.raises(ValueError, match="carries no unit"):
            wrap()

    def test_primitive_accepts_require(self):
        with pytest.raises(ValueError, match="carries no unit"):
            units.apply_output_units(_da(), "Pa", "r", "require")

    def test_stamp_and_strict_still_stamp_a_bare_array(self):
        for mode in ("stamp", "strict"):
            assert units.apply_output_units(_da(), "Pa", "r", mode).attrs == {
                "units": "Pa"
            }


class TestPerFieldDeclarationWithoutAContainer:
    class _Out(TypedDict):
        a: Annotated[xr.DataArray, Unit("Pa", on_output="require")]

    @pytest.mark.parametrize("returned", [None, 1.0, _da("Pa")])
    def test_require_raises(self, returned):
        @xra.enforce
        def f() -> TestPerFieldDeclarationWithoutAContainer._Out:
            return returned  # type: ignore[return-value]

        with pytest.raises(ValueError, match="output 'a' carries no value"):
            f()

    def test_other_modes_pass_it_through(self):
        class Out(TypedDict):
            a: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return None  # type: ignore[return-value]

        assert f() is None
