"""``on_output="require"``: refuse to stamp an output that has no unit of its own (#16).

A wrapper around a library that returns bare arrays declares units it never
computed.  Stamping makes that claim undetectable, since every consumer then
compares one declaration against another.  ``"require"`` makes the wrapper
attach the unit the values really are in, which is then checked and converted.
"""

import functools
from typing import Annotated

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
    def test_converted_as_under_stamp(self):
        out = _require(_da("hPa", 1.0).pint.quantify())
        assert units.units_equal(f"{out.pint.units:~P}", "Pa")
        np.testing.assert_allclose(out.pint.magnitude, [100.0])

    def test_incompatible_raises(self):
        with pytest.raises(pint.DimensionalityError):
            _require(_da("kg").pint.quantify())

    def test_foreign_registry_left_untouched(self):
        foreign = pint.UnitRegistry(force_ndarray_like=True)
        foreign.define("widget = [thing]")
        da = _da().pint.quantify("widget", unit_registry=foreign)
        assert _require(da) is da


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
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da().assign_attrs(units="hPa").pint.quantify()

        np.testing.assert_allclose(wrap().pint.magnitude, [100.0])

    def test_per_module_partial(self):
        enforce = functools.partial(xra.enforce, units_on_output="require")

        @enforce
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da()

        with pytest.raises(ValueError, match="carries no unit"):
            wrap()

    def test_env_var_accepts_require(self, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_UNITS_ON_OUTPUT", "require")
        assert units.get_policy().on_output == "require"
        with pytest.raises(ValueError, match="carries no unit"):
            units.apply_output_units(_da(), "Pa", "r")

    def test_set_policy_accepts_require(self):
        with units.policy(on_output="require"):
            assert units.get_policy().on_output == "require"

    def test_stamp_and_strict_still_stamp_a_bare_array(self):
        for mode in ("stamp", "strict"):
            assert units.apply_output_units(_da(), "Pa", "r", mode).attrs == {
                "units": "Pa"
            }
