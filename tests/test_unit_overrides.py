"""Per-declaration units behaviour: overrides carried on the ``Unit`` marker (#17).

Whether a value-changing conversion is acceptable is a property of a parameter,
not of the process.  A ``Unit`` marker may override each units axis, and the
override beats the decorator keyword and the default; only the package-wide
switch beats it.
"""

from typing import Annotated, TypedDict

import numpy as np
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import annotate, declarations_from_signature, units
from xarray_annotated.units import Unit


def _da(value=1.0, unit: str | None = "hPa") -> xr.DataArray:
    da = xr.DataArray(np.array([value]), dims="x")
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestMarker:
    @pytest.mark.parametrize("axis", ["on_missing", "on_output"])
    def test_invalid_override_rejected_at_construction(self, axis):
        with pytest.raises(ValueError, match=f"Invalid {axis} 'bogus'"):
            Unit("Pa", **{axis: "bogus"})  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_convert_must_be_a_bool(self, value):
        with pytest.raises(TypeError, match="convert must be a bool"):
            Unit("Pa", convert=value)  # type: ignore[arg-type]

    def test_overrides_exposed(self):
        u = Unit("Pa", convert=True, on_missing="error", on_output="strict")
        assert (u.unit, u.convert, u.on_missing, u.on_output) == (
            "Pa",
            True,
            "error",
            "strict",
        )
        assert Unit("Pa").convert is False

    def test_repr_shows_only_set_overrides(self):
        assert repr(Unit("Pa")) == "Unit('Pa')"
        assert repr(Unit("Pa", convert=True)) == "Unit('Pa', convert=True)"
        assert repr(Unit("Pa", convert=False)) == "Unit('Pa')"

    def test_overrides_take_part_in_equality_and_hash(self):
        assert Unit("Pa", convert=True) != Unit("Pa")
        assert Unit("Pa", convert=True) == Unit("Pa", convert=True)
        assert hash(Unit("Pa", on_output="strict")) == hash(
            Unit("Pa", on_output="strict")
        )

    def test_overrides_must_be_keywords(self):
        with pytest.raises(TypeError):
            Unit("Pa", True)  # type: ignore[misc]


class TestThroughEnforce:
    def test_issue_17_pair_in_one_function(self):
        # The motivating case: one parameter must convert, another must refuse,
        # and no single function-wide setting can say both.
        @xra.enforce
        def f(
            tair: Annotated[xr.DataArray, Unit("degC", convert=True)],
            total: Annotated[xr.DataArray, Unit("g / m**2 / year")],
        ) -> None: ...

        f(_da(283.15, "K"), _da(1.0, "g / m**2 / year"))  # tair converts
        with pytest.raises(ValueError, match=r"'total'.*convert=True"):
            f(_da(283.15, "K"), _da(1.0, "g / m**2 / day"))

    def test_converted_value_is_correct(self):
        @xra.enforce
        def f(t: Annotated[xr.DataArray, Unit("degC", convert=True)]):
            return t

        np.testing.assert_allclose(f(_da(283.15, "K")).values, [10.0])

    def test_opt_in_is_per_parameter(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)],
            q: Annotated[xr.DataArray, Unit("Pa")],
        ): ...

        with pytest.raises(ValueError, match="'q'"):
            f(_da(), _da())

    def test_on_missing_override(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", on_missing="error")]): ...

        with pytest.raises(ValueError, match="no 'units' attribute"):
            f(_da(unit=None))

    def test_on_output_override(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa", on_output="strict")]:
            return _da(unit="hPa")

        with pytest.raises(ValueError, match="on_output='strict'"):
            f()

    def test_on_output_override_on_a_typeddict_field(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa", on_output="strict")]
            q: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return {"p": _da(unit="Pa"), "q": _da(unit="hPa")}

        out = f()  # q is only stamped; p matched its strict declaration
        assert out["q"].attrs["units"] == "Pa"

    def test_input_axes_ignored_on_an_output(self):
        # on_missing has no meaning on an output; it must not leak into on_output.
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa", on_missing="error")]:
            return _da(unit="hPa")

        assert f().attrs["units"] == "Pa"  # stamped, as on_output="stamp"

    def test_master_switch_beats_marker(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", on_missing="error")]):
            return p

        with xra.enabled(False):
            assert f(_da(unit=None)).attrs == {}


class TestPrimitivesAcceptTheMarker:
    def test_check_units(self):
        out = units.check_units(_da(), Unit("Pa", convert=True), "p")
        np.testing.assert_allclose(out.values, [100.0])

    def test_check_units_either_marker_or_argument_opts_in(self):
        with pytest.raises(ValueError, match="convert=True"):
            units.check_units(_da(), Unit("Pa"), "p")
        out = units.check_units(_da(), Unit("Pa"), "p", convert=True)
        assert out.attrs["units"] == "Pa"

    def test_check_units_message_uses_the_unit_string(self):
        with pytest.raises(ValueError, match="declared 'Pa'"):
            units.check_units(_da(unit=None), Unit("Pa", on_missing="error"), "p")

    def test_apply_output_units(self):
        with pytest.raises(ValueError, match="on_output='strict'"):
            units.apply_output_units(_da(), Unit("Pa", on_output="strict"), "r")

    def test_apply_output_units_marker_beats_argument(self):
        out = units.apply_output_units(
            _da(), Unit("Pa", on_output="stamp"), "r", on_output="strict"
        )
        assert out.attrs["units"] == "Pa"

    def test_invalid_argument_still_rejected_with_a_marker(self):
        with pytest.raises(ValueError, match="Invalid on_missing"):
            units.check_units(_da(), Unit("Pa"), "p", on_missing="bogus")  # type: ignore[arg-type]


class TestReaders:
    def test_declarations_from_signature_keeps_overrides(self):
        marker = Unit("Pa", convert=True, on_output="strict")

        def f(
            p: Annotated[xr.DataArray, marker],
        ) -> Annotated[xr.DataArray, marker]: ...

        inputs, output = declarations_from_signature(f)
        assert inputs["p"].unit == marker
        assert isinstance(output, xra.Declared)
        assert output.unit == marker

    def test_annotate_round_trip_keeps_overrides(self):
        marker = Unit("Pa", on_missing="error")

        def f(p: annotate(unit=marker)) -> None: ...  # type: ignore[valid-type]

        assert declarations_from_signature(f)[0]["p"].unit == marker

    def test_string_readers_return_the_plain_unit(self):
        hint = Annotated[xr.DataArray, Unit("Pa", convert=True)]
        assert units.annotated_unit(hint) == "Pa"
