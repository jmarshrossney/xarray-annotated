"""Tests for unit declarations and runtime unit validation (plain-pint registry).

CF/UDUNITS-only cases live in ``test_cf.py``.
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pint
import pytest
import xarray as xr

from xarray_annotated import units
from xarray_annotated.units import _annotations, _check, _config


def _da(values, unit=None):
    """Build a (time, pixel) DataArray, optionally with a units attribute."""
    arr = np.asarray(values, dtype=float)
    time = xr.date_range("2020-01-01", periods=arr.shape[0], freq="7D")
    da = xr.DataArray(
        arr,
        dims=("time", "pixel"),
        coords={"time": time, "pixel": np.arange(arr.shape[1])},
    )
    if unit is not None:
        da.attrs["units"] = unit
    return da


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_defaults(self):
        with units.policy(enabled=None, on_missing=None, on_inexact=None):
            pol = units.get_policy()
            assert pol.enabled is True
            assert pol.on_missing == "warn"
            assert pol.on_inexact == "convert"

    def test_set_single_axis(self):
        with units.policy(on_missing="error"):
            assert units.get_policy().on_missing == "error"

    def test_set_all_axes_at_once(self):
        with units.policy(enabled=False, on_missing="ignore", on_inexact="error"):
            pol = units.get_policy()
            assert pol.enabled is False
            assert pol.on_missing == "ignore"
            assert pol.on_inexact == "error"

    def test_context_manager_restores_on_exit(self):
        with units.policy(on_missing="error"):
            pass
        assert units.get_policy().on_missing == "warn"

    def test_env_overrides_process(self, monkeypatch):
        with units.policy(on_missing="ignore"):
            monkeypatch.setenv(_config.ON_MISSING_ENV_VAR, "error")
            assert units.get_policy().on_missing == "error"

    def test_enabled_env_parses_bool(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "off")
        assert units.get_policy().enabled is False

    def test_invalid_on_missing_raises(self):
        with pytest.raises(ValueError, match="Invalid on_missing"):
            units.set_policy(on_missing="bogus")  # type: ignore[arg-type]

    def test_invalid_on_inexact_raises(self):
        with pytest.raises(ValueError, match="Invalid on_inexact"):
            units.set_policy(on_inexact="bogus")  # type: ignore[arg-type]

    def test_invalid_enabled_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "maybe")
        with pytest.raises(ValueError, match=_config.ENABLED_ENV_VAR):
            units.get_policy()


# ---------------------------------------------------------------------------
# Declared-unit validation (fail fast)
# ---------------------------------------------------------------------------


class TestAssertValidUnit:
    @pytest.mark.parametrize("unit", ["degC", "Pa", "1"])
    def test_valid_units_pass(self, unit):
        units.assert_valid_unit(unit, "ctx")  # no raise

    @pytest.mark.parametrize("unit", ["degrees_C", "not_a_unit", "kg/"])
    def test_invalid_units_raise_with_context(self, unit):
        with pytest.raises(ValueError, match="not a recognised"):
            units.assert_valid_unit(unit, "myctx input 'x'")

    def test_over_long_unit_refused_before_parsing(self):
        # A pathologically long unit string can hang pint's parser (DoS); it is
        # rejected on length *before* being handed to the parser.
        over_long = "e" * (_check._MAX_UNIT_LEN + 1)
        with pytest.raises(ValueError, match="over-long"):
            units.assert_valid_unit(over_long, "ctx")


# ---------------------------------------------------------------------------
# check_units: conversion, round-trip, incompatibility, missing
# ---------------------------------------------------------------------------


class TestCheckUnits:
    def test_round_trip_preserves_coords_and_stamps_declared(self):
        da = _da([[1.0, 2.0], [3.0, 4.0]], unit="Pa")
        out = units.check_units(da, "Pa", "vpd", on_missing="error")
        assert out.attrs["units"] == "Pa"
        xr.testing.assert_equal(out["time"], da["time"])
        xr.testing.assert_equal(out["pixel"], da["pixel"])
        np.testing.assert_allclose(out.values, da.values)

    def test_conversion_hpa_to_pa(self):
        da = _da([[10.0, 20.0]], unit="hPa")
        out = units.check_units(da, "Pa", "vpd", on_missing="error")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_incompatible_raises_dimensionality_error(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.check_units(da, "kg", "x", on_missing="error")

    def test_affine_kelvin_to_celsius(self):
        da = _da([[300.0, 273.15]], unit="K")
        out = units.check_units(da, "degC", "temperature", on_missing="error")
        np.testing.assert_allclose(out.values, [[26.85, 0.0]])

    def test_missing_units_error_raises(self):
        da = _da([[1.0, 2.0]])
        with pytest.raises(ValueError, match="no 'units' attribute"):
            units.check_units(da, "Pa", "vpd", on_missing="error")

    def test_missing_units_warn_warns_and_passes_through(self):
        da = _da([[1.0, 2.0]])
        with pytest.warns(UserWarning, match="unvalidated"):
            out = units.check_units(da, "Pa", "vpd", on_missing="warn")
        assert "units" not in out.attrs
        np.testing.assert_array_equal(out.values, da.values)

    def test_unparseable_units_error_raises(self):
        # A present-but-unparseable units string (a non-pint spelling) cannot be
        # validated; on_missing="error" reports it clearly rather than letting an
        # opaque pint parse error escape.
        da = _da([[1.0, 2.0]], unit="fraction")
        with pytest.raises(ValueError, match="unparseable 'units' attribute"):
            units.check_units(da, "1", "clay", on_missing="error")

    def test_unparseable_units_warn_warns_and_passes_through(self):
        da = _da([[1.0, 2.0]], unit="fraction")
        with pytest.warns(UserWarning, match="unparseable"):
            out = units.check_units(da, "1", "clay", on_missing="warn")
        # Left untouched (its original, un-validatable unit is preserved).
        assert out.attrs["units"] == "fraction"
        np.testing.assert_array_equal(out.values, da.values)

    def test_unparseable_units_ignore_passes_through_silently(self):
        da = _da([[1.0, 2.0]], unit="fraction")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = units.check_units(da, "1", "clay", on_missing="ignore")
        assert out.attrs["units"] == "fraction"
        np.testing.assert_array_equal(out.values, da.values)

    def test_missing_units_ignore_passes_through_silently(self):
        # ignore neither raises (unlike error) nor warns (unlike warn).
        da = _da([[1.0, 2.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = units.check_units(da, "Pa", "vpd", on_missing="ignore")
        assert "units" not in out.attrs
        np.testing.assert_array_equal(out.values, da.values)

    def test_on_inexact_error_forbids_converting_input(self):
        # hPa where Pa is declared: would scale values, so on_inexact="error" raises.
        da = _da([[10.0, 20.0]], unit="hPa")
        with pytest.raises(ValueError, match="on_inexact='error'"):
            units.check_units(da, "Pa", "vpd", on_inexact="error")

    def test_on_inexact_error_accepts_equivalent_spelling(self):
        # 'pascal' is the same unit as 'Pa' (no value change), so it still converts.
        da = _da([[10.0, 20.0]], unit="pascal")
        out = units.check_units(da, "Pa", "vpd", on_inexact="error")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])

    def test_on_inexact_error_still_raises_on_incompatible(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.check_units(da, "kg", "x", on_inexact="error")

    def test_on_inexact_error_forbids_affine_conversion(self):
        # K -> degC is an *affine* (offset) conversion, not just a scale; it still
        # changes the values, so on_inexact="error" must reject it like hPa/Pa.
        da = _da([[300.0]], unit="K")
        with pytest.raises(ValueError, match="on_inexact='error'"):
            units.check_units(da, "degC", "temperature", on_inexact="error")

    def test_on_inexact_warn_converts_with_warning(self):
        # warn converts (unlike error) but announces the value change (unlike convert).
        da = _da([[10.0, 20.0]], unit="hPa")
        with pytest.warns(units.UnitsWarning, match="value-changing"):
            out = units.check_units(da, "Pa", "vpd", on_inexact="warn")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_disabled_policy_is_noop(self):
        # enabled=False short-circuits: the array is returned untouched.
        da = _da([[10.0, 20.0]], unit="hPa")
        with units.policy(enabled=False), warnings.catch_warnings():
            warnings.simplefilter("error")
            out = units.check_units(da, "Pa", "vpd", on_missing="error")
        assert out.attrs["units"] == "hPa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])

    def test_over_long_units_attribute_error_raises_without_parsing(self):
        # A units attribute from an untrusted file can be crafted to hang pint's
        # parser (DoS). An over-long one is refused on length, before parsing,
        # and routed through on_missing (here: error). This must return quickly.
        da = _da([[1.0, 2.0]], unit="e" * (_check._MAX_UNIT_LEN + 1))
        with pytest.raises(ValueError, match="over-long 'units' attribute"):
            units.check_units(da, "Pa", "vpd", on_missing="error")

    def test_over_long_units_attribute_warn_passes_through(self):
        da = _da([[1.0, 2.0]], unit="e" * (_check._MAX_UNIT_LEN + 1))
        with pytest.warns(units.UnitsWarning, match="over-long"):
            out = units.check_units(da, "Pa", "vpd", on_missing="warn")
        # Left untouched: its (un-validatable) unit is preserved, values unchanged.
        assert len(out.attrs["units"]) == _check._MAX_UNIT_LEN + 1
        np.testing.assert_array_equal(out.values, da.values)

    def test_over_long_units_attribute_ignore_passes_through_silently(self):
        da = _da([[1.0, 2.0]], unit="e" * (_check._MAX_UNIT_LEN + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = units.check_units(da, "Pa", "vpd", on_missing="ignore")
        np.testing.assert_array_equal(out.values, da.values)

    def test_over_long_error_message_does_not_echo_the_unit(self):
        # The message must report the length, not the (potentially huge) string.
        over_long = "e" * (_check._MAX_UNIT_LEN + 1)
        da = _da([[1.0, 2.0]], unit=over_long)
        with pytest.raises(ValueError) as excinfo:
            units.check_units(da, "Pa", "vpd", on_missing="error")
        assert over_long not in str(excinfo.value)

    def test_max_length_unit_still_parses(self):
        # Boundary: a unit exactly at the cap is not rejected on length; a valid
        # (if verbose) unit padded with harmless factors of 1 still validates.
        unit = "Pa" + " * 1" * 60  # dimensionally Pa, well under the cap
        assert len(unit) <= _check._MAX_UNIT_LEN
        da = _da([[10.0, 20.0]], unit=unit)
        out = units.check_units(da, "Pa", "vpd", on_missing="error")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])


# ---------------------------------------------------------------------------
# Dimensional compatibility
# ---------------------------------------------------------------------------


class TestUnitsCompatible:
    @pytest.mark.parametrize(("a", "b"), [("Pa", "hPa"), ("1", "dimensionless")])
    def test_compatible(self, a, b):
        assert _check.units_compatible(a, b)

    @pytest.mark.parametrize(("a", "b"), [("Pa", "kg"), ("mm", "1")])
    def test_incompatible(self, a, b):
        assert not _check.units_compatible(a, b)


class TestUnitsEqual:
    @pytest.mark.parametrize(("a", "b"), [("Pa", "pascal"), ("1", "dimensionless")])
    def test_equal(self, a, b):
        assert _check.units_equal(a, b)

    @pytest.mark.parametrize(("a", "b"), [("hPa", "Pa"), ("g", "kg"), ("degC", "K")])
    def test_not_equal(self, a, b):
        # Compatible but value-changing → not equal.
        assert _check.units_compatible(a, b)
        assert not _check.units_equal(a, b)


# ---------------------------------------------------------------------------
# units_from_signature: reading declarations off a function's annotations
# ---------------------------------------------------------------------------


class TestUnitsFromSignature:
    def test_extracts_inputs_and_typeddict_outputs(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            lue: Annotated[xr.DataArray, "g / MJ"]

        def node(
            temp: Annotated[xr.DataArray, "degC"],
            plain: xr.DataArray,
            scalar: int = 3,
        ) -> Out: ...

        inputs, outputs = units.units_from_signature(node)
        # Only Annotated params with a string unit contribute; others are ignored.
        assert inputs == {"temp": "degC"}
        assert outputs == {"gpp": "g / m**2 / d", "lue": "g / MJ"}

    def test_bare_annotated_return(self):
        def node(x: Annotated[xr.DataArray, "1"]) -> Annotated[xr.DataArray, "1"]: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {"x": "1"}
        assert outputs == "1"

    def test_no_annotations(self):
        def node(x: xr.DataArray) -> xr.DataArray: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {}
        assert outputs is None

    def test_partial_typeddict_only_annotated_fields_contribute(self):
        # A node with a mix of unit-carrying and metadata-free outputs: only the
        # annotated fields appear in the declared output units.
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            diagnostic: xr.DataArray  # no unit annotation

        def node() -> Out: ...

        _, outputs = units.units_from_signature(node)
        assert outputs == {"gpp": "g / m**2 / d"}

    def test_extracts_dataclass_outputs(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            lue: Annotated[xr.DataArray, "g / MJ"]

        def node(
            temp: Annotated[xr.DataArray, "degC"],
        ) -> Out: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {"temp": "degC"}
        assert outputs == {"gpp": "g / m**2 / d", "lue": "g / MJ"}

    def test_partial_dataclass_only_annotated_fields_contribute(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            diagnostic: xr.DataArray

        def node() -> Out: ...

        _, outputs = units.units_from_signature(node)
        assert outputs == {"gpp": "g / m**2 / d"}

    def test_dataclass_skipped_when_not_return_annotation(self):
        @dataclass
        class Params:
            temp: Annotated[xr.DataArray, "degC"]

        def node(p: Params) -> xr.DataArray: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {}  # Params is not DataArray, so no input unit
        assert outputs is None  # bare DataArray return has no unit annotation

    def test_metadata_on_non_dataarray_param_is_not_a_unit(self):
        # A descriptive string on a *non-DataArray* parameter is metadata, not a
        # unit: only DataArray annotations carry units. So a config param like a
        # documented flag is ignored.
        def node(flag: Annotated[bool, "toggles X"] = True) -> xr.DataArray: ...

        inputs, _ = units.units_from_signature(node)
        assert inputs == {}

    def test_unit_then_description_takes_unit_first(self):
        # Extra metadata after the unit (e.g. a human-readable description) is
        # ignored: the unit is the first string. This is the supported way to
        # attach both a unit and a description to a parameter.
        def node(
            v: Annotated[xr.DataArray, "m / s", "z component of velocity"],
        ) -> xr.DataArray: ...

        inputs, _ = units.units_from_signature(node)
        assert inputs == {"v": "m / s"}
        units.assert_valid_unit(inputs["v"], "v")  # no raise: description ignored

    def test_non_string_metadata_before_unit_is_skipped(self):
        # Only strings are considered; a non-string marker before the unit string
        # does not shadow it.
        def node(v: Annotated[xr.DataArray, 42, "m / s"]) -> xr.DataArray: ...

        inputs, _ = units.units_from_signature(node)
        assert inputs == {"v": "m / s"}

    def test_description_before_unit_is_misread_and_fails_fast(self):
        # The convention is unit-first. A description placed *before* the unit is
        # mis-read as the unit -- but it fails loudly rather than passing silently
        # (unless the description itself parses as a unit).
        def node(
            v: Annotated[xr.DataArray, "z component of velocity", "m / s"],
        ) -> xr.DataArray: ...

        inputs, _ = units.units_from_signature(node)
        assert inputs == {"v": "z component of velocity"}
        with pytest.raises(ValueError, match="not a recognised"):
            units.assert_valid_unit(inputs["v"], "v")

    def test_unit_on_optional_dataarray_param_is_read(self):
        # An optional DataArray (DataArray | None) still carries its declared unit.
        def node(
            x: Annotated[xr.DataArray | None, "g / m**2"] = None,
        ) -> xr.DataArray: ...

        inputs, _ = units.units_from_signature(node)
        assert inputs == {"x": "g / m**2"}


# ---------------------------------------------------------------------------
# unwrap_annotated: seeing through unit metadata to the base type
# ---------------------------------------------------------------------------


class TestUnwrapAnnotated:
    def test_unwraps_annotated_to_base_type(self):
        assert (
            _annotations.unwrap_annotated(Annotated[xr.DataArray, "degC"])
            is xr.DataArray
        )

    def test_passes_through_plain_types(self):
        assert _annotations.unwrap_annotated(xr.DataArray) is xr.DataArray
        assert _annotations.unwrap_annotated(int) is int


# ---------------------------------------------------------------------------
# Unit typed marker: self-identifying, composable alternative to the bare string
# ---------------------------------------------------------------------------


class TestUnitMarker:
    def test_marker_resolves_to_unit_string(self):
        assert (
            _annotations.annotated_unit(Annotated[xr.DataArray, units.Unit("degC")])
            == "degC"
        )

    def test_marker_and_string_resolve_identically(self):
        marker = _annotations.annotated_unit(Annotated[xr.DataArray, units.Unit("Pa")])
        string = _annotations.annotated_unit(Annotated[xr.DataArray, "Pa"])
        assert marker == string == "Pa"

    def test_marker_wins_over_string_string_first(self):
        # The typed marker owns its slot regardless of ordering.
        assert (
            _annotations.annotated_unit(
                Annotated[xr.DataArray, "degC", units.Unit("Pa")]
            )
            == "Pa"
        )

    def test_marker_wins_over_string_marker_first(self):
        assert (
            _annotations.annotated_unit(
                Annotated[xr.DataArray, units.Unit("Pa"), "degC"]
            )
            == "Pa"
        )

    def test_marker_on_non_dataarray_is_not_a_unit(self):
        # A Unit marker on a non-DataArray param is not a unit (no leak), just as
        # a descriptive string on such a param is not.
        assert _annotations.annotated_unit(Annotated[bool, units.Unit("degC")]) is None

    def test_dataarray_with_no_unit_or_string_metadata_is_none(self):
        # A DataArray annotated only with non-string, non-Unit metadata declares
        # no unit.
        assert _annotations.annotated_unit(Annotated[xr.DataArray, 42]) is None

    def test_marker_round_trips_through_signature_as_str(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, units.Unit("g / m**2 / d")]

        def node(
            temp: Annotated[xr.DataArray, units.Unit("degC")],
        ) -> Out: ...

        inputs, outputs = units.units_from_signature(node)
        # Resolved to plain strings, exactly as the bare-string form.
        assert inputs == {"temp": "degC"}
        assert isinstance(inputs["temp"], str)
        assert outputs == {"gpp": "g / m**2 / d"}

    def test_marker_bare_return_round_trips_as_str(self):
        def node(
            x: Annotated[xr.DataArray, units.Unit("1")],
        ) -> Annotated[xr.DataArray, units.Unit("1")]: ...

        inputs, outputs = units.units_from_signature(node)
        assert inputs == {"x": "1"}
        assert outputs == "1"

    def test_assert_valid_unit_accepts_marker(self):
        units.assert_valid_unit(units.Unit("degC"), "ctx")  # no raise

    def test_assert_valid_unit_rejects_invalid_marker(self):
        with pytest.raises(ValueError, match="not a recognised"):
            units.assert_valid_unit(units.Unit("not_a_unit"), "ctx")

    def test_declare_units_end_to_end_with_marker_input_and_output(self):
        @units.declare_units
        def f(
            p: Annotated[xr.DataArray, units.Unit("Pa")],
        ) -> Annotated[xr.DataArray, units.Unit("Pa")]:
            return p

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_repr(self):
        assert repr(units.Unit("degC")) == "Unit('degC')"

    def test_equality(self):
        assert units.Unit("degC") == units.Unit("degC")
        assert units.Unit("degC") != units.Unit("K")
        # Comparison with a plain string is not equal (NotImplemented -> identity).
        assert units.Unit("degC") != "degC"

    def test_annotation_with_marker_is_hashable(self):
        # Regression guard: a Unit marker must not make the Annotated hint
        # unhashable (would break tools that cache/hash annotations).
        hash(Annotated[xr.DataArray, units.Unit("degC")])
        assert hash(units.Unit("degC")) == hash(units.Unit("degC"))
