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

import xarray_annotated as xra
from xarray_annotated import Unit, units
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
# Axis defaults and validation
# ---------------------------------------------------------------------------


class TestAxes:
    def test_defaults(self):
        assert _config.DEFAULT_ON_OUTPUT == "stamp"

    def test_on_missing_is_gone(self):
        with pytest.raises(TypeError, match="on_missing"):
            units.check_units(_da([[1.0]], "Pa"), "Pa", "x", on_missing="warn")  # type: ignore[call-arg]

    def test_convert_is_not_an_argument(self):
        with pytest.raises(TypeError, match="convert"):
            units.check_units(_da([[1.0]], "Pa"), "Pa", "x", convert=True)  # type: ignore[call-arg]

    def test_axis_types_are_public(self):
        # The axis types are importable from the public surface (no private
        # _config import needed).
        assert units.OnOutput is _config.OnOutput


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
        out = units.check_units(da, "Pa", "vpd")
        assert out.attrs["units"] == "Pa"
        xr.testing.assert_equal(out["time"], da["time"])
        xr.testing.assert_equal(out["pixel"], da["pixel"])
        np.testing.assert_allclose(out.values, da.values)

    def test_conversion_hpa_to_pa(self):
        da = _da([[10.0, 20.0]], unit="hPa")
        out = units.check_units(da, Unit("Pa", convert=True), "vpd")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_incompatible_raises_dimensionality_error(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.check_units(da, "kg", "x")

    def test_affine_kelvin_to_celsius(self):
        da = _da([[300.0, 273.15]], unit="K")
        out = units.check_units(da, Unit("degC", convert=True), "temperature")
        np.testing.assert_allclose(out.values, [[26.85, 0.0]])

    def test_missing_units_always_raise(self):
        da = _da([[1.0, 2.0]])
        with pytest.raises(ValueError, match="no 'units' attribute"):
            units.check_units(da, "Pa", "vpd")

    @pytest.mark.parametrize(
        "unit", [None, "fraction", "e" * (_check._MAX_UNIT_LEN + 1)]
    )
    def test_every_unreadable_unit_error_gives_the_remedy(self, unit):
        with pytest.raises(ValueError, match=r"xra\.enabled\(False\)"):
            units.check_units(_da([[1.0]], unit=unit), "Pa", "x")

    def test_unparseable_units_error_raises(self):
        # A present-but-unparseable units string (a non-pint spelling) cannot be
        # validated; it is reported clearly rather than letting an opaque pint
        # parse error escape.
        da = _da([[1.0, 2.0]], unit="fraction")
        with pytest.raises(ValueError, match="unparseable 'units' attribute"):
            units.check_units(da, "1", "clay")

    def test_conversion_refused_without_opt_in(self):
        # hPa where Pa is declared would scale the values, so it needs convert=True.
        da = _da([[10.0, 20.0]], unit="hPa")
        with pytest.raises(ValueError, match=r"Unit\('Pa', convert=True\)"):
            units.check_units(da, "Pa", "vpd")

    def test_equivalent_spelling_needs_no_opt_in(self):
        # 'pascal' is the same unit as 'Pa' (no value change), so it is relabelled.
        da = _da([[10.0, 20.0]], unit="pascal")
        out = units.check_units(da, "Pa", "vpd")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])

    def test_incompatible_raises_dimensionality_error_even_when_opted_in(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.check_units(da, Unit("kg", convert=True), "x")

    def test_affine_conversion_needs_opt_in(self):
        # K -> degC is an *affine* (offset) conversion, not just a scale; it still
        # changes the values, so it is refused like hPa/Pa.
        da = _da([[300.0]], unit="K")
        with pytest.raises(ValueError, match="convert=True"):
            units.check_units(da, "degC", "temperature")

    def test_disabled_policy_is_noop(self):
        # enabled=False short-circuits: the array is returned untouched.
        da = _da([[10.0, 20.0]], unit="hPa")
        with xra.enabled(False), warnings.catch_warnings():
            warnings.simplefilter("error")
            out = units.check_units(da, "Pa", "vpd")
        assert out.attrs["units"] == "hPa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])

    def test_over_long_units_attribute_error_raises_without_parsing(self):
        # A units attribute from an untrusted file can be crafted to hang pint's
        # parser (DoS). An over-long one is refused on length, before parsing,
        # which must return quickly.
        da = _da([[1.0, 2.0]], unit="e" * (_check._MAX_UNIT_LEN + 1))
        with pytest.raises(ValueError, match="over-long 'units' attribute"):
            units.check_units(da, "Pa", "vpd")

    def test_over_long_error_message_does_not_echo_the_unit(self):
        # The message must report the length, not the (potentially huge) string.
        over_long = "e" * (_check._MAX_UNIT_LEN + 1)
        da = _da([[1.0, 2.0]], unit=over_long)
        with pytest.raises(ValueError, match="over-long") as excinfo:
            units.check_units(da, "Pa", "vpd")
        assert over_long not in str(excinfo.value)

    def test_max_length_unit_still_parses(self):
        # Boundary: a unit exactly at the cap is not rejected on length; a valid
        # (if verbose) unit padded with harmless factors of 1 still validates.
        unit = "Pa" + " * 1" * 60  # dimensionally Pa, well under the cap
        assert len(unit) <= _check._MAX_UNIT_LEN
        da = _da([[10.0, 20.0]], unit=unit)
        out = units.check_units(da, "Pa", "vpd")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])


class TestCheckUnitsRelabelOnly:
    """An input already in the declared unit must not pay for a conversion.

    ``"pascal"`` where ``"Pa"`` is declared is the same unit spelled differently,
    so there is nothing to convert -- only a string to rewrite. Routing it
    through ``quantify().to().dequantify()`` anyway copies the whole buffer, and
    for a large array that is the dominant cost of declaring a unit it already
    has.
    """

    def test_equivalent_spelling_shares_the_buffer(self):
        da = _da([[1.0, 2.0]], unit="pascal")
        out = units.check_units(da, "Pa", "p")
        assert np.shares_memory(da.values, out.values)
        assert out.attrs["units"] == "Pa"

    def test_identical_spelling_shares_the_buffer(self):
        da = _da([[1.0]], unit="Pa")
        assert np.shares_memory(da.values, units.check_units(da, "Pa", "p").values)

    def test_dimensionless_spellings_are_equivalent(self):
        da = _da([[1.0]], unit="1")
        out = units.check_units(da, "dimensionless", "p")
        assert np.shares_memory(da.values, out.values)
        assert out.attrs["units"] == "dimensionless"

    def test_caller_array_is_left_alone(self):
        # Shallow copy, but its own attrs dict -- restamping must not reach back
        # into the caller's array.
        da = _da([[1.0]], unit="pascal")
        units.check_units(da, "Pa", "p")
        assert da.attrs["units"] == "pascal"

    def test_a_real_conversion_still_copies(self):
        da = _da([[1.0]], unit="hPa")
        out = units.check_units(da, Unit("Pa", convert=True), "p")
        assert not np.shares_memory(da.values, out.values)
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_coordinate_unit_spellings_are_not_rewritten(self):
        """The round-trip used to canonicalise *coordinate* attrs in passing.

        ``dequantify`` writes pint's long form for every quantified variable, so
        a ``"m"`` coordinate came back as ``"meter"`` on a call that was only
        ever about the data variable's own unit.
        """
        da = _da([[1.0]], unit="pascal")
        da.time.attrs["units"] = "m"  # any coord label; spelling is the point
        out = units.check_units(da, "Pa", "p")
        assert out.time.attrs["units"] == "m"


# ---------------------------------------------------------------------------
# Pint-quantified inputs
# ---------------------------------------------------------------------------


def _quantified(values, unit):
    """A DataArray holding a pint Quantity, as pint-xarray users produce."""
    return _da(values, unit=unit).pint.quantify()


class TestCheckUnitsQuantified:
    """A quantified input carries its unit in the data, not in ``attrs``.

    ``quantify()`` *moves* the label off ``attrs`` and into a ``pint.Quantity``,
    so such an array has empty ``attrs`` and is invisible to any check that only
    reads ``attrs["units"]``.
    """

    def test_quantified_input_is_converted(self):
        # Before the fix this returned 1.0 stamped "Pa" -- 100x wrong, and only
        # a "no 'units' attribute" warning to show for it.
        out = units.check_units(
            _quantified([[1.0]], "hPa"), Unit("Pa", convert=True), "p"
        )
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_quantified_input_is_normalised_to_dequantified(self):
        # The declared parameter type is a plain DataArray, so the body gets one:
        # the Quantity is unwrapped and the unit re-stamped onto attrs.
        out = units.check_units(
            _quantified([[1.0]], "hPa"), Unit("Pa", convert=True), "p"
        )
        assert out.pint.units is None

    def test_quantified_input_does_not_warn_about_missing_attrs(self):
        # The old diagnostic was actively misleading: the array does have a unit.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            units.check_units(_quantified([[1.0]], "Pa"), "Pa", "p")

    def test_quantified_input_is_never_missing(self):
        # A missing unit is an *unreadable* one; a Quantity is readable.
        out = units.check_units(
            _quantified([[1.0]], "hPa"), Unit("Pa", convert=True), "p"
        )
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_quantified_input_needs_opt_in(self):
        with pytest.raises(ValueError, match="convert=True"):
            units.check_units(_quantified([[1.0]], "hPa"), "Pa", "p")

    def test_quantified_input_dimensional_mismatch_raises(self):
        with pytest.raises(pint.DimensionalityError):
            units.check_units(_quantified([[1.0]], "degC"), "kg", "p")

    def test_pint_label_wins_over_stale_attrs(self):
        """A leftover ``attrs`` label must not be preferred over the Quantity.

        Reading ``attrs`` first is not merely wrong-in-principle here: pint
        refuses to reconcile the two, so ``quantify()`` raises ``Cannot attach
        units`` and the whole call dies. The Quantity is also the only one of
        the two that cannot have gone stale.
        """
        da = _quantified([[1.0]], "hPa")
        da.attrs["units"] = "kg"  # nonsense left over from upstream
        out = units.check_units(da, Unit("Pa", convert=True), "p")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_unit_from_a_foreign_registry_is_unparseable(self):
        """Units are compared as *strings* re-parsed in the active registry.

        That keeps every comparison in one registry (quantities from two
        registries cannot be mixed), at the price that a unit defined only in
        some other registry reads as unparseable -- which is the correct
        outcome: unverifiable, so it is refused.
        """
        foreign = pint.UnitRegistry(force_ndarray_like=True)
        foreign.define("widget = [thing]")
        da = _da([[1.0]]).pint.quantify("widget", unit_registry=foreign)
        with pytest.raises(ValueError, match="unparseable 'units'"):
            units.check_units(da, "Pa", "p")

    def test_disabled_policy_leaves_quantified_input_alone(self):
        da = _quantified([[1.0]], "hPa")
        with xra.enabled(False):
            assert units.check_units(da, "Pa", "p") is da


# ---------------------------------------------------------------------------
# Dimensional compatibility
# ---------------------------------------------------------------------------


class TestUnitsCompatible:
    def test_public_api(self):
        # units_compatible / units_equal are part of the public units surface.
        assert units.units_compatible is _check.units_compatible
        assert units.units_equal is _check.units_equal

    @pytest.mark.parametrize(("a", "b"), [("Pa", "hPa"), ("1", "dimensionless")])
    def test_compatible(self, a, b):
        assert units.units_compatible(a, b)

    @pytest.mark.parametrize(("a", "b"), [("Pa", "kg"), ("mm", "1")])
    def test_incompatible(self, a, b):
        assert not units.units_compatible(a, b)


class TestUnitsEqual:
    @pytest.mark.parametrize(("a", "b"), [("Pa", "pascal"), ("1", "dimensionless")])
    def test_equal(self, a, b):
        assert units.units_equal(a, b)

    @pytest.mark.parametrize(("a", "b"), [("hPa", "Pa"), ("g", "kg"), ("degC", "K")])
    def test_not_equal(self, a, b):
        # Compatible but value-changing → not equal.
        assert units.units_compatible(a, b)
        assert not units.units_equal(a, b)


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

    def test_enforce_end_to_end_with_marker_input_and_output(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, units.Unit("Pa", convert=True)],
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
