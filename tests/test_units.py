"""Tests for unit declarations and runtime unit validation (plain-pint registry).

CF/UDUNITS-only cases live in ``test_cf.py``.
"""

import pickle
import warnings
from typing import Annotated

import numpy as np
import pint
import pytest
import xarray as xr
from xarray import conventions
from xarray.backends import BackendArray
from xarray.core import indexing

import xarray_annotated as xra
from xarray_annotated import Unit, units
from xarray_annotated.units import UnitError, _check


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
    def test_on_missing_is_gone(self):
        with pytest.raises(TypeError, match="on_missing"):
            units.check_units(_da([[1.0]], "Pa"), "Pa", "x", on_missing="warn")  # type: ignore[call-arg]

    def test_convert_is_not_an_argument(self):
        with pytest.raises(TypeError, match="convert"):
            units.check_units(_da([[1.0]], "Pa"), "Pa", "x", convert=True)  # type: ignore[call-arg]


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
        out = units.convert(da, "Pa")
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_incompatible_raises_from_the_pint_error(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(UnitError):
            units.check_units(da, "kg", "x")

    def test_affine_kelvin_to_celsius(self):
        da = _da([[300.0, 273.15]], unit="K")
        out = units.convert(da, "degC")
        np.testing.assert_allclose(out.values, [[26.85, 0.0]])

    def test_missing_units_always_raise(self):
        da = _da([[1.0, 2.0]])
        with pytest.raises(UnitError, match="no 'units' attribute"):
            units.check_units(da, "Pa", "vpd")

    @pytest.mark.parametrize(
        "unit", [None, "fraction", "e" * (_check._MAX_UNIT_LEN + 1)]
    )
    def test_every_unreadable_unit_error_gives_the_remedy(self, unit):
        with pytest.raises(UnitError, match=r"xra\.enabled\(False\)"):
            units.check_units(_da([[1.0]], unit=unit), "Pa", "x")

    def test_unparseable_units_error_raises(self):
        # A present-but-unparseable units string (a non-pint spelling) cannot be
        # validated; it is reported clearly rather than letting an opaque pint
        # parse error escape.
        da = _da([[1.0, 2.0]], unit="fraction")
        with pytest.raises(UnitError, match="unparseable 'units' attribute"):
            units.check_units(da, "1", "clay")

    def test_a_compatible_unit_is_refused(self):
        # hPa where Pa is declared would scale the values, so it is refused.
        da = _da([[10.0, 20.0]], unit="hPa")
        with pytest.raises(UnitError, match=r"units\.convert\(da, 'Pa'\)"):
            units.check_units(da, "Pa", "vpd")

    def test_equivalent_spelling_passes(self):
        # 'pascal' is the same unit as 'Pa' (no value change), so it is relabelled.
        da = _da([[10.0, 20.0]], unit="pascal")
        assert units.check_units(da, "Pa", "vpd") is da

    def test_convert_refuses_an_incompatible_unit(self):
        da = _da([[1.0, 2.0]], unit="degC")
        with pytest.raises(pint.DimensionalityError):
            units.convert(da, "kg")

    def test_convert_refuses_a_malformed_unit(self):
        with pytest.raises(ValueError, match="not a recognised"):
            units.convert(_da([[1.0]], unit="Pa"), "not_a_unit")

    def test_convert_refuses_a_missing_unit(self):
        with pytest.raises(ValueError, match="no 'units' attribute"):
            units.convert(_da([[1.0]]), "Pa")

    def test_convert_accepts_a_marker(self):
        out = units.convert(_da([[1.0]], unit="hPa"), Unit("Pa"))
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_convert_to_the_same_unit_shares_the_buffer(self):
        da = _da([[1.0]], unit="pascal")
        out = units.convert(da, "Pa")
        assert np.shares_memory(da.values, out.values)
        assert out.attrs["units"] == "Pa"

    def test_convert_is_not_a_check_so_the_switch_does_not_affect_it(self):
        with xra.enabled(False):
            out = units.convert(_da([[1.0]], unit="hPa"), "Pa")
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_affine_conversion_is_refused(self):
        # K -> degC is an *affine* (offset) conversion, not just a scale; it still
        # changes the values, so it is refused like hPa/Pa.
        da = _da([[300.0]], unit="K")
        with pytest.raises(UnitError, match=r"units\.convert"):
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
        with pytest.raises(UnitError, match="over-long 'units' attribute"):
            units.check_units(da, "Pa", "vpd")

    def test_over_long_error_message_does_not_echo_the_unit(self):
        # The message must report the length, not the (potentially huge) string.
        over_long = "e" * (_check._MAX_UNIT_LEN + 1)
        da = _da([[1.0, 2.0]], unit=over_long)
        with pytest.raises(UnitError, match="over-long") as excinfo:
            units.check_units(da, "Pa", "vpd")
        assert over_long not in str(excinfo.value)

    def test_max_length_unit_still_parses(self):
        # Boundary: a unit exactly at the cap is not rejected on length; a valid
        # (if verbose) unit padded with harmless factors of 1 still validates.
        unit = "Pa" + " * 1" * 60  # dimensionally Pa, well under the cap
        assert len(unit) <= _check._MAX_UNIT_LEN
        da = _da([[10.0, 20.0]], unit=unit)
        assert units.check_units(da, "Pa", "vpd") is da


class TestCheckUnitsReturnsItsInput:
    """A check costs nothing and changes nothing: it hands back the same object."""

    @pytest.mark.parametrize(
        ("label", "declared"),
        [("pascal", "Pa"), ("Pa", "Pa"), ("1", "dimensionless")],
    )
    def test_same_object(self, label, declared):
        da = _da([[1.0, 2.0]], unit=label)
        assert units.check_units(da, declared, "p") is da
        assert da.attrs["units"] == label

    def test_a_real_conversion_still_copies(self):
        da = _da([[1.0]], unit="hPa")
        out = units.convert(da, "Pa")
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

    def test_quantified_input_is_converted_and_stays_quantified(self):
        out = units.convert(_quantified([[1.0]], "hPa"), "Pa")
        assert out.pint.units == units.get_registry().Unit("Pa")
        assert "units" not in out.attrs
        np.testing.assert_allclose(out.data.magnitude, [[100.0]])

    def test_a_plain_input_stays_plain(self):
        out = units.convert(_da([[1.0]], unit="hPa"), "Pa")
        assert out.pint.units is None
        assert out.attrs["units"] == "Pa"

    def test_a_quantified_coord_stays_quantified(self):
        da = xr.DataArray(
            [1.0],
            dims="depth",
            coords={"depth": ("depth", [10.0], {"units": "m"})},
            attrs={"units": "hPa"},
        ).pint.quantify()
        out = units.convert(da, "Pa")
        assert out.depth.pint.units == units.get_registry().Unit("m")

    def test_quantified_incompatible_unit_raises_dimensionality_error(self):
        with pytest.raises(pint.DimensionalityError):
            units.convert(_quantified([[1.0]], "hPa"), "kg")

    def test_quantified_input_in_the_declared_unit_stays_quantified(self):
        da = _quantified([[1.0]], "pascal")
        assert units.check_units(da, "Pa", "p") is da

    def test_quantified_input_does_not_warn_about_missing_attrs(self):
        # The old diagnostic was actively misleading: the array does have a unit.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            units.check_units(_quantified([[1.0]], "Pa"), "Pa", "p")

    def test_quantified_input_is_never_missing(self):
        # A missing unit is an *unreadable* one; a Quantity is readable.
        out = units.convert(_quantified([[1.0]], "hPa"), "Pa")
        np.testing.assert_allclose(out.data.magnitude, [[100.0]])

    def test_quantified_input_is_refused_in_another_unit(self):
        with pytest.raises(UnitError, match=r"units\.convert"):
            units.check_units(_quantified([[1.0]], "hPa"), "Pa", "p")

    def test_quantified_input_dimensional_mismatch_raises(self):
        with pytest.raises(UnitError):
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
        out = units.convert(da, "Pa")
        assert "units" not in out.attrs
        np.testing.assert_allclose(out.data.magnitude, [[100.0]])

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
        with pytest.raises(UnitError, match="unparseable 'units'"):
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
# Unit typed marker: the one way to declare a unit
# ---------------------------------------------------------------------------


class TestUnitMarker:
    def test_marker_is_read_regardless_of_position(self):
        for hint in (
            Annotated[xr.DataArray, "surface pressure", units.Unit("Pa")],
            Annotated[xr.DataArray, units.Unit("Pa"), "surface pressure"],
            Annotated[xr.DataArray, 42, units.Unit("Pa")],
        ):
            declared = xra.declaration_from_hint(hint)
            assert declared is not None
            assert declared.unit == units.Unit("Pa")

    def test_marker_on_non_dataarray_is_not_a_unit(self):
        hint = Annotated[bool, units.Unit("degC")]
        assert xra.declaration_from_hint(hint) is None

    def test_dataarray_with_no_unit_or_string_metadata_is_none(self):
        assert xra.declaration_from_hint(Annotated[xr.DataArray, 42]) is None

    def test_assert_valid_unit_accepts_marker(self):
        units.assert_valid_unit(units.Unit("degC"), "ctx")  # no raise

    def test_assert_valid_unit_rejects_invalid_marker(self):
        with pytest.raises(ValueError, match="not a recognised"):
            units.assert_valid_unit(units.Unit("not_a_unit"), "ctx")

    def test_enforce_end_to_end_with_marker_input_and_output(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, units.Unit("Pa")],
        ) -> Annotated[xr.DataArray, units.Unit("Pa")]:
            return p

        out = f(_da([[10.0, 20.0]], unit="pascal"))
        assert out.attrs["units"] == "pascal"
        np.testing.assert_allclose(out.values, [[10.0, 20.0]])

    def test_repr(self):
        assert repr(units.Unit("degC")) == "Unit('degC')"

    @pytest.mark.parametrize("setting", ["convert", "on_output", "on_missing"])
    def test_has_no_settings(self, setting):
        with pytest.raises(TypeError, match=setting):
            Unit("Pa", **{setting: True})  # type: ignore[arg-type]

    def test_check_units_accepts_a_marker(self):
        with pytest.raises(UnitError, match="declared 'Pa'"):
            units.check_units(_da([[1.0]]), Unit("Pa"), "p")
        da = _da([[1.0]], unit="pascal")
        assert units.check_output_units(da, Unit("Pa"), "r") is da

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


class TestConversionLeavesCoordsAlone:
    def _da(self):
        return xr.DataArray(
            [1.0, 2.0],
            dims="lat",
            coords={
                "lat": ("lat", [10.0, 20.0], {"units": "degrees_north"}),
                "height": ("lat", [1.0, 2.0], {"units": "metre"}),
            },
            attrs={"units": "hPa", "long_name": "pressure"},
        )

    def test_unparseable_coord_label_does_not_block_conversion(self):
        out = units.convert(self._da(), "Pa")
        np.testing.assert_allclose(out.values, [100.0, 200.0])
        assert out.attrs == {"units": "Pa", "long_name": "pressure"}

    def test_coord_labels_keep_their_spelling(self):
        out = units.convert(self._da(), "Pa")
        assert out.lat.attrs["units"] == "degrees_north"
        assert out.height.attrs["units"] == "metre"
        np.testing.assert_allclose(out.lat.values, [10.0, 20.0])

    def test_input_is_not_modified(self):
        da = self._da()
        units.convert(da, "Pa")
        assert da.attrs["units"] == "hPa"
        np.testing.assert_allclose(da.values, [1.0, 2.0])

    def test_dimensional_mismatch_is_still_named(self):
        da = xr.DataArray([1.0], attrs={"units": "kg"})
        with pytest.raises(pint.DimensionalityError) as info:
            units.convert(da, "Pa")
        assert "while converting" in info.value.__notes__[0]


class TestUnitsEqualIsPhysical:
    @pytest.mark.parametrize(
        ("a", "b"),
        [("Pa", "N/m**2"), ("Pa", "kg/m/s**2"), ("Hz", "1/s"), ("delta_degC", "K")],
    )
    def test_same_scale_different_composition_is_equal(self, a, b):
        assert units.units_equal(a, b)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Pa", "hPa"),
            ("degC", "K"),  # same size of degree, different zero
            ("degC", "delta_degC"),  # compatible, but pint refuses to convert
            ("percent", "1"),
            ("dB", "1"),
            ("Pa", "kg"),
        ],
    )
    def test_different_values_or_dimensions_are_not_equal(self, a, b):
        assert not units.units_equal(a, b)

    def test_recomposed_input_passes(self):
        da = xr.DataArray([1.0], attrs={"units": "kg/m/s**2"})
        assert units.check_units(da, "Pa", "p") is da

    def test_recomposed_output_label_is_no_mismatch(self):
        da = xr.DataArray([1.0], attrs={"units": "N/m**2"})
        assert units.check_output_units(da, "Pa", "return") is da


_PACKING = {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -32768, "zlib": True}


def _packed(unit="hPa"):
    time = xr.date_range("2020-01-01", periods=2, freq="D")
    da = xr.DataArray(
        [1000.0, 1010.0], dims="time", coords={"time": time}, attrs={"units": unit}
    )
    da.encoding.update(_PACKING)
    da.time.encoding["units"] = "days since 2000-01-01"
    return da


class TestConversionClearsEncoding:
    """Packing describes the old values, so keeping it corrupts the array on write."""

    def test_converted_input(self):
        da = _packed()
        out = units.convert(da, "Pa")
        assert out.encoding == {}
        assert da.encoding == _PACKING

    def test_converted_quantified_input(self):
        da = _packed().pint.quantify()
        out = units.convert(da, "Pa")
        assert out.encoding == {}
        assert da.encoding == _PACKING

    def test_same_unit_clears_a_stale_packing_on_a_quantified_array(self):
        da = _packed().pint.quantify().pint.to("Pa")
        out = units.convert(da, "Pa")
        assert out.encoding == {}
        assert np.shares_memory(out.data.magnitude, da.data.magnitude)
        assert da.encoding == _PACKING

    def test_coords_keep_theirs(self):
        out = units.convert(_packed(), "Pa")
        assert out.time.encoding == {"units": "days since 2000-01-01"}

    def test_same_unit_keeps_it(self):
        out = units.check_units(_packed("hPa"), "hectopascal", "p")
        assert out.encoding == _PACKING

    def test_a_check_leaves_a_stale_packing_for_the_writer_to_clear(self):
        da = _packed().pint.quantify().pint.to("Pa")
        assert da.encoding == _PACKING  # what pint-xarray leaves behind
        assert units.check_units(da, "Pa", "p").encoding == _PACKING
        assert units.check_output_units(da, "Pa", "return") is da


class _CountingBackend(BackendArray):
    """A lazily read array that counts how often the backend is read."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads = 0

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._read
        )

    def _read(self, key):
        self.reads += 1
        return self.values[key]


def _lazy(cache, unit="Pa"):
    backend = _CountingBackend([1.0, 2.0, 3.0])
    data = indexing.CopyOnWriteArray(
        indexing.LazilyIndexedArray(backend)  # pyright: ignore[reportArgumentType]
    )
    if cache:
        data = indexing.MemoryCachedArray(data)
    variable = xr.Variable(("x",), data, attrs={"units": unit})
    return xr.DataArray(variable), backend


@pytest.mark.parametrize("cache", [True, False], ids=["cache", "no-cache"])
class TestLazyArrays:
    """Asking whether an array is quantified must not read it from disk."""

    def test_check_units_reads_nothing(self, cache):
        da, backend = _lazy(cache)
        units.check_units(da, "Pa", "p")
        assert backend.reads == 0

    def test_check_output_units_reads_nothing(self, cache):
        da, backend = _lazy(cache)
        units.check_output_units(da, "Pa", "p")
        assert backend.reads == 0

    def test_enforce_reads_nothing_and_stays_lazy(self, cache):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        da, backend = _lazy(cache)
        out = f(da)
        assert backend.reads == 0
        assert not out.variable._in_memory
        assert not da.variable._in_memory

    def test_converting_reads_the_values(self, cache):
        da, backend = _lazy(cache, "hPa")
        out = units.convert(da, "Pa")
        assert backend.reads == 1
        np.testing.assert_allclose(out.values, [100.0, 200.0, 300.0])

    def test_finds_a_quantified_unit(self, cache):
        quantified = _quantified([[1.0]], "hPa")
        assert _check._quantified_units(quantified) == quantified.pint.units
        da, _ = _lazy(cache)
        assert _check._quantified_units(da) is None


class TestQuantify:
    def _with_lat(self):
        return xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="lat",
            coords={"lat": ("lat", [10.0, 20.0, 30.0], {"units": "degree"})},
            attrs={"units": "Pa", "long_name": "pressure"},
        )

    def test_moves_the_label_into_the_data(self):
        da = self._with_lat()
        out = units.quantify(da)
        assert out.pint.units == units.get_registry().Unit("Pa")
        assert out.attrs == {"long_name": "pressure"}
        assert da.attrs["units"] == "Pa"
        assert da.pint.units is None

    def test_shares_the_buffer(self):
        da = self._with_lat()
        assert np.shares_memory(units.quantify(da).data.magnitude, da.values)

    def test_a_quantified_array_comes_back_as_itself(self):
        da = _quantified([[1.0]], "Pa")
        assert units.quantify(da) is da

    def test_coords_keep_their_label_and_stay_plain(self):
        out = units.quantify(self._with_lat())
        assert out.lat.pint.units is None
        assert out.lat.attrs == {"units": "degree"}

    def test_a_coord_the_registry_cannot_parse_does_not_stop_it(self):
        da = self._with_lat().assign_coords(
            lat=("lat", [10.0, 20.0, 30.0], {"units": "not_a_unit"})
        )
        assert units.quantify(da).pint.units is not None

    def test_selecting_on_an_angle_coord_finds_the_right_row(self):
        out = units.quantify(self._with_lat())
        assert out.sel(lat=20.0, method="nearest").lat.item() == 20.0
        assert out.sel(lat=slice(10.0, 20.0)).sizes["lat"] == 2

    def test_pint_xarrays_own_quantify_expects_quantity_labels(self):
        # Pint-xarray's rule, which the docs warn about: a bare label is dimensionless.
        out = self._with_lat().pint.quantify()
        assert out.sel(lat=slice(10.0, 20.0)).sizes["lat"] == 0
        degree = units.get_registry().Unit("degree")
        assert out.sel(lat=slice(10.0 * degree, 20.0 * degree)).sizes["lat"] == 2

    def test_conversion_and_write_labels_work_on_it(self):
        out = units.quantify(self._with_lat()).pint.to("hPa")
        np.testing.assert_allclose(out.data.magnitude, [0.01, 0.02, 0.03])
        assert out.pint.dequantify().lat.attrs == {"units": "degree"}

    def test_missing_label_raises(self):
        with pytest.raises(ValueError, match="no 'units' attribute"):
            units.quantify(xr.DataArray([1.0]))

    def test_unparseable_label_raises(self):
        with pytest.raises(ValueError, match="unparseable"):
            units.quantify(xr.DataArray([1.0], attrs={"units": "not_a_unit"}))

    def test_over_long_label_is_refused_before_parsing(self):
        with pytest.raises(ValueError, match="over-long"):
            units.quantify(xr.DataArray([1.0], attrs={"units": "x" * 300}))

    def test_ignores_the_switch(self):
        with xra.enabled(False):
            assert units.quantify(_da([[1.0]], unit="Pa")).pint.units is not None

    def test_keeps_encoding(self):
        out = units.quantify(_packed())
        assert out.encoding == _PACKING


class TestDequantify:
    def _quantified_packed(self):
        return _packed().pint.quantify().pint.to("Pa")

    def test_a_plain_array_comes_back_as_itself(self):
        da = _packed()
        assert units.dequantify(da) is da
        assert da.encoding == _PACKING

    def test_moves_the_unit_into_the_label(self):
        out = units.dequantify(
            xr.DataArray([1.0], attrs={"units": "Pa"}).pint.quantify()
        )
        assert out.pint.units is None
        assert units.units_equal(out.attrs["units"], "Pa")

    def test_clears_a_stale_packing_on_a_copy(self):
        da = self._quantified_packed()
        assert units.dequantify(da).encoding == {}
        assert da.encoding == _PACKING

    def test_coords_keep_their_encoding(self):
        out = units.dequantify(self._quantified_packed())
        assert out.time.encoding == {"units": "days since 2000-01-01"}

    def test_values_survive_being_encoded(self):
        out = units.dequantify(self._quantified_packed())
        encoded = conventions.encode_cf_variable(out.variable)
        decoded = conventions.decode_cf_variable("p", encoded)
        np.testing.assert_allclose(decoded.values, [100000.0, 101000.0])

    def test_round_trips_with_quantify_leaving_coords_alone(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="lat",
            coords={"lat": ("lat", [10.0, 20.0], {"units": "degree"})},
            attrs={"units": "Pa", "long_name": "pressure"},
        )
        out = units.dequantify(units.quantify(da))
        np.testing.assert_array_equal(out.values, da.values)
        assert out.attrs["long_name"] == "pressure"
        assert out.lat.attrs == {"units": "degree"}

    def test_replaces_a_leftover_label(self):
        q = xr.DataArray([1.0], attrs={"units": "Pa"}).pint.quantify()
        out = units.dequantify(q.assign_attrs(units="degC"))
        assert units.units_equal(out.attrs["units"], "Pa")

    def test_ignores_the_switch(self):
        q = xr.DataArray([1.0], attrs={"units": "Pa"}).pint.quantify()
        with xra.enabled(False):
            assert units.dequantify(q).pint.units is None

    @pytest.mark.parametrize(
        "label", ["hPa", "degC", "m**2", "kg*m/s**2", "percent", "dimensionless"]
    )
    def test_the_default_registry_reads_its_label_back(self, label):
        q = xr.DataArray([1.0], attrs={"units": label}).pint.quantify()
        assert units.units_equal(units.dequantify(q).attrs["units"], label)

    def test_a_cf_import_does_not_change_the_default_registrys_spelling(self):
        pytest.importorskip("cf_xarray.units")
        q = xr.DataArray([1.0], attrs={"units": "m**2"}).pint.quantify()
        assert units.dequantify(q).attrs["units"] == "meter ** 2"


class TestUnitError:
    def test_is_a_contract_error_and_neither_value_nor_type_error(self):
        assert issubclass(UnitError, xra.ContractError)
        assert not issubclass(UnitError, ValueError | TypeError)

    def test_incompatible_units_keep_the_pint_error_as_the_cause(self):
        with pytest.raises(UnitError) as info:
            units.check_units(_da([[1.0]], unit="kg"), "Pa", "p", qualname="f")
        assert isinstance(info.value.__cause__, pint.DimensionalityError)

    @pytest.mark.parametrize(
        "check", [units.check_units, units.check_output_units], ids=lambda c: c.__name__
    )
    def test_pickling_keeps_the_whole_message(self, check):
        with pytest.raises(UnitError) as info:
            check(_da([[1.0]], unit="kg"), "Pa", "p", qualname="f")
        restored = pickle.loads(pickle.dumps(info.value))
        assert type(restored) is UnitError
        assert str(restored) == str(info.value)
        for part in ("[f]", "'p'", "'kg'", "'Pa'", "[mass]", "[length]"):
            assert part in str(restored)

    @pytest.mark.parametrize(
        "check", [units.check_units, units.check_output_units], ids=lambda c: c.__name__
    )
    @pytest.mark.parametrize("declared", ["not_a_unit", "e" * 257, Unit("not_a_unit")])
    def test_a_malformed_declared_unit_is_a_value_error(self, check, declared):
        with pytest.raises(ValueError, match="declared unit"):
            check(_da([[1.0]], unit="Pa"), declared, "p")

    def test_a_malformed_declared_unit_is_ignored_when_switched_off(self):
        da = _da([[1.0]], unit="Pa")
        with xra.enabled(False):
            assert units.check_units(da, "not_a_unit", "p") is da

    @pytest.mark.parametrize(
        "operation", [units.quantify, lambda d: units.convert(d, "Pa")]
    )
    def test_operations_keep_their_value_error(self, operation):
        with pytest.raises(ValueError, match="no 'units' attribute"):
            operation(_da([[1.0]]))

    def test_convert_keeps_its_pint_error(self):
        with pytest.raises(pint.DimensionalityError):
            units.convert(_da([[1.0]], unit="kg"), "Pa")
