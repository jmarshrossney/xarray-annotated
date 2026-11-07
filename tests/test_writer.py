"""Tests for the top-level ``annotate`` declaration writer.

``annotate`` is the inverse of ``declarations_from_signature``: what it writes
onto a function's annotations must read back identically.  The round-trip tests
below assert exactly that, which is the writer's whole contract (it lets a codegen
path stamp a contract onto a generated function and have ``check_annotations`` pick it
up unchanged).
"""

from typing import Annotated, get_args, get_origin

import xarray as xr

from xarray_annotated import Declared, annotate, declarations_from_signature
from xarray_annotated.schema import Coords, Dims, Dtype
from xarray_annotated.temporal import Freq
from xarray_annotated.units import Unit


def _return_hint(hint):
    """A zero-arg function whose *return* annotation is ``hint``."""

    def fn(): ...

    fn.__annotations__["return"] = hint
    return fn


class TestNoOp:
    def test_no_facets_returns_base_unchanged(self):
        assert annotate() is xr.DataArray

    def test_custom_base_no_facets(self):
        assert annotate(int) is int


class TestMarkers:
    def test_builds_markers_in_fixed_order(self):
        hint = annotate(
            unit="Pa", dims=("time", "x"), dtype="float64", coords=("x",), freq="7D"
        )
        assert get_origin(hint) is Annotated
        base, *markers = get_args(hint)
        assert base is xr.DataArray
        assert markers == [
            Unit("Pa"),
            Dims("time", "x"),
            Dtype("float64"),
            Coords("x"),
            Freq("7D"),
        ]

    def test_prebuilt_markers_pass_through(self):
        # An already-built marker is used as-is, not re-wrapped.
        hint = annotate(
            unit=Unit("degC"),
            dims=Dims("t", ordered=True),
            freq=Freq("W", dim="time"),
        )
        assert get_args(hint)[1:] == (
            Unit("degC"),
            Dims("t", ordered=True),
            Freq("W", dim="time"),
        )

    def test_only_given_facets_contribute(self):
        assert get_args(annotate(dtype="int32"))[1:] == (Dtype("int32"),)
        assert get_args(annotate(freq="ME"))[1:] == (Freq("ME"),)

    def test_freq_marker_keeps_gaps_ok(self):
        marker = Freq("D", gaps_ok=True)
        assert get_args(annotate(freq=marker))[1:] == (marker,)
        assert get_args(annotate(freq=marker))[1] != Freq("D")


class TestRoundTrip:
    def test_reads_back_through_signature_reader(self):
        hint = annotate(
            unit="Pa",
            dims=("time",),
            dtype="float64",
            coords=("time",),
            freq="7D",
        )
        fn = _return_hint(hint)
        assert declarations_from_signature(fn)[1] == Declared(
            unit=Unit("Pa"),
            dims=Dims("time"),
            dtype=Dtype("float64"),
            coords=Coords("time"),
            freq=Freq("7D"),
        )


class TestBareStringNames:
    def test_dims_string_is_one_name(self):
        assert Dims("time") in get_args(annotate(dims="time"))

    def test_coords_string_is_one_name(self):
        assert Coords("time") in get_args(annotate(coords="time"))
