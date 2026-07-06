"""Tests for the top-level ``annotate`` declaration writer.

``annotate`` is the inverse of the ``*_from_signature`` readers: what it writes
onto a function's annotations must read back identically.  The round-trip tests
below assert exactly that, which is the writer's whole contract (it lets a codegen
path stamp a contract onto a generated function and have ``declare_units`` /
``declare_schema`` pick it up unchanged).
"""

from typing import Annotated, get_args, get_origin

import xarray as xr

import xarray_annotated
from xarray_annotated import annotate, unwrap_annotated, walk_signature
from xarray_annotated import _annotations
from xarray_annotated.schema import Coords, Dims, Dtype, schema_from_signature
from xarray_annotated.units import Unit, units_from_signature


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
        hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64", coords=("x",))
        assert get_origin(hint) is Annotated
        base, *markers = get_args(hint)
        assert base is xr.DataArray
        assert markers == [Unit("Pa"), Dims("time", "x"), Dtype("float64"), Coords("x")]

    def test_prebuilt_markers_pass_through(self):
        # An already-built marker is used as-is, not re-wrapped.
        hint = annotate(unit=Unit("degC"), dims=Dims("t", ordered=True))
        assert get_args(hint)[1:] == (Unit("degC"), Dims("t", ordered=True))

    def test_only_given_facets_contribute(self):
        assert get_args(annotate(dtype="int32"))[1:] == (Dtype("int32"),)


class TestTopLevelKernelExports:
    def test_unwrap_annotated_is_the_kernel_helper(self):
        # The domain-agnostic helper is reachable from the package root, not only
        # via a private module (which downstream had to copy verbatim).
        assert unwrap_annotated is _annotations.unwrap_annotated
        assert xarray_annotated.unwrap_annotated is _annotations.unwrap_annotated

    def test_walk_signature_is_the_kernel_driver(self):
        assert walk_signature is _annotations.walk_signature

    def test_unwrap_annotated_sees_base_through_markers(self):
        assert unwrap_annotated(annotate(unit="Pa", dims=("x",))) is xr.DataArray
        assert unwrap_annotated(xr.DataArray) is xr.DataArray


class TestRoundTrip:
    def test_reads_back_through_signature_readers(self):
        hint = annotate(unit="Pa", dims=("time",), dtype="float64", coords=("time",))
        fn = _return_hint(hint)
        assert units_from_signature(fn)[1] == "Pa"
        assert schema_from_signature(fn)[1] == [
            Dims("time"),
            Dtype("float64"),
            Coords("time"),
        ]
