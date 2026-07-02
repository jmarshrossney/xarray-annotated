"""Tests for the `xarray_annotated.schema` stub.

The schema domain is not implemented yet; these tests lock the marker API shape
(so it does not drift before the real implementation lands) and assert that the
validator advertises its unimplemented state loudly rather than silently no-op.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_annotated.schema import Coords, Dims, Dtype, check_schema


class TestMarkers:
    def test_dims_names(self):
        assert Dims("time", "lat", "lon").names == ("time", "lat", "lon")

    def test_coords_names(self):
        assert Coords("time", "lat").names == ("time", "lat")

    def test_dtype_value(self):
        assert Dtype("float64").dtype == "float64"

    def test_repr_roundtrips_eval(self):
        for marker in (Dims("time", "lat"), Coords("time"), Dtype("float64")):
            assert eval(repr(marker)) == marker  # noqa: S307

    def test_equality_and_hash(self):
        assert Dims("time") == Dims("time")
        assert Dims("time") != Dims("lat")
        assert Dims("time") != Coords("time")  # distinct marker types
        assert hash(Dims("time")) == hash(Dims("time"))

    def test_markers_are_hashable_in_annotations(self):
        # A Unit marker stays hashable so Annotated[...] hints can be cached;
        # the schema markers must share that property.
        assert {Dims("time"), Coords("lat"), Dtype("int32")}


class TestCheckSchemaStub:
    def test_raises_not_implemented(self):
        da = xr.DataArray(np.zeros(3), dims="time")
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            check_schema(da, Dims("time"), "x")
