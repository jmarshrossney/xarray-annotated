"""Tests for composing declarations with jaxtyping array types.

jaxtyping's ``Float[xr.DataArray, "time x"]`` is an ordinary class with a custom
``__instancecheck__``, not an ``Annotated`` competitor: it occupies the *base-type*
slot, so ``Annotated[Float[xr.DataArray, "time x"], Unit("Pa")]`` is a legal hint that
both sides read. The only thing that made this not work was
``_is_dataarray_type``, which required ``tp is xr.DataArray`` and so silently
dropped every declaration behind a jaxtyping base type.

The check is duck-typed on an ``array_type`` attribute — jaxtyping is never
imported by the package, only by these tests, which skip when it is absent.

Note jaxtyping's dim *names* are positional **size symbols**, not xarray dim names:
``Float[xr.DataArray, "time x"]`` happily accepts an array whose dims are
``("foo", "bar")``. ``Dims`` and jaxtyping are complementary, not redundant, and
``test_jaxtyping_ignores_dim_names`` pins that.
"""

from typing import Annotated

import numpy as np
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import annotate
from xarray_annotated._annotations import _is_dataarray_type
from xarray_annotated.schema import (
    Dims,
    SchemaError,
    schema_from_signature,
)
from xarray_annotated.units import Unit, units_from_signature

jaxtyping = pytest.importorskip("jaxtyping")

Float = jaxtyping.Float

#: The jaxtyping base type, aliased at module level rather than spelled inline in the
#: annotations below.  Its shape argument is a *string*, which pyright reads as a
#: forward reference when it appears inside an annotation ("time is not defined");
#: binding it here puts the subscript in expression position, where it type-checks
#: cleanly.  Purely a static-analysis accommodation — the runtime object is identical.
FloatDA = Float[xr.DataArray, "time x"]


def _da(dims=("time", "x"), dtype="float64", unit=None, fill=1.0):
    shape = tuple(range(2, 2 + len(dims)))
    da = xr.DataArray(np.full(shape, fill, dtype=dtype), dims=dims)
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestIsDataArrayType:
    def test_accepts_jaxtyping_array_type(self):
        assert _is_dataarray_type(Float[xr.DataArray, "time x"])

    def test_rejects_jaxtyping_over_other_array_types(self):
        assert not _is_dataarray_type(Float[np.ndarray, "time x"])

    def test_accepts_optional_jaxtyping_array_type(self):
        assert _is_dataarray_type(Float[xr.DataArray, "time x"] | None)

    def test_plain_dataarray_still_accepted(self):
        assert _is_dataarray_type(xr.DataArray)

    def test_unrelated_type_still_rejected(self):
        assert not _is_dataarray_type(str)

    def test_array_type_attribute_is_not_a_type(self):
        """A stray ``array_type`` attribute must not be mistaken for a wrapper."""

        class Decoy:
            array_type = "xr.DataArray"

        assert not _is_dataarray_type(Decoy)


class TestReaders:
    """Every domain's reader is gated on the same helper, so all three unblock."""

    def test_units_reader_sees_declaration(self):
        def f(x: Annotated[FloatDA, Unit("Pa")]): ...

        assert units_from_signature(f)[0] == {"x": "Pa"}

    def test_schema_reader_sees_declaration(self):
        def f(x: Annotated[FloatDA, Dims("time", "x")]): ...

        assert schema_from_signature(f)[0] == {"x": [Dims("time", "x")]}

    def test_return_declaration_is_read(self):
        def f() -> Annotated[FloatDA, Unit("Pa")]: ...

        assert units_from_signature(f)[1] == "Pa"


class TestDecorators:
    def test_units_are_converted(self):
        @xra.enforce
        def f(
            x: Annotated[FloatDA, Unit("Pa", convert=True)],
        ) -> Annotated[FloatDA, Unit("Pa")]:
            return x

        out = f(_da(unit="hPa", fill=1.0))
        assert out.attrs["units"] == "Pa"
        assert out.values[0, 0] == pytest.approx(100.0)

    def test_schema_mismatch_still_raises(self):
        @xra.enforce
        def f(x: Annotated[FloatDA, Dims("time", "x")]): ...

        with pytest.raises(SchemaError):
            f(_da(dims=("time", "y")))

    def test_jaxtyping_ignores_dim_names(self):
        """The trap: jaxtyping's symbols are sizes, ours are names."""
        hint = Float[xr.DataArray, "time x"]
        assert isinstance(_da(dims=("foo", "bar")), hint)

    def test_jaxtyping_still_checks_dtype(self):
        hint = Float[xr.DataArray, "time x"]
        assert not isinstance(_da(dtype="int64"), hint)


class TestFullStack:
    """The stack documented in ``docs/guides/jaxtyping.md``, pinned end to end.

    ``jaxtyped`` must be **outermost** so it sees the original call, before
    ``enforce`` has converted anything; the size-symbol binding it provides is
    what makes ``"time x"`` mean *the same sizes across arguments*.
    """

    @staticmethod
    def _stack():
        beartype = pytest.importorskip("beartype").beartype

        @jaxtyping.jaxtyped(typechecker=beartype)
        @xra.enforce
        def f(
            x: Annotated[FloatDA, Unit("Pa", convert=True), Dims("time", "x")],
            y: Annotated[FloatDA, Unit("Pa"), Dims("time", "x")],
        ) -> Annotated[FloatDA, Unit("Pa")]:
            return x + y

        return f

    def test_units_converted_under_full_stack(self):
        out = self._stack()(_da(unit="hPa", fill=1.0), _da(unit="Pa", fill=1.0))
        assert out.attrs["units"] == "Pa"

    def test_wrong_dtype_caught_by_jaxtyping(self):
        with pytest.raises(jaxtyping.TypeCheckError):
            self._stack()(_da(dtype="int64", unit="Pa"), _da(unit="Pa"))

    def test_wrong_dim_names_caught_by_schema(self):
        with pytest.raises(SchemaError):
            self._stack()(_da(dims=("time", "y"), unit="Pa"), _da(unit="Pa"))

    def test_inconsistent_sizes_caught_by_jaxtyped(self):
        big = xr.DataArray(np.ones((5, 5)), dims=("time", "x"), attrs={"units": "Pa"})
        with pytest.raises(jaxtyping.TypeCheckError):
            self._stack()(_da(unit="Pa"), big)


class TestWriterRoundTrip:
    """``annotate`` takes a ``base``, so a jaxtyping base type survives a rewrite."""

    def test_round_trip_preserves_jaxtyping_base(self):
        base = Float[xr.DataArray, "time x"]
        hint = annotate(base, unit="Pa", dims=("time", "x"))

        def fn(): ...

        fn.__annotations__["return"] = hint

        assert units_from_signature(fn)[1] == "Pa"
        assert schema_from_signature(fn)[1] == [Dims("time", "x")]

    def test_default_base_is_unchanged(self):
        """The writer never invents a jaxtyping base; the caller passes one."""
        assert annotate() is xr.DataArray
