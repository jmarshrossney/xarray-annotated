"""Tests for the unified ``declarations_from_signature`` reader.

The all-facets reader is the read-side counterpart to ``annotate``: where the
writer turns facet values into an ``Annotated`` hint, this turns a whole signature
back into one homogeneous ``Declared`` value per declared DataArray.  The tests
below pin the uniform shape (one marker, or ``None``, per facet slot), the
inherited ``TypedDict`` / dataclass / single-return handling, and the exact
round-trip with ``annotate`` that is the reader's whole point.
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

import xarray as xr

from xarray_annotated import Declared, annotate, declarations_from_signature
from xarray_annotated.schema import Coords, Dims, Dtype
from xarray_annotated.units import Unit, units_compatible, units_equal


def _return_hint(hint):
    """A zero-arg function whose *return* annotation is ``hint``."""

    def fn(): ...

    fn.__annotations__["return"] = hint
    return fn


class TestInputs:
    def test_each_facet_and_combination(self):
        def node(
            u: Annotated[xr.DataArray, Unit("Pa")],
            s: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")],
            both: Annotated[xr.DataArray, Unit("degC"), Coords("time")],
            plain: xr.DataArray,
            flag: bool,
        ) -> None: ...

        inputs, output = declarations_from_signature(node)
        assert output is None
        # Only the declaration-bearing params contribute; `plain`/`flag` drop out.
        assert set(inputs) == {"u", "s", "both"}
        assert inputs["u"] == Declared(unit=Unit("Pa"))
        assert inputs["s"] == Declared(dims=Dims("time", "x"), dtype=Dtype("float64"))
        assert inputs["both"] == Declared(unit=Unit("degC"), coords=Coords("time"))

    def test_bare_string_unit_normalises_to_marker(self):
        def node(a: Annotated[xr.DataArray, "Pa"]) -> None: ...

        inputs, _ = declarations_from_signature(node)
        # The bare-string shorthand reads back as a Unit marker, so the shape does
        # not depend on how the unit was spelled.
        assert inputs["a"] == Declared(unit=Unit("Pa"))
        assert isinstance(inputs["a"].unit, Unit)


class TestOutputShapes:
    def test_single_declared_return(self):
        def node() -> Annotated[xr.DataArray, Unit("Pa"), Dims("t")]: ...

        _, output = declarations_from_signature(node)
        assert output == Declared(unit=Unit("Pa"), dims=Dims("t"))

    def test_typeddict_return(self):
        class Out(TypedDict):
            hi: Annotated[xr.DataArray, Unit("Pa")]
            lo: Annotated[xr.DataArray, Dtype("f8")]

        def node() -> Out: ...

        _, output = declarations_from_signature(node)
        assert output == {
            "hi": Declared(unit=Unit("Pa")),
            "lo": Declared(dtype=Dtype("f8")),
        }

    def test_dataclass_return(self):
        @dataclass
        class Out:
            hi: Annotated[xr.DataArray, Dims("time")]
            plain: int

        def node() -> Out: ...

        _, output = declarations_from_signature(node)
        assert output == {"hi": Declared(dims=Dims("time"))}

    def test_undeclared_return_is_none(self):
        def node() -> xr.DataArray: ...

        _, output = declarations_from_signature(node)
        assert output is None


class TestDeclared:
    def test_defaults_are_all_none(self):
        assert Declared() == Declared(None, None, None, None)

    def test_field_order_matches_annotate_kwargs(self):
        # Positional construction follows (unit, dims, dtype, coords).
        d = Declared(Unit("Pa"), Dims("t"), Dtype("f8"), Coords("t"))
        assert (d.unit, d.dims, d.dtype, d.coords) == (
            Unit("Pa"),
            Dims("t"),
            Dtype("f8"),
            Coords("t"),
        )


class TestRoundTripWithAnnotate:
    def test_reader_is_inverse_of_writer(self):
        hint = annotate(unit="Pa", dims=("time",), dtype="float64", coords=("time",))
        _, output = declarations_from_signature(_return_hint(hint))
        assert output == Declared(
            unit=Unit("Pa"),
            dims=Dims("time"),
            dtype=Dtype("float64"),
            coords=Coords("time"),
        )

    def test_declared_rebuilds_its_own_hint(self):
        d = Declared(Unit("hPa"), Dims("x", ordered=True), Dtype("i4"), Coords("x"))
        hint = annotate(unit=d.unit, dims=d.dims, dtype=d.dtype, coords=d.coords)
        _, output = declarations_from_signature(_return_hint(hint))
        assert output == d


class TestUnitStr:
    def test_str_is_bare_unit(self):
        assert str(Unit("Pa")) == "Pa"
        assert f"{Unit('m s-1')}" == "m s-1"

    def test_repr_unchanged(self):
        assert repr(Unit("Pa")) == "Unit('Pa')"


class TestPredicatesAcceptMarkers:
    def test_units_compatible_with_markers(self):
        assert units_compatible(Unit("hPa"), Unit("Pa"))
        assert units_compatible(Unit("hPa"), "Pa")  # mixed
        assert not units_compatible(Unit("Pa"), Unit("kg"))

    def test_units_equal_with_markers(self):
        assert units_equal(Unit("Pa"), Unit("pascal"))
        assert units_equal("Pa", Unit("pascal"))  # mixed
        assert not units_equal(Unit("hPa"), Unit("Pa"))
