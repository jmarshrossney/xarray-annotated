"""Tests for the `xarray_annotated.schema` structural-validation domain.

Covers the markers (with their options), the annotation reader, the per-property
checkers, the policy, and `enforce` over schema markers.  Structural validation
never mutates, so the checks assert/raise/warn but return arrays unchanged.
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import schema
from xarray_annotated.schema import Coords, Dims, Dtype, _check


def _da(dims=("time", "x"), dtype="float64", coords=()):
    """Build a DataArray with the given dims/dtype; attach coords for `coords` dims."""
    shape = tuple(range(2, 2 + len(dims)))
    arr = np.zeros(shape, dtype=dtype)
    coord_map = {
        d: np.arange(s) for d, s in zip(dims, shape, strict=True) if d in coords
    }
    return xr.DataArray(arr, dims=dims, coords=coord_map)


def _run(da, marker):
    """Run check_schema for one marker."""
    return schema.check_schema(da, marker, "x")


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


class TestMarkers:
    def test_accessors(self):
        d = Dims("time", "x")
        assert d.names == ("time", "x")
        assert d.ordered is False
        assert Dims("time", ordered=True).ordered is True
        assert Coords("lat").names == ("lat",)
        assert Dtype("float64").dtype == "float64"

    def test_repr_roundtrips_eval(self):
        markers = [
            Dims("time", "x"),
            Dims("time", ordered=True),
            Dims("time", ordered=True),
            Coords("time", "lat"),
            Coords("time", "x"),
            Dtype("float64"),
            Dtype("float"),
        ]
        for marker in markers:
            assert eval(repr(marker)) == marker

    def test_equality_and_hash_include_options(self):
        assert Dims("time") == Dims("time")
        assert Dims("time") != Dims("time", ordered=True)
        assert Dtype("f8") != Dtype("float")
        assert Dims("time") != Coords("time")  # distinct marker types
        assert Dims("time") != "time"  # not equal to a bare string
        assert Dtype("f8") != "f8"
        assert hash(Dims("time", ordered=True)) == hash(Dims("time", ordered=True))

    def test_markers_hashable_in_annotations(self):
        assert {Dims("time"), Coords("lat"), Dtype("int32")}


# ---------------------------------------------------------------------------
# Per-property checkers (via check_schema)
# ---------------------------------------------------------------------------


class TestCheckDims:
    def test_set_match_passes_regardless_of_order(self):
        _run(_da(("time", "x")), Dims("time", "x"))
        _run(_da(("x", "time")), Dims("time", "x"))  # order-insensitive

    def test_extra_or_missing_dim_fails(self):
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            _run(_da(("time", "x", "z")), Dims("time", "x"))
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            _run(_da(("time",)), Dims("time", "x"))

    def test_ordered_requires_order(self):
        _run(_da(("time", "x")), Dims("time", "x", ordered=True))
        with pytest.raises(schema.SchemaError, match="dims order mismatch"):
            _run(_da(("x", "time")), Dims("time", "x", ordered=True))

    def test_scalar_dims(self):
        _run(_da(()), Dims())


class TestCheckCoords:
    def test_present_passes_extras_ok(self):
        _run(_da(("time", "x"), coords=("time", "x")), Coords("time"))

    def test_missing_coord_fails(self):
        # dims present but no coordinate labels attached
        with pytest.raises(schema.SchemaError, match="missing coords"):
            _run(_da(("time", "x")), Coords("time"))


class TestCheckDtype:
    def test_a_width_means_that_width(self):
        _run(_da(dtype="float64"), Dtype("float64"))
        with pytest.raises(schema.SchemaError, match="expected float64, got float32"):
            _run(_da(dtype="float32"), Dtype("float64"))

    def test_a_width_mismatch_names_the_marker_that_accepts_it(self):
        with pytest.raises(schema.SchemaError, match=r"Dtype\('float'\) would accept"):
            _run(_da(dtype="float32"), Dtype("float64"))

    def test_a_kind_mismatch_suggests_no_marker(self):
        with pytest.raises(schema.SchemaError) as info:
            _run(_da(dtype="int64"), Dtype("float64"))
        assert "would accept" not in str(info.value)

    @pytest.mark.parametrize(
        ("declared", "actual"),
        [
            ("float", "float32"),
            ("float", "float64"),
            ("int", "int8"),
            ("int", "int64"),
            ("uint", "uint8"),
            ("complex", "complex64"),
            ("str", "U5"),
            ("bytes", "S3"),
            ("datetime64", "datetime64[ns]"),
            ("timedelta64", "timedelta64[s]"),
        ],
    )
    def test_a_name_with_no_width_means_any_of_that_kind(self, declared, actual):
        _run(_da(dtype=actual), Dtype(declared))

    @pytest.mark.parametrize(
        ("declared", "actual"),
        [("float", "int64"), ("int", "uint8"), ("uint", "int8"), ("int", "bool")],
    )
    def test_any_of_a_kind_is_still_a_kind(self, declared, actual):
        with pytest.raises(schema.SchemaError, match=f"expected any {declared}"):
            _run(_da(dtype=actual), Dtype(declared))

    @pytest.mark.parametrize("actual", [">f8", "<f8"])
    def test_byte_order_is_not_part_of_the_dtype(self, actual):
        # NetCDF-3 files are big-endian.
        _run(_da(dtype=actual), Dtype("float64"))
        _run(_da(dtype=actual), Dtype("float"))
        with pytest.raises(schema.SchemaError):
            _run(_da(dtype=actual), Dtype("float32"))

    def test_exact_is_gone(self):
        with pytest.raises(TypeError, match="exact"):
            Dtype("float64", exact=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Marker-vs-marker compatibility (no array in hand)
# ---------------------------------------------------------------------------


class TestDimsCompatible:
    def test_public_api(self):
        assert schema.dims_compatible is _check.dims_compatible

    def test_same_set_any_order(self):
        assert schema.dims_compatible(Dims("x", "y"), Dims("y", "x"))

    def test_different_set(self):
        assert not schema.dims_compatible(Dims("x"), Dims("x", "y"))

    def test_both_ordered_order_differs(self):
        a = Dims("x", "y", ordered=True)
        b = Dims("y", "x", ordered=True)
        assert not schema.dims_compatible(a, b)

    def test_one_side_loose_order(self):
        # Only one side pins order → not provably inconsistent.
        assert schema.dims_compatible(Dims("x", "y", ordered=True), Dims("y", "x"))


class TestDtypeCompatible:
    def test_public_api(self):
        assert schema.dtype_compatible is _check.dtype_compatible

    def test_same_kind_different_width(self):
        assert not schema.dtype_compatible(Dtype("float64"), Dtype("float32"))

    def test_same_dtype_any_spelling(self):
        assert schema.dtype_compatible(Dtype("float64"), Dtype("f8"))
        assert schema.dtype_compatible(Dtype(">f8"), Dtype("f8"))

    def test_different_kind(self):
        assert not schema.dtype_compatible(Dtype("float64"), Dtype("int64"))

    def test_a_kind_is_compatible_with_any_width_of_it(self):
        assert schema.dtype_compatible(Dtype("float"), Dtype("float32"))
        assert schema.dtype_compatible(Dtype("float32"), Dtype("float"))
        assert schema.dtype_compatible(Dtype("float"), Dtype("float"))

    def test_a_kind_is_not_compatible_with_another(self):
        assert not schema.dtype_compatible(Dtype("float"), Dtype("int32"))
        assert not schema.dtype_compatible(Dtype("int"), Dtype("uint"))


# ---------------------------------------------------------------------------
# check_schema router behaviour
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_returns_array_unchanged(self):
        da = _da(("time", "x"))
        assert schema.check_schema(da, Dims("time", "x"), "x") is da

    def test_qualname_prefix_in_message(self):
        with pytest.raises(schema.SchemaError, match=r"\[myfunc\] 'x'"):
            schema.check_schema(
                _da(("time",)),
                Dims("time", "x"),
                "x",
                qualname="myfunc",
            )

    def test_disabled_is_noop(self):
        with xra.enabled(False):
            schema.check_schema(_da(("time",)), Dims("nope"), "x")  # would raise

    def test_list_of_markers(self):
        da = _da(("time", "x"), dtype="int64")
        with pytest.raises(schema.SchemaError, match="dtype"):
            schema.check_schema(da, [Dims("time", "x"), Dtype("float64")], "x")


# ---------------------------------------------------------------------------
# assert_valid_schema (fail-fast declaration checks)
# ---------------------------------------------------------------------------


class TestAssertValidSchema:
    def test_bad_dtype_string(self):
        with pytest.raises(ValueError, match="invalid dtype"):
            _check.assert_valid_schema(Dtype("not_a_dtype"), "ctx")

    def test_duplicate_dim_names(self):
        with pytest.raises(ValueError, match="duplicate names"):
            _check.assert_valid_schema(Dims("time", "time"), "ctx")

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            _check.assert_valid_schema(Dims("time", ""), "ctx")

    def test_valid_markers_pass(self):
        _check.assert_valid_schema(Dims("time", "x"), "ctx")
        _check.assert_valid_schema(Coords("time"), "ctx")
        _check.assert_valid_schema(Dtype("float64"), "ctx")


# ---------------------------------------------------------------------------
# enforce over schema markers
# ---------------------------------------------------------------------------


class TestDeclareSchema:
    def test_validates_input_and_passes_through(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray:
            return x

        da = _da(("time", "x"))
        assert f(da) is da
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(_da(("time",)))

    def test_keyword_and_non_dataarray_args(self):
        @xra.enforce
        def f(
            flag: bool,
            x: Annotated[xr.DataArray, Dtype("float64")],
        ) -> xr.DataArray:
            return x

        assert f(True, x=_da()).shape  # keyword DataArray; non-DataArray passes through

    def test_optional_none_input_skipped(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray | None, Dims("time", "x")] = None,
        ) -> xr.DataArray | None:
            return x

        assert f(None) is None

    def test_declared_default_is_validated(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x")] = _da(("time",)),
        ) -> xr.DataArray:
            return x

        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f()  # x left at its non-conforming default

    def test_varargs_and_kwargs_survive_apply_defaults(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x")], *rest: int, **kw: int
        ) -> xr.DataArray:
            assert rest == (1, 2)
            assert kw == {"z": 3}
            return x

        da = _da(("time", "x"))
        assert f(da, 1, 2, z=3) is da

    def test_output_single_validated(self):
        @xra.enforce
        def f(n: int) -> Annotated[xr.DataArray, Dtype("float64")]:
            return _da(dtype="int64")

        with pytest.raises(schema.SchemaError, match="dtype"):
            f(1)

    def test_output_typeddict_validated(self):
        class Out(TypedDict):
            good: Annotated[xr.DataArray, Dims("time", "x")]

        @xra.enforce
        def f() -> Out:
            return {"good": _da(("time",))}

        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f()

    def test_output_dataclass_validated(self):
        @dataclass
        class Out:
            good: Annotated[xr.DataArray, Dims("time", "x")]

        @xra.enforce
        def f() -> Out:
            return Out(good=_da(("time",)))

        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f()

    def test_disabled_is_total_noop(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Dims("nope")]) -> xr.DataArray:
            return x

        with xra.enabled(False):
            assert f(_da(("time", "x"))).shape

    def test_fail_fast_bad_input_declaration(self):
        with pytest.raises(ValueError, match="invalid dtype"):

            @xra.enforce
            def f(x: Annotated[xr.DataArray, Dtype("bogus")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_output_declaration(self):
        with pytest.raises(ValueError, match="invalid dtype"):

            @xra.enforce
            def f(n: int) -> Annotated[xr.DataArray, Dtype("bogus")]:
                return _da()


class TestAnnotationsSurviveWraps:
    def test_injected_annotations_survive(self):
        import typing

        src = "def f(p):\n    return p\n"
        ns: dict[str, typing.Any] = {}
        exec(src, ns)
        fn = ns["f"]
        fn.__annotations__["return"] = Annotated[xr.DataArray, Dims("time")]

        wrapped = xra.enforce(fn)
        hints = typing.get_type_hints(wrapped, include_extras=True)
        assert "return" in hints
