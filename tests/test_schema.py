"""Tests for the `xarray_annotated.schema` structural-validation domain.

Covers the markers (with their options), the annotation reader, the per-property
checkers, the policy, and `enforce` over schema markers.  Structural validation
never mutates, so the checks assert/raise/warn but return arrays unchanged.
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import schema
from xarray_annotated.schema import Coords, Dims, Dtype, _annotations, _check, _config


def _da(dims=("time", "x"), dtype="float64", coords=()):
    """Build a DataArray with the given dims/dtype; attach coords for `coords` dims."""
    shape = tuple(range(2, 2 + len(dims)))
    arr = np.zeros(shape, dtype=dtype)
    coord_map = {
        d: np.arange(s) for d, s in zip(dims, shape, strict=True) if d in coords
    }
    return xr.DataArray(arr, dims=dims, coords=coord_map)


def _run(da, marker, on_mismatch: _config.OnMismatch = "error"):
    """Run check_schema for one marker under a fixed severity."""
    return schema.check_schema(da, marker, "x", on_mismatch=on_mismatch)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


class TestMarkers:
    def test_accessors(self):
        d = Dims("time", "x")
        assert d.names == ("time", "x")
        assert d.ordered is False
        assert d.on_mismatch is None
        assert Dims("time", ordered=True).ordered is True
        assert Coords("lat").names == ("lat",)
        assert Dtype("float64").dtype == "float64"
        assert Dtype("float64").exact is False
        assert Dtype("f4", exact=True).exact is True

    def test_repr_roundtrips_eval(self):
        markers = [
            Dims("time", "x"),
            Dims("time", ordered=True),
            Dims("time", on_mismatch="warn"),
            Coords("time", "lat"),
            Coords("time", on_mismatch="ignore"),
            Dtype("float64"),
            Dtype("int32", exact=True, on_mismatch="warn"),
        ]
        for marker in markers:
            assert eval(repr(marker)) == marker

    def test_equality_and_hash_include_options(self):
        assert Dims("time") == Dims("time")
        assert Dims("time") != Dims("time", ordered=True)
        assert Dtype("f8") != Dtype("f8", exact=True)
        assert Dims("time") != Coords("time")  # distinct marker types
        assert Dims("time") != "time"  # not equal to a bare string
        assert Dtype("f8") != "f8"
        assert hash(Dims("time", ordered=True)) == hash(Dims("time", ordered=True))

    def test_markers_hashable_in_annotations(self):
        assert {Dims("time"), Coords("lat"), Dtype("int32")}

    def test_invalid_on_mismatch_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            Dims("time", on_mismatch="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reading declarations off annotations
# ---------------------------------------------------------------------------


class TestAnnotatedSchema:
    def test_collects_multiple_markers(self):
        hint = Annotated[xr.DataArray, Dims("time"), Dtype("f8")]
        assert _annotations.annotated_schema(hint) == [Dims("time"), Dtype("f8")]

    def test_ignores_bare_strings(self):
        hint = Annotated[xr.DataArray, "a description", Dims("time")]
        assert _annotations.annotated_schema(hint) == [Dims("time")]

    def test_non_dataarray_yields_none(self):
        assert _annotations.annotated_schema(Annotated[bool, Dims("x")]) is None

    def test_no_markers_yields_none(self):
        assert _annotations.annotated_schema(Annotated[xr.DataArray, "note"]) is None
        assert _annotations.annotated_schema(xr.DataArray) is None

    def test_optional_dataarray_read(self):
        hint = Annotated[xr.DataArray | None, Dims("time")]
        assert _annotations.annotated_schema(hint) == [Dims("time")]

    def test_from_signature_inputs_and_single_output(self):
        def node(
            a: Annotated[xr.DataArray, Dims("time")],
            b: int,
        ) -> Annotated[xr.DataArray, Dtype("f8")]:
            return a

        inputs, output = _annotations.schema_from_signature(node)
        assert inputs == {"a": [Dims("time")]}
        assert output == [Dtype("f8")]

    def test_from_signature_typeddict_output(self):
        class Out(TypedDict):
            hi: Annotated[xr.DataArray, Dims("time")]
            lo: Annotated[xr.DataArray, Dtype("f8")]

        def node() -> Out: ...

        _, output = _annotations.schema_from_signature(node)
        assert output == {"hi": [Dims("time")], "lo": [Dtype("f8")]}

    def test_from_signature_dataclass_output(self):
        @dataclass
        class Out:
            hi: Annotated[xr.DataArray, Dims("time")]
            plain: int

        def node() -> Out: ...

        _, output = _annotations.schema_from_signature(node)
        assert output == {"hi": [Dims("time")]}

    def test_from_signature_no_output(self):
        def node(a: int) -> int:
            return a

        inputs, output = _annotations.schema_from_signature(node)
        assert inputs == {}
        assert output is None


# ---------------------------------------------------------------------------
# Per-property checkers (via check_schema, on_mismatch="error")
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

    def test_warn_and_ignore(self):
        with pytest.warns(schema.SchemaWarning, match="dims mismatch"):
            _run(_da(("time",)), Dims("time", "x"), on_mismatch="warn")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _run(_da(("time",)), Dims("time", "x"), on_mismatch="ignore")


class TestCheckCoords:
    def test_present_passes_extras_ok(self):
        _run(_da(("time", "x"), coords=("time", "x")), Coords("time"))

    def test_missing_coord_fails(self):
        # dims present but no coordinate labels attached
        with pytest.raises(schema.SchemaError, match="missing coords"):
            _run(_da(("time", "x")), Coords("time"))

    def test_warn(self):
        with pytest.warns(schema.SchemaWarning, match="missing coords"):
            _run(_da(("time",)), Coords("time"), on_mismatch="warn")


class TestCheckDtype:
    def test_kind_match_ignores_width(self):
        _run(_da(dtype="float32"), Dtype("float64"))  # kind 'f' == 'f'

    def test_kind_mismatch_fails(self):
        with pytest.raises(schema.SchemaError, match="dtype kind mismatch"):
            _run(_da(dtype="int64"), Dtype("float64"))

    def test_exact_requires_precise_dtype(self):
        _run(_da(dtype="float64"), Dtype("float64", exact=True))
        with pytest.raises(schema.SchemaError, match="dtype mismatch"):
            _run(_da(dtype="float32"), Dtype("float64", exact=True))


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
        assert schema.dtype_compatible(Dtype("float64"), Dtype("float32"))

    def test_different_kind(self):
        assert not schema.dtype_compatible(Dtype("float64"), Dtype("int64"))

    def test_both_exact_differ(self):
        a = Dtype("float64", exact=True)
        b = Dtype("float32", exact=True)
        assert not schema.dtype_compatible(a, b)

    def test_one_side_loose(self):
        # Only one side requires exact → width mismatch is not provably inconsistent.
        assert schema.dtype_compatible(Dtype("float64", exact=True), Dtype("float32"))


# ---------------------------------------------------------------------------
# check_schema router behaviour
# ---------------------------------------------------------------------------


class TestCheckSchema:
    def test_returns_array_unchanged(self):
        da = _da(("time", "x"))
        assert schema.check_schema(da, Dims("time", "x"), "x") is da

    def test_marker_override_beats_argument(self):
        da = _da(("time",))
        # argument says error, but the marker says warn → warns, no raise
        with pytest.warns(schema.SchemaWarning):
            schema.check_schema(
                da, Dims("time", "x", on_mismatch="warn"), "x", on_mismatch="error"
            )

    def test_qualname_prefix_in_message(self):
        with pytest.raises(schema.SchemaError, match=r"\[myfunc\] 'x'"):
            schema.check_schema(
                _da(("time",)),
                Dims("time", "x"),
                "x",
                on_mismatch="error",
                qualname="myfunc",
            )

    def test_disabled_is_noop(self):
        with xra.enabled(False):
            schema.check_schema(_da(("time",)), Dims("nope"), "x")  # would raise

    def test_list_of_markers(self):
        da = _da(("time", "x"), dtype="int64")
        with pytest.raises(schema.SchemaError, match="dtype"):
            schema.check_schema(
                da, [Dims("time", "x"), Dtype("float64")], "x", on_mismatch="error"
            )

    def test_invalid_on_mismatch_argument(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            schema.check_schema(_da(), Dims("time", "x"), "x", on_mismatch="bogus")  # type: ignore[arg-type]


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
# Axis default
# ---------------------------------------------------------------------------


class TestAxis:
    def test_default(self):
        assert _config.DEFAULT_ON_MISMATCH == "error"

    def test_axis_type_is_public(self):
        assert schema.OnMismatch is _config.OnMismatch


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

    def test_declared_default_honours_on_mismatch(self):
        @xra.enforce(schema_on_mismatch="warn")
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x")] = _da(("time",)),
        ) -> xr.DataArray:
            return x

        with pytest.warns(schema.SchemaWarning, match="dims mismatch"):
            f()

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

    def test_decorator_on_mismatch_kwarg(self):
        @xra.enforce(schema_on_mismatch="warn")
        def f(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray:
            return x

        with pytest.warns(schema.SchemaWarning):
            f(_da(("time",)))

    def test_marker_override_beats_decorator(self):
        @xra.enforce(schema_on_mismatch="error")
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x", on_mismatch="warn")],
        ) -> xr.DataArray:
            return x

        with pytest.warns(schema.SchemaWarning):
            f(_da(("time",)))

    def test_disabled_is_total_noop(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Dims("nope")]) -> xr.DataArray:
            return x

        with xra.enabled(False):
            assert f(_da(("time", "x"))).shape

    def test_invalid_decorator_kwarg(self):
        with pytest.raises(ValueError, match="Invalid schema_on_mismatch"):

            @xra.enforce(schema_on_mismatch="bogus")  # type: ignore[arg-type]
            def f(x: Annotated[xr.DataArray, Dims("time")]) -> xr.DataArray:
                return x

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
