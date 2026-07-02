"""Tests for the `xarray_annotated.schema` structural-validation domain.

Covers the markers (with their options), the annotation reader, the per-property
checkers, the policy, and the `declare_schema` decorator.  Structural validation
never mutates, so the checks assert/raise/warn but return arrays unchanged.
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pytest
import xarray as xr

from xarray_annotated import schema, units
from xarray_annotated.schema import Coords, Dims, Dtype
from xarray_annotated.schema import _annotations, _check, _config


def _da(dims=("time", "x"), dtype="float64", coords=()):
    """Build a DataArray with the given dims/dtype; attach coords for `coords` dims."""
    shape = tuple(range(2, 2 + len(dims)))
    arr = np.zeros(shape, dtype=dtype)
    coord_map = {d: np.arange(s) for d, s in zip(dims, shape) if d in coords}
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
            assert eval(repr(marker)) == marker  # noqa: S307

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
        with schema.policy(enabled=False):
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
# Policy
# ---------------------------------------------------------------------------


class TestSchemaPolicy:
    def test_defaults(self):
        with schema.policy(enabled=None, on_mismatch=None):
            pol = schema.get_policy()
            assert pol.enabled is True
            assert pol.on_mismatch == "error"

    def test_set_and_restore(self):
        with schema.policy(on_mismatch="warn"):
            assert schema.get_policy().on_mismatch == "warn"
        assert schema.get_policy().on_mismatch == "error"

    def test_env_overrides_process(self, monkeypatch):
        with schema.policy(on_mismatch="ignore"):
            monkeypatch.setenv(_config.ON_MISMATCH_ENV_VAR, "warn")
            assert schema.get_policy().on_mismatch == "warn"

    def test_enabled_switch_is_shared_with_units(self):
        # The master switch is package-wide: toggling it in units is seen by schema.
        with units.policy(enabled=False):
            assert schema.get_policy().enabled is False

    def test_invalid_on_mismatch_raises(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            schema.set_policy(on_mismatch="bogus")  # type: ignore[arg-type]

    def test_invalid_enabled_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "maybe")
        with pytest.raises(ValueError, match=_config.ENABLED_ENV_VAR):
            schema.get_policy()


# ---------------------------------------------------------------------------
# declare_schema decorator
# ---------------------------------------------------------------------------


class TestDeclareSchema:
    def test_validates_input_and_passes_through(self):
        @schema.declare_schema
        def f(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray:
            return x

        da = _da(("time", "x"))
        assert f(da) is da
        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f(_da(("time",)))

    def test_keyword_and_non_dataarray_args(self):
        @schema.declare_schema
        def f(
            flag: bool,
            x: Annotated[xr.DataArray, Dtype("float64")],
        ) -> xr.DataArray:
            return x

        assert f(True, x=_da()).shape  # keyword DataArray; non-DataArray passes through

    def test_optional_none_input_skipped(self):
        @schema.declare_schema
        def f(
            x: Annotated[xr.DataArray | None, Dims("time", "x")] = None,
        ) -> xr.DataArray | None:
            return x

        assert f(None) is None

    def test_output_single_validated(self):
        @schema.declare_schema
        def f(n: int) -> Annotated[xr.DataArray, Dtype("float64")]:
            return _da(dtype="int64")

        with pytest.raises(schema.SchemaError, match="dtype"):
            f(1)

    def test_output_typeddict_validated(self):
        class Out(TypedDict):
            good: Annotated[xr.DataArray, Dims("time", "x")]

        @schema.declare_schema
        def f() -> Out:
            return {"good": _da(("time",))}

        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f()

    def test_output_dataclass_validated(self):
        @dataclass
        class Out:
            good: Annotated[xr.DataArray, Dims("time", "x")]

        @schema.declare_schema
        def f() -> Out:
            return Out(good=_da(("time",)))

        with pytest.raises(schema.SchemaError, match="dims mismatch"):
            f()

    def test_decorator_on_mismatch_kwarg(self):
        @schema.declare_schema(on_mismatch="warn")
        def f(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray:
            return x

        with pytest.warns(schema.SchemaWarning):
            f(_da(("time",)))

    def test_marker_override_beats_decorator(self):
        @schema.declare_schema(on_mismatch="error")
        def f(
            x: Annotated[xr.DataArray, Dims("time", "x", on_mismatch="warn")],
        ) -> xr.DataArray:
            return x

        with pytest.warns(schema.SchemaWarning):
            f(_da(("time",)))

    def test_disabled_is_total_noop(self):
        @schema.declare_schema
        def f(x: Annotated[xr.DataArray, Dims("nope")]) -> xr.DataArray:
            return x

        with schema.policy(enabled=False):
            assert f(_da(("time", "x"))).shape

    def test_invalid_decorator_kwarg(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):

            @schema.declare_schema(on_mismatch="bogus")  # type: ignore[arg-type]
            def f(x: Annotated[xr.DataArray, Dims("time")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_input_declaration(self):
        with pytest.raises(ValueError, match="invalid dtype"):

            @schema.declare_schema
            def f(x: Annotated[xr.DataArray, Dtype("bogus")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_output_declaration(self):
        with pytest.raises(ValueError, match="invalid dtype"):

            @schema.declare_schema
            def f(n: int) -> Annotated[xr.DataArray, Dtype("bogus")]:
                return _da()
