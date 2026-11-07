"""Tests for ``stamp_output_units``, which states declared output units."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated, NotRequired, TypedDict

import numpy as np
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import ContractError, Dims, Unit
from xarray_annotated.units import UnitError

MM = Annotated[xr.DataArray, Unit("mm")]
MM_DAY = Annotated[xr.DataArray, Unit("mm / day")]


def _bare(n: int = 2) -> xr.DataArray:
    return xr.DataArray(np.arange(float(n)), dims="x")


def _labelled(unit: str) -> xr.DataArray:
    return _bare().assign_attrs(units=unit)


class Out(TypedDict):
    depth: MM
    rate: MM_DAY


class TestSingleReturn:
    def test_fills_a_missing_label(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> MM:
            return _bare()

        assert f().attrs["units"] == "mm"

    def test_stamps_a_copy_sharing_the_data(self):
        raw = _bare()

        @xra.stamp_output_units
        def f() -> MM:
            return raw

        out = f()
        assert "units" not in raw.attrs
        assert np.shares_memory(out.values, raw.values)

    def test_leaves_an_equal_label_as_spelled(self):
        labelled = _labelled("millimeter")

        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> MM:
            return labelled

        assert f() is labelled

    def test_leaves_a_different_label_for_the_check(self):
        @xra.stamp_output_units
        def bare() -> MM:
            return _labelled("m")

        assert bare().attrs["units"] == "m"
        with pytest.raises(UnitError):
            xra.check_annotations(bare)()

    def test_overwrite_true_replaces_a_present_label(self, caplog):
        caplog.set_level(logging.DEBUG, logger="xarray_annotated.units")

        @xra.check_annotations
        @xra.stamp_output_units(overwrite=True)
        def f() -> MM:
            return _labelled("1")

        assert f().attrs["units"] == "mm"
        assert "'1' -> 'mm'" in caplog.text

    def test_no_log_when_overwrite_finds_the_declared_spelling(self, caplog):
        caplog.set_level(logging.DEBUG, logger="xarray_annotated.units")

        @xra.stamp_output_units(overwrite=True)
        def f() -> MM:
            return _labelled("mm")

        f()
        assert not caplog.records

    def test_quantified_output_is_left_alone(self):
        q = xra.units.quantify(_labelled("m"))

        @xra.stamp_output_units(overwrite=True)
        def f() -> MM:
            return q

        assert f() is q

    def test_a_value_other_than_an_array_is_left_for_the_check(self):
        @xra.stamp_output_units
        def bare() -> MM:
            return np.zeros(2)  # type: ignore[return-value]

        assert isinstance(bare(), np.ndarray)
        with pytest.raises(ContractError):
            xra.check_annotations(bare)()

    def test_optional_return_of_none(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> MM | None:
            return None

        assert f() is None

    def test_runs_when_checks_are_switched_off(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> MM:
            return _bare()

        with xra.enabled(False):
            assert f().attrs["units"] == "mm"

    def test_async(self):
        @xra.check_annotations
        @xra.stamp_output_units
        async def f() -> MM:
            return _bare()

        assert asyncio.run(f()).attrs["units"] == "mm"

    def test_other_markers_still_checked(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Annotated[xr.DataArray, Unit("mm"), Dims("y")]:
            return _bare()

        with pytest.raises(ContractError):
            f()

    def test_dask_array_stays_lazy(self):
        da = pytest.importorskip("dask.array")

        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> MM:
            return xr.DataArray(da.zeros(4, chunks=2), dims="x")

        assert isinstance(f().data, da.Array)


class TestRecords:
    def test_typeddict_fields_filled(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Out:
            return {"depth": _bare(), "rate": _bare()}

        out = f()
        assert (out["depth"].attrs["units"], out["rate"].attrs["units"]) == (
            "mm",
            "mm / day",
        )

    def test_typeddict_returned_as_is_when_nothing_changes(self):
        result = {"depth": _labelled("mm"), "rate": _labelled("mm / day")}

        @xra.stamp_output_units
        def f() -> Out:
            return result  # type: ignore[return-value]

        assert f() is result

    def test_typeddict_is_copied_not_mutated(self):
        result = {"depth": _bare(), "rate": _bare()}

        @xra.stamp_output_units
        def f() -> Out:
            return result  # type: ignore[return-value]

        f()
        assert "units" not in result["depth"].attrs

    def test_named_overwrite_leaves_other_stale_labels_to_raise(self):
        @xra.check_annotations
        @xra.stamp_output_units(overwrite=("depth",))
        def f(stale: str) -> Out:
            return {"depth": _labelled("1"), "rate": _labelled(stale)}

        assert f("mm / day")["depth"].attrs["units"] == "mm"
        with pytest.raises(UnitError, match="rate"):
            f("mm")

    def test_overwrite_true_covers_every_field(self):
        @xra.check_annotations
        @xra.stamp_output_units(overwrite=True)
        def f() -> Out:
            # As xr.apply_ufunc does, copying the first input's attrs to each output.
            return {"depth": _labelled("1"), "rate": _labelled("1")}

        out = f()
        assert out["rate"].attrs["units"] == "mm / day"

    def test_missing_not_required_field(self):
        class Partial(TypedDict):
            depth: MM
            rate: NotRequired[MM_DAY]

        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Partial:
            return {"depth": _bare()}

        out = f()
        assert list(out) == ["depth"]
        assert out["depth"].attrs["units"] == "mm"

    def test_field_without_a_unit_is_left_alone(self):
        class Mixed(TypedDict):
            depth: MM
            mask: Annotated[xr.DataArray, Dims("x")]

        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Mixed:
            return {"depth": _bare(), "mask": _bare()}

        assert "units" not in f()["mask"].attrs

    def test_frozen_dataclass_copied_without_post_init(self):
        calls = []

        @dataclass(frozen=True)
        class Frozen:
            depth: MM

            def __post_init__(self):
                calls.append(1)

        original = Frozen(_bare())

        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Frozen:
            return original

        out = f()
        assert out.depth.attrs["units"] == "mm"
        assert "units" not in original.depth.attrs
        assert len(calls) == 1

    def test_optional_record_of_none(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Out | None:
            return None

        assert f() is None

    def test_wrong_record_class_left_for_the_check(self):
        @xra.check_annotations
        @xra.stamp_output_units
        def f() -> Out:
            return [1]  # type: ignore[return-value]

        with pytest.raises(ContractError):
            f()


class TestRefusedAtDecoration:
    def test_above_check_annotations(self):
        def f() -> MM:
            return _bare()

        with pytest.raises(ValueError, match="below @check_annotations"):
            xra.stamp_output_units(xra.check_annotations(f))

    def test_no_output_unit(self):
        with pytest.raises(ValueError, match="nothing to stamp"):

            @xra.stamp_output_units
            def f(p: MM) -> xr.DataArray:
                return p

    def test_malformed_unit(self):
        with pytest.raises(ValueError, match="not a recognised unit"):

            @xra.stamp_output_units
            def f() -> Annotated[xr.DataArray, Unit("not_a_unit")]:
                return _bare()

    def test_bare_string_overwrite(self):
        with pytest.raises(ValueError, match="bare string"):

            @xra.stamp_output_units(overwrite="depth")  # type: ignore[arg-type]
            def f() -> Out:
                return {"depth": _bare(), "rate": _bare()}

    def test_unknown_field(self):
        with pytest.raises(ValueError, match="not fields of Out"):

            @xra.stamp_output_units(overwrite=("dept",))
            def f() -> Out:
                return {"depth": _bare(), "rate": _bare()}

    def test_field_without_a_unit(self):
        class Mixed(TypedDict):
            depth: MM
            mask: xr.DataArray

        with pytest.raises(ValueError, match="declare no Unit"):

            @xra.stamp_output_units(overwrite=("mask",))
            def f() -> Mixed:
                return {"depth": _bare(), "mask": _bare()}

    def test_field_names_on_a_single_return(self):
        with pytest.raises(ValueError, match="overwrite=True"):

            @xra.stamp_output_units(overwrite=("return",))
            def f() -> MM:
                return _bare()
