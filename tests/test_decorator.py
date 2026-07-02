"""Tests for the ``declare_units`` signature-driven decorator (plain-pint registry)."""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pint
import pytest
import xarray as xr

from xarray_annotated import units


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


class TestBareDecorator:
    def test_converts_input_and_stamps_output(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> Annotated[xr.DataArray, "Pa"]:
            return p

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_stamps_typeddict_outputs(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            plain: xr.DataArray  # no declared unit

        @units.declare_units
        def f() -> Out:
            return {"gpp": _da([[1.0]]), "plain": _da([[2.0]])}

        out = f()
        assert out["gpp"].attrs["units"] == "g / m**2 / d"
        assert "units" not in out["plain"].attrs

    def test_stamps_dataclass_outputs(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            plain: xr.DataArray  # no declared unit

        @units.declare_units
        def f() -> Out:
            return Out(gpp=_da([[1.0]]), plain=_da([[2.0]]))

        out = f()
        assert out.gpp.attrs["units"] == "g / m**2 / d"
        assert "units" not in out.plain.attrs

    def test_non_dataarray_args_pass_through(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"], scale: int) -> xr.DataArray:
            return p * scale

        out = f(_da([[1.0]], unit="Pa"), 3)
        np.testing.assert_allclose(out.values, [[3.0]])

    def test_input_passed_by_keyword(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        out = f(p=_da([[10.0]], unit="hPa"))
        np.testing.assert_allclose(out.values, [[1000.0]])

    def test_optional_dataarray_none_is_skipped(self):
        @units.declare_units
        def f(x: Annotated[xr.DataArray | None, "Pa"] = None) -> xr.DataArray:
            return _da([[1.0]], unit="Pa")

        out = f()  # x defaults to None; must not raise
        assert out.attrs["units"] == "Pa"


class TestFailFast:
    def test_bad_declared_unit_raises_at_decoration(self):
        with pytest.raises(ValueError, match="not a recognised"):

            @units.declare_units
            def f(p: Annotated[xr.DataArray, "not_a_unit"]) -> xr.DataArray:
                return p

    def test_dimensional_mismatch_always_raises(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "kg"]) -> xr.DataArray:
            return p

        with pytest.raises(pint.DimensionalityError):
            f(_da([[1.0]], unit="degC"))


class TestPolicyResolution:
    def test_on_missing_kwarg_overrides_global(self):
        @units.declare_units(on_missing="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        # Global on_missing is warn, but the per-decorator "error" must raise.
        with (
            units.policy(on_missing="warn"),
            pytest.raises(ValueError, match="no 'units'"),
        ):
            f(_da([[1.0]]))

    def test_default_resolves_active_policy_per_call(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        da = _da([[1.0]])  # missing units
        with (
            units.policy(on_missing="error"),
            pytest.raises(ValueError, match="no 'units'"),
        ):
            f(da)
        with units.policy(on_missing="warn"), pytest.warns(units.UnitsWarning):
            f(da)

    def test_on_missing_ignore_skips_input_validation(self):
        @units.declare_units(on_missing="ignore")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = f(_da([[1.0]]))  # missing units, but ignore neither raises nor warns
        assert "units" not in out.attrs

    def test_disabled_policy_is_total_noop(self):
        # enabled=False: input is not converted AND output is not stamped.
        @units.declare_units
        def f(
            p: Annotated[xr.DataArray, "Pa"],
        ) -> Annotated[xr.DataArray, "Pa"]:
            return p

        with units.policy(enabled=False):
            out = f(_da([[10.0]], unit="hPa"))
        assert out.attrs["units"] == "hPa"  # not converted, not re-stamped
        np.testing.assert_allclose(out.values, [[10.0]])

    def test_invalid_on_missing_kwarg_raises_at_decoration(self):
        with pytest.raises(ValueError, match="Invalid on_missing"):

            @units.declare_units(on_missing="bogus")  # type: ignore[arg-type]
            def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
                return p


class TestInexact:
    def test_on_inexact_error_forbids_conversion(self):
        @units.declare_units(on_inexact="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with pytest.raises(ValueError, match="on_inexact='error'"):
            f(_da([[10.0]], unit="hPa"))

    def test_on_inexact_error_accepts_equivalent_spelling(self):
        @units.declare_units(on_inexact="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        out = f(_da([[10.0]], unit="pascal"))
        np.testing.assert_allclose(out.values, [[10.0]])

    def test_on_inexact_warn_converts_with_warning(self):
        @units.declare_units(on_inexact="warn")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with pytest.warns(units.UnitsWarning, match="value-changing"):
            out = f(_da([[10.0]], unit="hPa"))
        np.testing.assert_allclose(out.values, [[1000.0]])
