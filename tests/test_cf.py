"""Tests for the CF/UDUNITS registry path (``use_cf_units()``), needs the ``[cf]`` extra."""

import numpy as np
import pytest

pytest.importorskip("cf_xarray")

import xarray as xr

from xarray_annotated import units
from xarray_annotated.units import _check


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


@pytest.fixture(autouse=True)
def _cf_registry():
    """Activate the CF registry for each test in this module.

    Cleanup (restoring the process-global registry) is handled by the autouse
    ``_isolate_registry`` fixture in ``conftest.py``, which snapshots and
    restores the full registry state around every test.
    """
    units.use_cf_units()


class TestAssertValidUnit:
    @pytest.mark.parametrize("unit", ["umol m-2 s-1", "g m-2 d-1", "t ha-1 month-1"])
    def test_cf_units_pass(self, unit):
        units.assert_valid_unit(unit, "ctx")  # no raise


class TestCFParsing:
    @pytest.mark.parametrize(
        "unit",
        ["umol m-2 s-1", "g m-2 d-1", "t ha-1", "mm d-1", "ppm", "degC"],
    )
    def test_cf_unit_strings_parse_and_convert(self, unit):
        da = _da([[1.0, 2.0]], unit=unit)
        out = units.check_units(da, unit, "x", on_missing="error")
        assert out.attrs["units"] == unit
        np.testing.assert_allclose(out.values, da.values)


class TestUnitsCompatible:
    def test_compatible(self):
        assert _check.units_compatible("g m-2 d-1", "kg m-2 s-1")

    def test_incompatible(self):
        assert not _check.units_compatible("g m-2 d-1", "Pa")


class TestUnitsEqual:
    def test_equal(self):
        assert _check.units_equal("g m-2 d-1", "g m-2 d-1")


class TestUseCfUnitsHint:
    def test_parse_failure_hints_at_use_cf_units(self):
        import pint

        # Switch back to plain pint (cf-xarray still importable) to exercise the
        # "cf-xarray installed but not active" branch of the hint.
        units.set_registry(pint.UnitRegistry())
        with pytest.raises(ValueError, match="use_cf_units"):
            units.assert_valid_unit("umol m-2 s-1", "ctx")
