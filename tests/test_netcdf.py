"""Real files: packing written to disk, and arrays opened lazily from it."""

from typing import Annotated

import numpy as np
import pint
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Dims, Dtype, Freq, Unit, units

pytest.importorskip("netCDF4")

PACKING = {"scale_factor": 0.1, "dtype": "int16", "_FillValue": -32768}


def _hpa() -> xr.DataArray:
    time = xr.date_range("2020-01-01", periods=2, freq="D")
    return xr.DataArray(
        [1000.0, 1010.0],
        dims="time",
        coords={"time": time},
        attrs={"units": "hPa"},
        name="p",
    )


def _packed() -> xr.DataArray:
    da = _hpa()
    da.encoding.update(PACKING)
    return da


def _round_trip(da: xr.DataArray, path) -> np.ndarray:
    da.to_netcdf(path, engine="netcdf4")
    with xr.open_dataarray(path, engine="netcdf4") as back:
        return back.values


@xra.check_annotations
def _pascals(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p


class TestConvertedArraysSurviveWriting:
    """int16 cannot hold 1013000, so stale packing overflows, with no error."""

    EXPECTED = (100000.0, 101000.0)

    def test_convert(self, tmp_path):
        out = units.convert(_packed(), "Pa")
        np.testing.assert_allclose(_round_trip(out, tmp_path / "a.nc"), self.EXPECTED)

    def test_convert_of_a_packed_file(self, tmp_path):
        _hpa().to_netcdf(tmp_path / "packed.nc", encoding={"p": PACKING})
        with xr.open_dataarray(tmp_path / "packed.nc") as opened:
            assert opened.encoding["dtype"] == np.dtype("int16")
            out = units.convert(opened, "Pa")
            back = _round_trip(out, tmp_path / "a.nc")
        np.testing.assert_allclose(back, self.EXPECTED)

    def test_a_quantified_array_cannot_be_written(self, tmp_path):
        out = _pascals(_packed().pint.quantify().pint.to("Pa"))
        with pytest.raises(pint.DimensionalityError):
            _round_trip(out, tmp_path / "a.nc")

    def test_units_dequantify_clears_a_stale_packing(self, tmp_path):
        out = units.dequantify(_pascals(_packed().pint.quantify().pint.to("Pa")))
        np.testing.assert_allclose(_round_trip(out, tmp_path / "a.nc"), self.EXPECTED)

    def test_pints_own_dequantify_corrupts_a_stale_packing(self, tmp_path):
        # Why `units.dequantify` exists: pint keeps the packing, and check_annotations only checks.
        out = _pascals(_packed().pint.quantify().pint.to("Pa")).pint.dequantify()
        back = _round_trip(out, tmp_path / "a.nc")
        assert not np.allclose(back, self.EXPECTED)

    def test_pint_by_hand_with_the_documented_drop_encoding(self, tmp_path):
        out = _packed().pint.quantify().pint.to("Pa").pint.dequantify().drop_encoding()
        np.testing.assert_allclose(_round_trip(out, tmp_path / "a.nc"), self.EXPECTED)

    def test_same_unit_keeps_its_packing(self, tmp_path):
        @xra.check_annotations
        def f(p: Annotated[xr.DataArray, Unit("hPa")]):
            return p

        out = f(_packed())
        assert out.encoding == PACKING
        np.testing.assert_allclose(
            _round_trip(out, tmp_path / "a.nc"), [1000.0, 1010.0]
        )


class TestOpenedFiles:
    def test_every_check_leaves_the_array_on_disk(self, tmp_path):
        _hpa().to_netcdf(tmp_path / "a.nc")

        @xra.check_annotations
        def f(
            p: Annotated[
                xr.DataArray,
                Unit("hPa"),
                Dims("time"),
                Dtype("float64"),
                Freq("D"),
            ],
        ) -> Annotated[xr.DataArray, Unit("hPa")]:
            return p

        with xr.open_dataarray(tmp_path / "a.nc") as opened:
            out = f(opened)
            assert not out.variable._in_memory
            assert not opened.variable._in_memory

    @pytest.mark.parametrize("file_format", ["NETCDF3_CLASSIC", "NETCDF4"])
    def test_dtype_matches_whatever_the_file_format(self, tmp_path, file_format):
        _hpa().to_netcdf(tmp_path / "a.nc", format=file_format)

        @xra.check_annotations
        def f(p: Annotated[xr.DataArray, Dtype("float64")]): ...

        with xr.open_dataarray(tmp_path / "a.nc") as opened:
            f(opened)
