"""The deprecated per-domain decorators: they warn, and still work as before.

``declare_units`` / ``declare_schema`` / ``declare_freq`` are ``enforce``
restricted to their own domain, so stacked usage keeps its old behaviour.
Remove this module with them.
"""

from typing import Annotated

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_annotated import schema, temporal, units
from xarray_annotated.schema import Dims
from xarray_annotated.temporal import Freq
from xarray_annotated.units import Unit


def _da(unit: str | None = "hPa") -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=4, freq="D")
    da = xr.DataArray(np.ones(4), dims="time", coords={"time": times})
    if unit is not None:
        da.attrs["units"] = unit
    return da


@pytest.mark.parametrize(
    ("decorator", "kwargs", "replacement"),
    [
        (units.declare_units, {"on_missing": "error"}, "units_on_missing"),
        (schema.declare_schema, {"on_mismatch": "error"}, "schema_on_mismatch"),
        (temporal.declare_freq, {"on_mismatch": "error"}, "temporal_on_mismatch"),
    ],
)
class TestWarns:
    def test_bare(self, decorator, kwargs, replacement):
        with pytest.warns(DeprecationWarning, match="xarray_annotated.enforce") as rec:

            @decorator
            def f(x: xr.DataArray): ...

        assert len(rec) == 1
        assert rec[0].filename == __file__  # points at the decoration site

    def test_called_names_replacement_keyword(self, decorator, kwargs, replacement):
        with pytest.warns(DeprecationWarning, match=replacement):
            decorator(**kwargs)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestBehaviour:
    def test_each_shim_reads_only_its_own_domain(self):
        hint = Annotated[xr.DataArray, Unit("Pa"), Dims("x"), Freq("7D")]

        @units.declare_units
        def f(x: hint) -> xr.DataArray:  # type: ignore[valid-type]
            return x

        out = f(_da())  # Dims / Freq violated, but not this shim's business
        assert out.attrs["units"] == "Pa"

        @schema.declare_schema
        def g(x: hint): ...  # type: ignore[valid-type]

        with pytest.raises(schema.SchemaError):
            g(_da())

        @temporal.declare_freq
        def h(x: hint): ...  # type: ignore[valid-type]

        with pytest.raises(temporal.FreqError):
            h(_da())

    def test_stacked_units_outer_converts_before_schema(self):
        @units.declare_units
        @schema.declare_schema
        def f(
            x: Annotated[xr.DataArray, Unit("Pa"), Dims("time")],
        ) -> Annotated[xr.DataArray, Unit("Pa"), Dims("time")]:
            return x

        out = f(_da())
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, 100.0)

    def test_old_keywords_still_honoured(self):
        @units.declare_units(on_missing="error")
        def f(x: Annotated[xr.DataArray, Unit("Pa")]): ...

        with pytest.raises(ValueError, match="no 'units' attribute"):
            f(_da(unit=None))

        @temporal.declare_freq(on_mismatch="warn")
        def g(x: Annotated[xr.DataArray, Freq("7D")]): ...

        with pytest.warns(temporal.FreqWarning):
            g(_da())
