"""Structural-property domain for xarray-annotated.

Declares and validates a DataArray's *structural* properties — the ones every
DataArray possesses regardless of physical units: its dimensions (``Dims``),
coordinates (``Coords``), and dtype (``Dtype``).  The structural counterpart to
``xarray_annotated.units``, but simpler: structural validation only *asserts* (it
never converts or mutates).

Declare several at once inside ``Annotated`` and apply them with
``xarray_annotated.enforce``:

    from typing import Annotated
    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Dims, Dtype

    @xra.enforce
    def f(
        x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")],
    ) -> xr.DataArray: ...
"""

from ._annotations import (
    Coords,
    Dims,
    Dtype,
)
from ._check import (
    SchemaError,
    assert_valid_schema,
    check_schema,
    dims_compatible,
    dtype_compatible,
)

__all__ = [
    "Coords",
    "Dims",
    "Dtype",
    "SchemaError",
    "assert_valid_schema",
    "check_schema",
    "dims_compatible",
    "dtype_compatible",
]
