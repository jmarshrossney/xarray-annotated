"""Structural-property domain for xarray-annotated.

Declares and validates a DataArray's *structural* properties — the ones every
DataArray possesses regardless of physical units: its dimensions (``Dims``),
coordinates (``Coords``), and dtype (``Dtype``).  The structural counterpart to
``xarray_annotated.units``, but simpler: structural validation only *asserts* (it
never converts or mutates).

Declare several at once inside ``Annotated`` and apply them with ``declare_schema``::

    from typing import Annotated
    import xarray as xr
    from xarray_annotated.schema import declare_schema, Dims, Dtype

    @declare_schema
    def f(x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")]) -> xr.DataArray:
        ...
"""

from ._annotations import Coords, Dims, Dtype, schema_from_signature
from ._check import (
    SchemaError,
    SchemaWarning,
    assert_valid_schema,
    check_schema,
)
from ._config import (
    Policy,
    get_policy,
    policy,
    set_policy,
)
from ._decorator import declare_schema

__all__ = [
    "Coords",
    "Dims",
    "Dtype",
    "Policy",
    "SchemaError",
    "SchemaWarning",
    "assert_valid_schema",
    "check_schema",
    "declare_schema",
    "get_policy",
    "policy",
    "schema_from_signature",
    "set_policy",
]
