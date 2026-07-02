"""Runtime validation of declared structural properties (stub).

The structural counterpart to ``units._check.check_units``: given a DataArray and
a declared ``Dims`` / ``Coords`` / ``Dtype``, validate the array's dimensions,
coordinates, and dtype.  Not yet implemented — this module reserves the entry
point so downstream consumers (and a future ``declare_schema`` decorator) have a
stable name to target.
"""

from typing import Any

import xarray as xr


def check_schema(
    da: xr.DataArray,
    declared: Any,
    name: str,
) -> xr.DataArray:
    """Validate a DataArray against a declared structural marker (stub).

    Args:
        da: The input DataArray to validate.
        declared: A ``Dims`` / ``Coords`` / ``Dtype`` marker (from
            ``schema._annotations``) describing the expected structure.
        name: A label for the array, used in error messages.

    Raises:
        NotImplementedError: Always — structural validation is not yet built.
    """
    raise NotImplementedError("schema validation is not yet implemented")
