"""Public API for the units domain of xarray-annotated.

Run-time validation of pint/CF units declared via ``typing.Annotated`` on xarray
DataArrays.
"""

from ._annotations import Unit
from ._check import (
    apply_output_units,
    assert_valid_unit,
    check_units,
    convert,
    dequantify,
    quantify,
    units_compatible,
    units_equal,
)
from ._registry import (
    get_registry,
    set_registry,
    use_cf_units,
)

__all__ = [
    "Unit",
    "apply_output_units",
    "assert_valid_unit",
    "check_units",
    "convert",
    "dequantify",
    "get_registry",
    "quantify",
    "set_registry",
    "units_compatible",
    "units_equal",
    "use_cf_units",
]
