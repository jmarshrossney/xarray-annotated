"""Public API for the units domain of xarray-annotated.

Run-time validation of pint/CF units declared via ``typing.Annotated`` on xarray
DataArrays.
"""

from ._annotations import Unit, annotated_unit, units_from_signature
from ._check import (
    UnitsWarning,
    apply_output_units,
    assert_valid_unit,
    check_units,
    units_compatible,
    units_equal,
)
from ._config import (
    OnMissing,
    OnOutput,
)
from ._registry import (
    get_registry,
    set_registry,
    use_cf_units,
)

__all__ = [
    "OnMissing",
    "OnOutput",
    "Unit",
    "UnitsWarning",
    "annotated_unit",
    "apply_output_units",
    "assert_valid_unit",
    "check_units",
    "get_registry",
    "set_registry",
    "units_compatible",
    "units_equal",
    "units_from_signature",
    "use_cf_units",
]
