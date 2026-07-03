"""Public API for the units domain of xarray-annotated.

Run-time validation of pint/CF units declared via ``typing.Annotated`` on xarray
DataArrays.
"""

from ._annotations import Unit, units_from_signature
from ._check import (
    UnitsWarning,
    assert_valid_unit,
    check_units,
    units_compatible,
    units_equal,
)
from ._config import (
    OnInexact,
    OnMissing,
    Policy,
    get_policy,
    policy,
    set_policy,
)
from ._decorator import declare_units
from ._registry import (
    set_registry,
    use_cf_units,
)

__all__ = [
    "OnInexact",
    "OnMissing",
    "Policy",
    "Unit",
    "UnitsWarning",
    "assert_valid_unit",
    "check_units",
    "declare_units",
    "get_policy",
    "policy",
    "set_policy",
    "set_registry",
    "units_compatible",
    "units_equal",
    "units_from_signature",
    "use_cf_units",
]
