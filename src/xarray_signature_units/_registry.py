r"""Pluggable pint registry selection.

The active pint registry is **process-global**. By default nothing special
happens at import time: `get_registry` falls back to
`pint.get_application_registry()`, so plain pint unit strings work out of the
box. CF-convention strings such as `"umol m-2 s-1"` or `"g m-2 d-1"` require
cf-xarray's UDUNITS-aware registry, activated once via `use_cf_units`.
"""

import importlib.util

import pint
import pint_xarray

_UREG: pint.UnitRegistry | pint.registry.ApplicationRegistry | None = None
_using_cf: bool = False


def get_registry() -> pint.UnitRegistry | pint.registry.ApplicationRegistry:
    """Return the active pint `UnitRegistry`.

    Defaults to `pint.get_application_registry()` (plain pint) until
    `set_registry` or `use_cf_units` is called.

    Returns:
        The process-global pint `UnitRegistry`.
    """
    if _UREG is not None:
        return _UREG
    return pint.get_application_registry()


def set_registry(ureg: pint.UnitRegistry | pint.registry.ApplicationRegistry) -> None:
    """Set the process-wide pint registry used by this module and pint-xarray.

    Also calls `pint_xarray.setup_registry(ureg)` so the `.pint` accessor and
    this module's parse/compat helpers never drift apart.

    pint has a single process-global application registry, so this is a
    one-time, startup choice, not a per-array setting: quantities created under
    two different registries cannot be mixed (pint raises).

    Args:
        ureg: The `UnitRegistry` to install as the process-global registry.
    """
    global _UREG, _using_cf
    _UREG = ureg
    _using_cf = False
    pint_xarray.setup_registry(ureg)


def use_cf_units() -> None:
    """Activate cf-xarray's UDUNITS-aware registry.

    Lazily imports `cf_xarray.units` (from the `[cf]` extra) and installs its
    registry via `set_registry`, so CF-convention unit strings such as
    `"umol m-2 s-1"` and `"g m-2 d-1"` parse.

    Raises:
        ImportError: If `cf-xarray` is not installed.
    """
    global _using_cf
    try:
        import cf_xarray.units
    except ImportError as exc:
        raise ImportError(
            "use_cf_units() requires cf-xarray. Install it with "
            "`pip install xarray-signature-units[cf]`."
        ) from exc
    set_registry(cf_xarray.units.units)
    _using_cf = True


def _cf_hint(unit: str) -> str:
    """Suggest `use_cf_units` when a unit likely needs the CF registry.

    Returns a parenthetical hint only when cf-xarray is installed but not yet
    active, i.e. exactly the case where switching registries would fix a parse
    failure. Empty string otherwise.

    Args:
        unit: The unit string that failed to parse.

    Returns:
        A hint string, or `""` when the hint is not applicable.
    """
    if _using_cf or importlib.util.find_spec("cf_xarray") is None:
        return ""
    return f" (call use_cf_units() to enable CF/UDUNITS units like {unit!r})"
