"""Runtime unit validation and conversion.

The "usage side" of the package: given a validation `Policy` and a declared unit
string (typically read off a signature by `_annotations`), validate an actual
`DataArray` against it and convert it into the declared unit. Conversion is
delegated to pint-xarray's `.pint` accessor via the active registry.
"""

import warnings

import pint
import xarray as xr
from pint_xarray.errors import PintExceptionGroup

from ._annotations import Unit
from ._config import (
    OnInexact,
    OnMissing,
    _validate_on_inexact,
    _validate_on_missing,
    get_policy,
)
from ._registry import _cf_hint, get_registry

#: Upper bound on the length of a unit string we are willing to hand to pint's
#: parser. Real units are short (``"umol m-2 s-1"`` is 12 chars); anything far
#: longer is not a unit. pint's parser is not hardened against adversarial input
#: and can hang on pathological strings (e.g. a single very long token), so an
#: over-long ``units`` attribute read from an untrusted file must be refused
#: *before* it reaches the parser rather than after. This does not defend against
#: every DoS input pint accepts (a short string can still be pathological, e.g.
#: nested numeric exponentiation), only the length-driven class.
_MAX_UNIT_LEN = 256


class UnitsWarning(UserWarning):
    """Emitted when a DataArray input cannot be fully unit-validated.

    Raised by `check_units` when an input has a missing or unparseable `units`
    attribute and `on_missing` is `"warn"`, or when a value-changing conversion
    happens under `on_inexact="warn"`.  Subclasses `UserWarning` so callers can
    target it specifically.

    Example:
        >>> import warnings
        >>> from xarray_annotated.units import UnitsWarning
        >>> issubclass(UnitsWarning, UserWarning)
        True
        >>> warnings.filterwarnings("error", category=UnitsWarning)
    """


def assert_valid_unit(unit: str | Unit, context: str) -> None:
    """Raise `ValueError` if `unit` is not parseable by the active registry.

    Used to fail fast at declaration time: a malformed or undefined unit string
    (a typo such as `"degrees_C"`, or `"not_a_unit"`) is rejected as soon as it
    is declared, rather than only when it is later used to validate data.

    Accepts either a bare unit string or a `Unit` marker (its `.unit` string is
    validated), so a consumer that read a `Unit` off an annotation can pass it
    straight through.

    The registry raises a variety of exception types for bad input
    (`pint.UndefinedUnitError`, `AssertionError`, …); all are caught and
    re-raised as a single, clear `ValueError`.

    Args:
        unit: The unit string (or `Unit` marker) to validate.
        context: Names the offending site (e.g. `"mymodel input 'vpd_weekly'"`)
            for the error message.

    Raises:
        ValueError: If `unit` cannot be parsed by the active registry.

    Examples:
        >>> from xarray_annotated.units import assert_valid_unit
        >>> assert_valid_unit("Pa", "test")  # no raise
        >>> assert_valid_unit("degC", "test")  # no raise
    """
    if isinstance(unit, Unit):
        unit = unit.unit
    if len(unit) > _MAX_UNIT_LEN:
        # Refuse before parsing: an over-long unit string can hang pint's parser.
        raise ValueError(
            f"{context}: declared unit is over-long "
            f"({len(unit)} chars > {_MAX_UNIT_LEN}); refusing to parse"
        )
    try:
        get_registry().Unit(unit)
    except Exception as exc:
        # The registry raises several error types for bad input; normalise them.
        raise ValueError(
            f"{context}: declared unit {unit!r} is not a recognised "
            f"unit ({type(exc).__name__}: {exc}){_cf_hint(unit)}"
        ) from exc


def units_compatible(a: str, b: str) -> bool:
    """Return whether two unit strings are *dimensionally* compatible.

    Mirrors the runtime conversion semantics of `check_units`: `"hPa"` and
    `"Pa"` are compatible (one converts to the other), whereas `"Pa"` and
    `"kg"` are not.  Both strings are assumed already validated by
    `assert_valid_unit`.

    Args:
        a: A unit string.
        b: Another unit string.

    Returns:
        `True` if the units are dimensionally compatible.

    Examples:
        >>> from xarray_annotated.units._check import units_compatible
        >>> units_compatible("hPa", "Pa")
        True
        >>> units_compatible("Pa", "kg")
        False
    """
    return get_registry().Unit(a).is_compatible_with(get_registry().Unit(b))


def units_equal(a: str, b: str) -> bool:
    """Return whether two units are the *same* unit (no conversion needed).

    Compares the parsed units, so different spellings of the same unit are equal
    (`"Pa"` == `"pascal"`, `"1"` == `"dimensionless"`) while a prefixed unit
    differs (`"hPa"` != `"Pa"`).  This is the distinction the `on_inexact`
    axis turns on: a value-changing conversion is one where the units are
    compatible but *not* equal; equivalent spellings imply no value change.

    Args:
        a: A unit string.
        b: Another unit string.

    Returns:
        `True` if the units are the same (no conversion needed).

    Examples:
        >>> from xarray_annotated.units._check import units_equal
        >>> units_equal("Pa", "pascal")
        True
        >>> units_equal("hPa", "Pa")
        False
    """
    return get_registry().Unit(a) == get_registry().Unit(b)


def check_units(
    da: xr.DataArray,
    declared: str,
    name: str,
    on_missing: OnMissing | None = None,
    on_inexact: OnInexact | None = None,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate and convert an input `DataArray` to its declared unit.

    Returns a `DataArray` whose data is expressed in `declared` and whose
    `units` attribute equals `declared`.

    Behaviour follows the active `Policy` (`get_policy`); `on_missing` and
    `on_inexact` override their axes for this call when given (`None` defers to
    the policy).  If the policy is disabled the array is returned unchanged.

    Three events are handled, each building on the one before:

    * **No parseable unit** — the input has no `units` attribute, or one the
      registry cannot parse (e.g. a non-CF string like `"fraction"`): follows
      `on_missing` (`"error"` raises, `"warn"` warns and returns unchanged,
      `"ignore"` returns unchanged silently).
    * **Value-changing conversion** — the unit is dimensionally compatible with
      `declared` but not the same unit (e.g. `"hPa"` where `"Pa"` is
      declared): follows `on_inexact` (`"error"` raises, `"warn"` warns then
      converts, `"convert"` converts silently).  Equivalent spellings
      (`"pascal"` for `"Pa"`) imply no value change and always convert.
    * **Dimensional mismatch** — two parseable but incompatible units (e.g. a
      mass where a pressure is declared): always raises
      `pint.DimensionalityError`, regardless of policy.

    Args:
        da: The input `DataArray` to validate and convert.
        declared: The expected unit string (e.g. `"Pa"`).
        name: A label for the array, used in error/warning messages.
        on_missing: Override the on-missing axis for this call (`None` defers
            to the active policy).
        on_inexact: Override the on-inexact axis for this call (`None` defers
            to the active policy).
        qualname: A qualifier prepended to warning messages as `[qualname]`,
            identifying the calling site.

    Returns:
        A new `DataArray` converted to `declared`, with `attrs["units"]` set
        to `declared`.

    Raises:
        ValueError: When `on_missing="error"` and no parseable unit is found,
            or when `on_inexact="error"` and a value-changing conversion would occur.
        pint.DimensionalityError: When the actual unit is dimensionally
            incompatible with `declared` (always, regardless of policy).

    Examples:
        >>> import numpy as np, xarray as xr
        >>> from xarray_annotated.units import check_units
        >>> da = xr.DataArray([1013.0, 1000.0], attrs={"units": "hPa"})
        >>> out = check_units(da, "Pa", "pressure")
        >>> out.attrs["units"]
        'Pa'
        >>> out.values  # 10 * 100 = 1000
        array([101300., 100000.])
    """
    pol = get_policy()
    if not pol.enabled:
        return da
    on_missing = (
        pol.on_missing if on_missing is None else _validate_on_missing(on_missing)
    )
    on_inexact = (
        pol.on_inexact if on_inexact is None else _validate_on_inexact(on_inexact)
    )

    prefix = f"[{qualname}] " if qualname else ""
    have = da.attrs.get("units")
    if have is None:
        if on_missing == "error":
            raise ValueError(
                f"{prefix}input {name!r} has no 'units' attribute "
                f"(declared {declared!r})"
            )
        if on_missing == "warn":
            warnings.warn(
                f"{prefix}input {name!r} unvalidated: no 'units' attribute "
                f"(declared {declared!r})",
                UnitsWarning,
                stacklevel=2,
            )
        return da
    if isinstance(have, str) and len(have) > _MAX_UNIT_LEN:
        # An over-long units attribute can hang pint's parser (DoS); refuse it
        # *before* parsing and route it through on_missing like any other input we
        # cannot validate. Never echo `have` in the message -- it may be huge.
        if on_missing == "error":
            raise ValueError(
                f"{prefix}input {name!r} has an over-long 'units' attribute "
                f"({len(have)} chars > {_MAX_UNIT_LEN}; declared {declared!r}); "
                f"refusing to parse"
            )
        if on_missing == "warn":
            warnings.warn(
                f"{prefix}input {name!r} unvalidated: over-long 'units' attribute "
                f"({len(have)} chars > {_MAX_UNIT_LEN}; declared {declared!r})",
                UnitsWarning,
                stacklevel=2,
            )
        return da
    try:
        get_registry().Unit(have)
    except Exception as exc:
        # A units attribute that exists but the registry cannot parse can no more be
        # validated than a missing one; route it through the same on_missing policy
        # rather than letting an opaque parse error escape (and break a warn run).
        if on_missing == "error":
            raise ValueError(
                f"{prefix}input {name!r} has unparseable 'units' attribute {have!r} "
                f"(declared {declared!r}): {type(exc).__name__}: {exc}"
                f"{_cf_hint(have)}"
            ) from exc
        if on_missing == "warn":
            warnings.warn(
                f"{prefix}input {name!r} unvalidated: unparseable 'units' attribute "
                f"{have!r} (declared {declared!r}){_cf_hint(have)}",
                UnitsWarning,
                stacklevel=2,
            )
        return da
    if units_compatible(have, declared) and not units_equal(have, declared):
        # Dimensionally compatible but value-changing (e.g. hPa -> Pa).
        if on_inexact == "error":
            raise ValueError(
                f"{prefix}input {name!r}: unit {have!r} differs from declared "
                f"{declared!r} and on_inexact='error' forbids value-changing conversion"
            )
        if on_inexact == "warn":
            warnings.warn(
                f"{prefix}input {name!r}: converting {have!r} -> {declared!r} "
                f"(value-changing)",
                UnitsWarning,
                stacklevel=2,
            )
    try:
        converted = da.pint.quantify().pint.to(declared).pint.dequantify()
    except PintExceptionGroup as group:
        # pint-xarray wraps conversion failures in an ExceptionGroup; surface the
        # underlying DimensionalityError directly for a clean, catchable error.
        dim_errors = [
            exc for exc in group.exceptions if isinstance(exc, pint.DimensionalityError)
        ]
        if dim_errors:
            err = dim_errors[0]
            err.add_note(f"while validating input {name!r}")
            raise err from None
        raise
    # dequantify writes pint's canonical unit name (e.g. 'pascal'); restore the
    # declared unit string so downstream re-parsing uses our spelling.
    converted.attrs["units"] = declared
    return converted
