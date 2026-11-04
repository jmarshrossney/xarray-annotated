"""Runtime unit validation, and explicit conversion.

The "usage side" of the package: given a declared unit string (typically read off a
signature by `_annotations`), validate an actual `DataArray` against it.  Checking never
changes values; `convert` does, where the caller says so.

Every conversion is logged to the `xarray_annotated.units` logger at DEBUG.
"""

import logging
import math
from typing import NoReturn

import pint
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import xarray as xr

from .._config import resolve_enabled
from ._annotations import Unit
from ._registry import _cf_hint, get_registry

_log = logging.getLogger("xarray_annotated.units")

#: Upper bound on the length of a unit string we are willing to hand to pint's
#: parser. Real units are short (``"umol m-2 s-1"`` is 12 chars); anything far
#: longer is not a unit. pint's parser is not hardened against adversarial input
#: and can hang on pathological strings (e.g. a single very long token), so an
#: over-long ``units`` attribute read from an untrusted file must be refused
#: *before* it reaches the parser rather than after. This does not defend against
#: every DoS input pint accepts (a short string can still be pathological, e.g.
#: nested numeric exponentiation), only the length-driven class.
_MAX_UNIT_LEN = 256

_ATTACH_UNIT = (
    "Attach the unit where the data enters the pipeline "
    "(`.assign_attrs(units=...)`), or switch checking off with "
    "`xra.enabled(False)`."
)

_STATE_UNIT = "State the unit the values are in with `.assign_attrs(units=...)`."


def _stamped(da: xr.DataArray, declared: str) -> xr.DataArray:
    """A copy of `da` labelled `declared`, sharing its data but not its attrs."""
    stamped = da.copy(deep=False)
    stamped.attrs["units"] = declared
    return stamped


def _quantified_units(da: xr.DataArray) -> pint.Unit | None:
    """Return a quantified array's own pint unit, or `None`, without loading it.

    `da.pint.units` goes through `da.data`, which reads a whole lazily opened variable
    from disk just to find that it holds no `Quantity`.  `Variable._data` is the
    wrapper itself, so asking it reads nothing.  It is private to xarray, so
    `TestLazyArrays` in `test_units.py` guards this.
    """
    return getattr(da.variable._data, "units", None)


def _quantified_label(unit: pint.Unit) -> str:
    """Render a quantified array's own pint unit as a unit string.

    Compact-pretty (`"hPa"`, `"µmol/m²/s"`, `"°C"`) rather than pint's canonical
    long form, because this string is what users see in error messages. It
    round-trips through the registry -- pint's parser accepts the Unicode
    superscripts and `µ` it emits -- so it can be compared and re-parsed like
    any label read from `attrs`.

    Going back through a *string* rather than passing the `pint.Unit` object on
    is deliberate: it keeps every comparison inside this package's registry,
    even when the array was quantified under a different one (quantities from
    two registries cannot be mixed). A unit that is undefined in the active
    registry then fails to parse and is refused like any other unreadable
    label.

    Args:
        unit: The `pint.Unit` taken from `_quantified_units`.

    Returns:
        The unit as a compact-pretty string.
    """
    return f"{unit:~P}"


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


def units_compatible(a: str | Unit, b: str | Unit) -> bool:
    """Return whether two units are *dimensionally* compatible.

    Mirrors the runtime conversion semantics of `check_units`: `"hPa"` and
    `"Pa"` are compatible (one converts to the other), whereas `"Pa"` and
    `"kg"` are not.  Both are assumed already validated by `assert_valid_unit`.

    Accepts either a bare unit string or a `Unit` marker on each side (its
    `.unit` string is used), so a consumer holding markers — e.g. `Declared`
    values from `declarations_from_signature` — can compare them directly.

    Args:
        a: A unit string or `Unit` marker.
        b: Another unit string or `Unit` marker.

    Returns:
        `True` if the units are dimensionally compatible.

    Examples:
        >>> from xarray_annotated.units import Unit, units_compatible
        >>> units_compatible("hPa", "Pa")
        True
        >>> units_compatible(Unit("hPa"), "Pa")  # markers accepted too
        True
        >>> units_compatible("Pa", "kg")
        False
    """
    a = a.unit if isinstance(a, Unit) else a
    b = b.unit if isinstance(b, Unit) else b
    return get_registry().Unit(a).is_compatible_with(get_registry().Unit(b))


def units_equal(a: str | Unit, b: str | Unit) -> bool:
    """Return whether two units are the *same* unit (no conversion needed).

    Compares what converting would do to the values, so different spellings and
    compositions of one unit are equal (`"Pa"` == `"pascal"` == `"N/m**2"`,
    `"1"` == `"dimensionless"`) while a prefixed unit differs (`"hPa"` != `"Pa"`).
    pint treats radians as dimensionless, so `"rad"` == `"1"` and `"Hz"` ==
    `"rad/s"`.  An input must be equal to its declared unit, not merely compatible:
    equivalent spellings imply no value change.

    Accepts either a bare unit string or a `Unit` marker on each side (its
    `.unit` string is used), matching `units_compatible`.

    Args:
        a: A unit string or `Unit` marker.
        b: Another unit string or `Unit` marker.

    Returns:
        `True` if the units are the same (no conversion needed).

    Examples:
        >>> from xarray_annotated.units import Unit, units_equal
        >>> units_equal("Pa", "pascal")
        True
        >>> units_equal(Unit("Pa"), Unit("pascal"))  # markers accepted too
        True
        >>> units_equal("Pa", "kg/m/s**2")
        True
        >>> units_equal("hPa", "Pa")
        False
    """
    a = a.unit if isinstance(a, Unit) else a
    b = b.unit if isinstance(b, Unit) else b
    registry = get_registry()
    unit_a, unit_b = registry.Unit(a), registry.Unit(b)
    if unit_a == unit_b:
        return True
    if not unit_a.is_compatible_with(unit_b):
        return False
    try:
        # Two points, because an offset unit can agree with another at one of them.
        return all(
            math.isclose(
                registry.Quantity(x, unit_a).to(unit_b).magnitude,
                x,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for x in (0.0, 1.0)
        )
    except pint.DimensionalityError:
        return False  # compatible by dimension, yet no conversion: degC, delta_degC


def _own_unit(
    da: xr.DataArray, declared: str, where: str, remedy: str
) -> tuple[str, bool]:
    """Read the unit an array says it is in, refusing one that cannot be checked.

    A pint label lives in the data and cannot go stale, so it wins outright over
    attrs. It has to: a quantified array normally has *empty* attrs (quantify moves
    the label into the data), and when a leftover attrs label is also present,
    quantify() refuses to reconcile the two ("Cannot attach units"), so reading attrs
    first would crash the pipeline on exactly the arrays this exists to serve.

    Args:
        da: The array to read.
        declared: The declared unit string, for error messages.
        where: Names the array in error messages.
        remedy: What to do about a missing or unreadable unit.

    Returns:
        The unit as a string, and whether the array is quantified.

    Raises:
        ValueError: If the unit is missing, over-long or unparseable.
    """
    quantified = _quantified_units(da)
    have = (
        da.attrs.get("units") if quantified is None else _quantified_label(quantified)
    )
    if have is None:
        raise ValueError(
            f"{where} has no 'units' attribute (declared {declared!r}). {remedy}"
        )
    if isinstance(have, str) and len(have) > _MAX_UNIT_LEN:
        # An over-long units attribute can hang pint's parser (DoS); refuse it
        # *before* parsing. Never echo `have` in the message -- it may be huge.
        raise ValueError(
            f"{where} has an over-long 'units' attribute "
            f"({len(have)} chars > {_MAX_UNIT_LEN}; declared {declared!r}); "
            f"refusing to parse. {remedy}"
        )
    try:
        get_registry().Unit(have)
    except Exception as exc:
        # A units attribute that exists but the registry cannot parse can no more be
        # validated than a missing one; refuse it with a message rather than letting
        # an opaque parse error escape.
        raise ValueError(
            f"{where} has unparseable 'units' attribute {have!r} "
            f"(declared {declared!r}): {type(exc).__name__}: {exc}"
            f"{_cf_hint(have)}. {remedy}"
        ) from exc
    return have, quantified is not None


def _refuse(have: str, declared: str, where: str, hint: str, note: str) -> NoReturn:
    """Raise for a unit that is not the declared one, by whether the two convert.

    Raises:
        pint.DimensionalityError: If `have` cannot be converted to `declared`.
        ValueError: If it can, as the values would change.
    """
    try:
        get_registry().Quantity(1.0, have).to(declared)
    except pint.DimensionalityError as err:
        err.add_note(note)
        raise
    raise ValueError(
        f"{where}: unit {have!r} differs from declared {declared!r}. {hint}"
    )


def _dequantified(da: xr.DataArray) -> xr.DataArray:
    """A plain copy of a quantified array, with no packing.

    Its `.encoding` may describe other values, as `.pint.to` keeps it, and that would
    corrupt these on write.
    """
    plain = da.pint.dequantify()
    plain.encoding = {}
    return plain


def check_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate an input `DataArray` against its declared unit.

    Values never change: a different unit raises, however it could be converted,
    because whether a conversion is correct depends on the quantity.  Convert before
    the call, with `convert`.  When xarray-annotated is switched off (`enabled(False)`)
    the array is returned unchanged.

    Args:
        da: The input `DataArray` to validate.
        declared: The expected unit string (e.g. `"Pa"`), or a `Unit` marker.
        name: A label for the array, used in error messages.
        qualname: A qualifier prepended to error messages as `[qualname]`,
            identifying the calling site.

    Returns:
        A plain `DataArray` with `attrs["units"]` spelled as `declared`, sharing the
        caller's buffer, so declaring a unit an array already has costs nothing and a
        lazy array stays lazy.  A quantified array is dequantified, which copies it,
        and loses its `.encoding`, as packing that described other values would
        corrupt these on write.

    Raises:
        ValueError: When the unit is missing, over-long or unparseable, or compatible
            with but not equal to `declared`.
        pint.DimensionalityError: When the unit is dimensionally incompatible with
            `declared`.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import check_units
        >>> da = xr.DataArray([1013.0], attrs={"units": "pascal"})
        >>> check_units(da, "Pa", "pressure").attrs["units"]
        'Pa'
    """
    if not resolve_enabled():
        return da
    declared = str(declared)
    prefix = f"[{qualname}] " if qualname else ""
    where = f"{prefix}input {name!r}"
    have, quantified = _own_unit(da, declared, where, _ATTACH_UNIT)
    if not units_equal(have, declared):
        _refuse(
            have,
            declared,
            where,
            f"Converting would change the values; if that is correct for this "
            f"quantity, pass xra.units.convert(da, {declared!r}).",
            f"while validating input {name!r}",
        )
    # Same unit, perhaps spelled differently ("pascal" for "Pa"): nothing to convert.
    return _stamped(_dequantified(da) if quantified else da, declared)


def convert(da: xr.DataArray, unit: str | Unit) -> xr.DataArray:
    """Convert a `DataArray` to `unit`, where you know that is correct for the data.

    Pint knows whether two units are compatible, but not whether converting is right
    for this quantity, so `enforce` never does this for you.

    Only the data is converted: coordinates and their labels are left alone, so a
    `degrees_north` label does not stop the conversion the way `.pint.quantify()` does.
    Every conversion is logged at DEBUG on the `xarray_annotated.units` logger.

    This is not a check, so `enabled(False)` does not affect it.

    Args:
        da: The array, labelled in `attrs["units"]` or quantified.
        unit: The unit to convert to.

    Returns:
        A plain `DataArray` with `attrs["units"]` spelled as `unit`.  A converted array
        has a new buffer and empty `.encoding`, as packing that described the old values
        would corrupt these on write.  An array already in `unit` is treated as by
        `check_units`.

    Raises:
        ValueError: When `unit` is not a recognised unit, or the array's own unit is
            missing, over-long or unparseable.
        pint.DimensionalityError: When the two units are dimensionally incompatible.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import convert
        >>> da = xr.DataArray([1013.0], attrs={"units": "hPa"})
        >>> converted = convert(da, "Pa")
        >>> converted.values, converted.attrs["units"]
        (array([101300.]), 'Pa')
    """
    unit = str(unit)
    assert_valid_unit(unit, "convert")
    have, quantified = _own_unit(da, unit, "the array to convert", _STATE_UNIT)
    plain = _dequantified(da) if quantified else da
    if units_equal(have, unit):
        return _stamped(plain, unit)
    try:
        data = get_registry().Quantity(plain.data, have).to(unit).magnitude
    except pint.DimensionalityError as err:
        err.add_note(f"while converting {da.name!r}")
        raise
    _log.debug("converting %r: %r -> %r", da.name, have, unit)
    converted = _stamped(plain.copy(deep=False, data=data), unit)
    converted.encoding = {}
    return converted


def apply_output_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Check a *returned* `DataArray` against its declared unit.

    A function's author controls what it returns, so an output that does not match
    its declaration is a bug.  Nothing is converted or labelled on the author's behalf.

    Two things can leave a body's label out of step with its values.  `attrs` are
    inert under arithmetic, so `p * 100.0` keeps its `hPa` label; a body that really
    changes the unit states the new one with `.assign_attrs(units=...)`.  Values are
    never inspected, so that statement is taken on trust.  A pint-quantified array
    carries its unit in the data, so that cannot happen to it; a body converts one
    with `.pint.to(...)`.

    When xarray-annotated is switched off (`enabled(False)`) the array is returned
    unchanged.

    Args:
        da: The returned `DataArray`.
        declared: The declared unit string (e.g. `"Pa"`), or a `Unit` marker.
        name: A label for the value, used in error messages.
        qualname: A qualifier prepended to messages as `[qualname]`.

    Returns:
        For a labelled array, a shallow copy of `da` with `attrs["units"]` respelled
        as `declared`; `da` itself is left as it was, since a body may have returned
        the caller's own array.  For a quantified array, `da` itself, as dequantifying
        would copy the whole buffer, or a shallow copy with empty `.encoding` if it has
        any.  Always use the return value.

    Raises:
        pint.DimensionalityError: When the unit is dimensionally incompatible with
            `declared`.
        ValueError: When the unit is missing or unparseable, or compatible with but
            not equal to `declared`.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import apply_output_units
        >>> da = xr.DataArray([1.0], attrs={"units": "pascal"})
        >>> apply_output_units(da, "Pa", "return").attrs["units"]
        'Pa'
    """
    if not resolve_enabled():
        return da
    declared = str(declared)
    prefix = f"[{qualname}] " if qualname else ""
    where = f"{prefix}output {name!r}"

    have, quantified = _own_unit(da, declared, where, _STATE_UNIT)
    if units_equal(have, declared):
        if not quantified:
            return _stamped(da, declared)
        if not da.encoding:
            return da
        # As for an input: `.pint.to` keeps a packing that describes the old values.
        unpacked = da.copy(deep=False)
        unpacked.encoding = {}
        return unpacked
    hint = (
        f"Convert it in the body with .pint.to({declared!r})."
        if quantified
        else (
            f"If the values really are in {declared!r} and the label is stale, state "
            f"that in the body with .assign_attrs(units={declared!r}); otherwise the "
            f"body returned the wrong unit."
        )
    )
    _refuse(
        have,
        declared,
        where,
        hint,
        f"{where} has unit {have!r} but declares {declared!r}. {hint}",
    )
