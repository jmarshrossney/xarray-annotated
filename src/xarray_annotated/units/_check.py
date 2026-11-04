"""Runtime unit validation and conversion.

The "usage side" of the package: given a declared unit string (typically read
off a signature by `_annotations`), validate an actual `DataArray` against it
and, where the declaration opts in, convert it into the declared unit.
Conversion goes through the active registry: pint directly for an `attrs`-labelled
array, pint-xarray's `.pint` accessor for a quantified one.

Every conversion is logged to the `xarray_annotated.units` logger at DEBUG.
"""

import logging
import math

import pint
import xarray as xr
from pint_xarray.errors import PintExceptionGroup

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
    `"rad/s"`.  This is the distinction `convert` turns
    on: a value-changing conversion is one where the units are compatible but
    *not* equal, and needs opting into; equivalent spellings imply no value
    change.

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


def check_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate and convert an input `DataArray` to its declared unit.

    Returns a `DataArray` whose data is expressed in `declared` and whose
    `units` attribute equals `declared`.

    A value-changing conversion happens only when the declaration opts in, by
    passing `Unit(..., convert=True)` as `declared`; a plain string never
    converts.  When xarray-annotated is switched off
    (`enabled(False)`) the array is returned unchanged.

    Three events are handled, each building on the one before:

    * **No parseable unit** — the input has no `units` attribute, or one the
      registry cannot parse (e.g. a CF string like `"umol m-2 s-1"` without
      `use_cf_units()`): always raises `ValueError`.  The declaration cannot be
      checked, so the unit should be attached where the data enters.
    * **Value-changing conversion** — the unit is dimensionally compatible with
      `declared` but not the same unit (e.g. `"hPa"` where `"Pa"` is
      declared): converted when opted into, otherwise raises `ValueError`.
      Equivalent spellings
      (`"pascal"` for `"Pa"`) imply no value change and are simply relabelled,
      without touching the data.
    * **Dimensional mismatch** — two parseable but incompatible units (e.g. a
      mass where a pressure is declared): always raises
      `pint.DimensionalityError`, whether or not conversion is opted into.

    Args:
        da: The input `DataArray` to validate and convert.
        declared: The expected unit string (e.g. `"Pa"`), or a `Unit` marker,
            whose `convert` then applies.
        name: A label for the array, used in error/warning messages.
        qualname: A qualifier prepended to error messages as `[qualname]`,
            identifying the calling site.

    Returns:
        A new `DataArray` converted to `declared`, with `attrs["units"]` set
        to `declared`.  When the input is already in `declared` the data is not
        copied — the result is a shallow copy sharing the caller's buffer, so
        declaring a unit an array already has costs nothing, and a lazy array stays
        lazy.

    Raises:
        ValueError: When no parseable unit is found, or when a value-changing
            conversion would occur without `convert`.
        pint.DimensionalityError: When the actual unit is dimensionally
            incompatible with `declared`.

    Examples:
        >>> import numpy as np, xarray as xr
        >>> from xarray_annotated.units import Unit, check_units
        >>> da = xr.DataArray([1013.0, 1000.0], attrs={"units": "hPa"})
        >>> out = check_units(da, Unit("Pa", convert=True), "pressure")
        >>> out.attrs["units"]
        'Pa'
        >>> out.values
        array([101300., 100000.])
    """
    if not resolve_enabled():
        return da
    convert = isinstance(declared, Unit) and declared.convert
    declared = str(declared)

    prefix = f"[{qualname}] " if qualname else ""
    where = f"{prefix}input {name!r}"
    have, quantified = _own_unit(da, declared, where, _ATTACH_UNIT)
    if not quantified and units_equal(have, declared):
        # Same unit, differently spelled ("pascal" for "Pa", "1" for
        # "dimensionless"): nothing to convert, so the conversion below would copy
        # the entire buffer purely to rewrite a string.
        # A shallow copy shares the data and takes its own attrs dict, so the
        # caller's array is left exactly as it was.
        #
        # Quantified inputs still go the long way round: they are contracted to
        # come back dequantified, and only dequantify can do that.
        return _stamped(da, declared)
    if units_compatible(have, declared) and not units_equal(have, declared):
        # Dimensionally compatible but value-changing (e.g. hPa -> Pa).  Whether
        # that is correct depends on the quantity (an annual total is not a daily
        # mean), so it is only done when the declaration opts in.
        if not convert:
            raise ValueError(
                f"{where}: unit {have!r} differs from declared "
                f"{declared!r}. Converting would change the values; if that is "
                f"correct for this quantity, pass "
                f".pint.quantify().pint.to({declared!r}) or declare "
                f"Unit({declared!r}, convert=True)."
            )
        _log.debug(
            "%sinput %r: converting %r -> %r (value-changing, convert=True)",
            prefix,
            name,
            have,
            declared,
        )
    try:
        if quantified:
            converted = da.pint.to(declared).pint.dequantify()
        else:
            # Only the data has a declared unit. quantify() would also parse the
            # coords' labels, so "degrees_north" could abort this conversion.
            data = get_registry().Quantity(da.data, have).to(declared).magnitude
            converted = da.copy(deep=False, data=data)
    except pint.DimensionalityError as err:
        err.add_note(f"while validating input {name!r}")
        raise
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
    # dequantify writes pint's canonical unit name (e.g. 'pascal'), and the plain
    # path still holds the old label; stamp the declared spelling over either.
    converted.attrs["units"] = declared
    return converted


def apply_output_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Check a *returned* `DataArray` against its declared unit.

    A function's author controls what it returns, so an output that does not match
    its declaration is a bug.  Nothing is converted or labelled on the author's
    behalf, and `convert` does not apply here: a marker with `convert=True` can be
    shared with an input, and a mismatch still raises.

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
        would copy the whole buffer.  Always use the return value.

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
        return da if quantified else _stamped(da, declared)
    hint = (
        f"Convert it in the body with .pint.to({declared!r})."
        if quantified
        else (
            f"If the values really are in {declared!r} and the label is stale, state "
            f"that in the body with .assign_attrs(units={declared!r}); otherwise the "
            f"body returned the wrong unit."
        )
    )
    try:
        get_registry().Quantity(1.0, have).to(declared)
    except pint.DimensionalityError as err:
        err.add_note(f"{where} has unit {have!r} but declares {declared!r}. {hint}")
        raise
    raise ValueError(
        f"{where}: unit {have!r} differs from declared {declared!r}. {hint}"
    )
