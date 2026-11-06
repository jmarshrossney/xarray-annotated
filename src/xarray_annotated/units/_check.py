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
from .._errors import ContractError
from ._annotations import Unit
from ._registry import _cf_hint, _label_format, get_registry

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

_STATE_UNIT = (
    "Compute on `xra.units.quantify(...)` so the result carries its unit, or, for "
    "values from outside pint, state the unit with `.assign_attrs(units=...)`."
)


class UnitError(ContractError):
    """Raised for a unit that does not match its declaration, or cannot be checked.

    Deliberately **not** a ``ValueError``: a mismatch (the *data* fails the declaration)
    is a distinct event from a malformed *declaration* (an unrecognised unit string),
    which raises ``ValueError`` from ``assert_valid_unit``.  Keeping them on separate
    hierarchies means catching one never silently swallows the other.

    Nor is it a ``pint.DimensionalityError``, which is a ``TypeError``.  Where units
    cannot convert, the pint error is the ``__cause__``.  A pickle drops that, as when
    crossing a dask process boundary, so the message carries the units and their
    dimensionalities.
    """


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
    if not isinstance(unit, str):
        raise ValueError(f"{context}: declared unit must be a string, got {unit!r}.")
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

    `"hPa"` and `"Pa"` are compatible (one converts to the other), whereas `"Pa"` and
    `"kg"` are not.  Both are assumed already validated by `assert_valid_unit`.

    This is not what `check_units` requires, which refuses `"hPa"` for `"Pa"`.  To ask
    whether one unit would pass for another, use `units_equal`, or
    `declaration_conflicts` for whole declarations.

    Accepts either a bare unit string or a `Unit` marker on each side (its
    `.unit` string is used), so a consumer holding markers — e.g. `Declared`
    values from `declarations_from_signature` — can compare them directly.

    Args:
        a: A unit string or `Unit` marker.
        b: Another unit string or `Unit` marker.

    Returns:
        `True` if the units are dimensionally compatible.

    Raises:
        pint.UndefinedUnitError: If a unit is not defined in the active registry.
        ValueError: If a unit string is malformed.

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
    `.unit` string is used), matching `units_compatible`.  Both are assumed already
    validated by `assert_valid_unit`, as this runs on every unit check.

    Args:
        a: A unit string or `Unit` marker.
        b: Another unit string or `Unit` marker.

    Returns:
        `True` if the units are the same (no conversion needed).

    Raises:
        pint.UndefinedUnitError: If a unit is not defined in the active registry.
        ValueError: If a unit string is malformed.

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
    da: xr.DataArray,
    declared: str | None,
    where: str,
    remedy: str,
    error: type[Exception],
) -> tuple[str, bool]:
    """Read the unit an array says it is in, refusing one that cannot be checked.

    A pint label lives in the data and cannot go stale, so it wins outright over
    attrs. It has to: a quantified array normally has *empty* attrs (quantify moves
    the label into the data), and when a leftover attrs label is also present,
    quantify() refuses to reconcile the two ("Cannot attach units"), so reading attrs
    first would crash the pipeline on exactly the arrays this exists to serve.

    Args:
        da: The array to read.
        declared: The declared unit string, for error messages, if there is one.
        where: Names the array in error messages.
        remedy: What to do about a missing or unreadable unit.
        error: What to raise, as a check and an operation raise different things.

    Returns:
        The unit as a string, and whether the array is quantified.

    Raises:
        Exception: `error`, if the unit is missing, over-long or unparseable.
    """
    quantified = _quantified_units(da)
    have = (
        da.attrs.get("units") if quantified is None else _quantified_label(quantified)
    )
    expected = "" if declared is None else f"declared {declared!r}"
    if have is None:
        about = f" ({expected})" if expected else ""
        raise error(f"{where} has no 'units' attribute{about}. {remedy}")
    if isinstance(have, str) and len(have) > _MAX_UNIT_LEN:
        # An over-long units attribute can hang pint's parser (DoS); refuse it
        # *before* parsing. Never echo `have` in the message -- it may be huge.
        also = f"; {expected}" if expected else ""
        raise error(
            f"{where} has an over-long 'units' attribute "
            f"({len(have)} chars > {_MAX_UNIT_LEN}{also}); "
            f"refusing to parse. {remedy}"
        )
    try:
        get_registry().Unit(have)
    except Exception as exc:
        # A units attribute that exists but the registry cannot parse can no more be
        # validated than a missing one; refuse it with a message rather than letting
        # an opaque parse error escape.
        about = f"({expected}): " if expected else ": "
        raise error(
            f"{where} has unparseable 'units' attribute {have!r} "
            f"{about}{type(exc).__name__}: {exc}{_cf_hint(have)}. {remedy}"
        ) from exc
    return have, quantified is not None


def _refuse(have: str, declared: str, where: str, hint: str, fix: str) -> NoReturn:
    """Raise for a unit that is not the declared one, by whether the two convert.

    Args:
        have: The array's unit.
        declared: The declared unit.
        where: Names the array in the message.
        hint: What to do about a unit that converts.
        fix: What to do about one that does not.

    Raises:
        UnitError: Always, from the pint error if `have` cannot convert to `declared`.
    """
    registry = get_registry()
    try:
        registry.Quantity(1.0, have).to(declared)
    except pint.DimensionalityError as err:
        raise UnitError(
            f"{where}: unit {have!r} ({registry.Unit(have).dimensionality}) cannot "
            f"convert to declared {declared!r} "
            f"({registry.Unit(declared).dimensionality}). {fix}"
        ) from err
    raise UnitError(
        f"{where}: unit {have!r} differs from declared {declared!r}. {hint}"
    )


def _check_input_unit(
    da: xr.DataArray, declared: str | Unit, name: str, qualname: str | None
) -> None:
    """`check_units` on a validated declaration, whatever `enabled` says."""
    declared = str(declared)
    prefix = f"[{qualname}] " if qualname else ""
    where = f"{prefix}input {name!r}"
    have, _ = _own_unit(da, declared, where, _ATTACH_UNIT, UnitError)
    if not units_equal(have, declared):
        _refuse(
            have,
            declared,
            where,
            f"Converting would change the values; if that is correct for this "
            f"quantity, pass xra.units.convert(da, {declared!r}).",
            "Fix the data or the declaration.",
        )


def check_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Check an input `DataArray` against its declared unit; return it unchanged.

    A different unit raises, however it could be converted, because whether a
    conversion is correct depends on the quantity.  Convert before the call, with
    `convert`.  When xarray-annotated is switched off (`enabled(False)`) nothing is
    checked.

    Args:
        da: The input `DataArray` to check.
        declared: The expected unit string (e.g. `"Pa"`), or a `Unit` marker.
        name: A label for the array, used in error messages.
        qualname: A qualifier prepended to error messages as `[qualname]`,
            identifying the calling site.

    Returns:
        `da` itself: quantified if it was, with its label spelled as it was.

    Raises:
        UnitError: When the array's unit is missing, over-long or unparseable, or is not
            equal to `declared`.
        ValueError: When `declared` is not a recognised unit, or
            `XARRAY_ANNOTATED_ENABLED` holds an unrecognised value.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import check_units
        >>> da = xr.DataArray([1013.0], attrs={"units": "pascal"})
        >>> check_units(da, "Pa", "pressure") is da
        True
    """
    if not resolve_enabled():
        return da
    prefix = f"[{qualname}] " if qualname else ""
    assert_valid_unit(declared, f"{prefix}input {name!r}")
    _check_input_unit(da, declared, name, qualname)
    return da


def quantify(da: xr.DataArray) -> xr.DataArray:
    """Quantify a `DataArray`'s data from its `units` label, leaving coordinates alone.

    Unlike `.pint.quantify()`, which also quantifies every coordinate with a `units`
    label: on one in degrees, `.sel(lat=20.0, method="nearest")` then reads the bare
    number as radians and silently selects the wrong row, and a label the registry
    cannot parse stops the call.  A missing label raises here, where `.pint.quantify()`
    returns the array unquantified.

    This is not a check, so `enabled(False)` does not affect it.

    Args:
        da: The array, labelled in `attrs["units"]` or already quantified.

    Returns:
        `da` itself if it is already quantified.  Otherwise a shallow copy holding a
        `Quantity`, with no `units` in its attrs.  A dask array stays lazy, but a lazily
        opened array that dask does not back is read into memory, so open files with
        `chunks={}` to avoid that.

    Raises:
        ValueError: When the label is missing, over-long or unparseable.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import quantify
        >>> p = quantify(xr.DataArray([101325.0], attrs={"units": "Pa"}))
        >>> p.pint.to("hPa").data.magnitude.tolist()
        [1013.25]
    """
    have, quantified = _own_unit(
        da, None, "the array to quantify", _ATTACH_UNIT, ValueError
    )
    if quantified:
        return da
    out = da.copy(deep=False, data=get_registry().Quantity(da.data, have))
    del out.attrs["units"]
    return out


def dequantify(da: xr.DataArray) -> xr.DataArray:
    """Return a plain `DataArray` labelled with its unit, ready to write to a file.

    Unlike `.pint.dequantify()`, this clears the array's `.encoding`.  Packing such as
    `scale_factor` may describe other values, since `.pint.to` keeps it, and writing
    with it silently corrupts these.  Set compression and packing again before writing.
    Coordinates keep their encoding, so `time` is written as it was read.

    The label uses CF spelling (`"hPa"`, `"g d-1 m-2"`) when `use_cf_units` is active,
    and otherwise pint's long form (`"hectopascal"`), so the active registry can always
    read it back.

    This is not a check, so `enabled(False)` does not affect it.

    Args:
        da: The array, quantified or plain.

    Returns:
        `da` itself if it is not quantified.  Otherwise a copy holding the magnitudes,
        which is a new buffer unless dask backs it, with empty `.encoding`.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import dequantify, quantify
        >>> p = quantify(xr.DataArray([101325.0], attrs={"units": "Pa"}))
        >>> dequantify(p.pint.to("hPa")).attrs["units"]
        'hectopascal'
    """
    if _quantified_units(da) is None:
        return da
    plain = da.pint.dequantify(format=_label_format())
    plain.encoding = {}
    return plain


def convert(da: xr.DataArray, unit: str | Unit) -> xr.DataArray:
    """Convert a `DataArray` to `unit`, where you know that is correct for the data.

    Pint knows whether two units are compatible, but not whether converting is right
    for this quantity, so `enforce` never does this for you.

    The result is the same kind of array as `da`.  Only the data is converted, so
    coordinates and their labels are left alone.  Every conversion is logged at DEBUG on
    the `xarray_annotated.units` logger.

    This is not a check, so `enabled(False)` does not affect it.

    Args:
        da: The array, labelled in `attrs["units"]` or quantified.
        unit: The unit to convert to.

    Returns:
        A quantified array in `unit` if `da` is quantified, otherwise a plain one with
        `attrs["units"]` spelled as `unit`.  A converted array has a new buffer and
        empty `.encoding`, as packing that described the old values would corrupt these
        on write.  A quantified array already in `unit` shares its buffer and also has
        empty `.encoding`, since `.pint.to` may have left a stale packing on it.  A
        plain array already in `unit` shares its buffer, keeps its `.encoding` and stays
        lazy.

    Raises:
        ValueError: When `unit` is not a recognised unit, or the array's own unit is
            missing, over-long or unparseable.
        pint.DimensionalityError: When the two units are dimensionally incompatible.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import convert, quantify
        >>> da = xr.DataArray([1013.0], attrs={"units": "hPa"})
        >>> converted = convert(da, "Pa")
        >>> converted.values, converted.attrs["units"]
        (array([101300.]), 'Pa')
        >>> convert(quantify(da), "Pa").data.magnitude
        array([101300.])
    """
    unit = str(unit)
    assert_valid_unit(unit, "convert")
    have, quantified = _own_unit(
        da, unit, "the array to convert", _ATTACH_UNIT, ValueError
    )
    same = units_equal(have, unit)
    if same and not quantified:
        return _stamped(da, unit)
    magnitude = da.data.magnitude if quantified else da.data
    try:
        data = get_registry().Quantity(magnitude, have).to(unit)
    except pint.DimensionalityError as err:
        err.add_note(f"while converting {da.name!r}")
        raise
    if not same:
        _log.debug("converting %r: %r -> %r", da.name, have, unit)
    if quantified:
        converted = da.copy(deep=False, data=data)
        converted.attrs.pop(
            "units", None
        )  # a leftover label, which a Quantity outranks
    else:
        converted = _stamped(da.copy(deep=False, data=data.magnitude), unit)
    converted.encoding = {}
    return converted


def _check_output_unit(
    da: xr.DataArray, declared: str | Unit, name: str, qualname: str | None
) -> None:
    """`check_output_units` on a validated declaration, whatever `enabled` says."""
    declared = str(declared)
    prefix = f"[{qualname}] " if qualname else ""
    where = f"{prefix}output {name!r}"

    have, quantified = _own_unit(da, declared, where, _STATE_UNIT, UnitError)
    if units_equal(have, declared):
        return
    hint = (
        f"Convert it in the body with xra.units.convert(da, {declared!r})."
        if quantified
        else (
            f"A label does not follow arithmetic: compute on xra.units.quantify(...) "
            f"and convert with xra.units.convert(da, {declared!r}). If the values come "
            f"from outside pint and really are in {declared!r}, state that with "
            f".assign_attrs(units={declared!r})."
        )
    )
    _refuse(have, declared, where, hint, "Fix the body or the declaration.")


def check_output_units(
    da: xr.DataArray,
    declared: str | Unit,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Check a *returned* `DataArray` against its declared unit; return it unchanged.

    A function's author controls what it returns, so an output that does not match
    its declaration is a bug.  Nothing is converted or labelled on the author's behalf.

    Two things can leave a body's label out of step with its values.  `attrs` are
    inert under arithmetic, so `p * 100.0` keeps its `hPa` label, and values are never
    inspected, so a label set with `.assign_attrs(units=...)` is taken on trust.  A
    pint-quantified array carries its unit in the data, so neither can happen to it;
    a body converts one with `convert`.

    When xarray-annotated is switched off (`enabled(False)`) nothing is checked.

    Args:
        da: The returned `DataArray`.
        declared: The declared unit string (e.g. `"Pa"`), or a `Unit` marker.
        name: A label for the value, used in error messages.
        qualname: A qualifier prepended to messages as `[qualname]`.

    Returns:
        `da` itself: quantified if it was, with its label spelled as it was.

    Raises:
        UnitError: When the array's unit is missing, over-long or unparseable, or is not
            equal to `declared`.
        ValueError: When `declared` is not a recognised unit, or
            `XARRAY_ANNOTATED_ENABLED` holds an unrecognised value.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units import check_output_units
        >>> da = xr.DataArray([1.0], attrs={"units": "pascal"})
        >>> check_output_units(da, "Pa", "return") is da
        True
    """
    if not resolve_enabled():
        return da
    prefix = f"[{qualname}] " if qualname else ""
    assert_valid_unit(declared, f"{prefix}output {name!r}")
    _check_output_unit(da, declared, name, qualname)
    return da
