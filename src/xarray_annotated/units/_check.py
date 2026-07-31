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
    OnOutput,
    _validate_on_inexact,
    _validate_on_missing,
    _validate_on_output,
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
    registry then fails to parse and is routed through the usual `on_missing`
    path, exactly like any other unreadable label.

    Args:
        unit: The `pint.Unit` taken from `da.pint.units`.

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

    Compares the parsed units, so different spellings of the same unit are equal
    (`"Pa"` == `"pascal"`, `"1"` == `"dimensionless"`) while a prefixed unit
    differs (`"hPa"` != `"Pa"`).  This is the distinction the `on_inexact`
    axis turns on: a value-changing conversion is one where the units are
    compatible but *not* equal; equivalent spellings imply no value change.

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
        >>> units_equal("hPa", "Pa")
        False
    """
    a = a.unit if isinstance(a, Unit) else a
    b = b.unit if isinstance(b, Unit) else b
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
    quantified = da.pint.units  # None unless the data is a pint Quantity
    if quantified is not None:
        # A pint label lives in the data and cannot go stale, so it wins outright
        # over attrs. It has to: a quantified array normally has *empty* attrs
        # (quantify moves the label into the data), and when a leftover attrs
        # label is also present, quantify() refuses to reconcile the two
        # ("Cannot attach units"), so reading attrs here would crash the pipeline
        # below on exactly the arrays this branch exists to serve.
        have = _quantified_label(quantified)
    else:
        have = da.attrs.get("units")
    if have is None:
        if on_missing == "error":
            raise ValueError(
                f"{prefix}input {name!r} has no 'units' attribute "
                f"(declared {declared!r})"
            )
        if on_missing == "warn":
            # stacklevel=2 targets a direct check_units call; via declare_units it
            # lands on the wrapper line, still identified by the [qualname] prefix.
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
        # quantify() is not a no-op on already-quantified data -- it raises when
        # a leftover attrs label disagrees with the Quantity -- so skip it there.
        source = da if quantified is not None else da.pint.quantify()
        converted = source.pint.to(declared).pint.dequantify()
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


def _apply_output_quantified(
    da: xr.DataArray,
    have: pint.Unit,
    declared: str,
    name: str,
    on_output: OnOutput,
    prefix: str,
) -> xr.DataArray:
    """Apply a declared unit to a *pint-quantified* return value.

    See `apply_output_units` for why this path converts where the `attrs` path
    stamps. The array stays quantified throughout: `dequantify` copies the whole
    buffer even when it has no conversion to do, and a quantified array already
    describes itself, so there is nothing to gain by paying for that.

    Args:
        da: The returned `DataArray`, known to hold a pint `Quantity`.
        have: Its own unit, from `da.pint.units`.
        declared: The declared unit string.
        name: A label for the value, used in error messages.
        on_output: The already-resolved on-output axis.
        prefix: The `[qualname] ` message prefix, or `""`.

    Returns:
        `da` unchanged when it is already in `declared`; otherwise a converted
        array (still quantified).

    Raises:
        pint.DimensionalityError: When the unit cannot be converted to
            `declared` — under `"strict"`, or under `"stamp"` from the
            conversion itself.
        ValueError: Under `"strict"`, when the unit is compatible with but not
            equal to `declared`.
    """
    try:
        label = _quantified_label(have)
        equal = units_equal(label, declared)
    except Exception:
        # A unit undefined in the active registry (the array was quantified
        # under a different one). Unverifiable and unconvertible from here --
        # return it untouched rather than stamping attrs onto a Quantity and
        # creating the two-labels-at-once inconsistency this path exists to
        # avoid.
        return da
    if equal:
        return da
    if on_output == "strict":
        if units_compatible(label, declared):
            raise ValueError(
                f"{prefix}output {name!r}: unit {label!r} differs from declared "
                f"{declared!r} and on_output='strict' requires them to match"
            )
        # Fall through: the conversion below raises DimensionalityError for us.
    try:
        return da.pint.to(declared)
    except PintExceptionGroup as group:
        dim_errors = [
            exc for exc in group.exceptions if isinstance(exc, pint.DimensionalityError)
        ]
        if dim_errors:
            err = dim_errors[0]
            err.add_note(
                f"while applying declared units to output {name!r}: the returned "
                f"array is quantified as {label!r} but declares {declared!r}"
            )
            raise err from None
        raise


def apply_output_units(
    da: xr.DataArray,
    declared: str,
    name: str,
    on_output: OnOutput | None = None,
    qualname: str | None = None,
) -> xr.DataArray:
    """Apply a declared unit to a *returned* `DataArray`, per the on-output axis.

    The output counterpart to `check_units`, and deliberately **not** symmetric
    with it: a declared output carrying an `attrs` label is *stamped*, never
    converted.

    The reason is that xarray's `attrs` are inert under arithmetic.  A body that
    converts by scalar multiplication — `return p * 100.0`, or `flux * F` for a
    molar-mass factor — produces an array still labelled with its *input's* unit,
    because multiplying by a float cannot update a string.  So for the dominant
    idiom the returned label describes neither the values nor the declaration,
    and converting on the strength of it would rescale correct values a second
    time.

    Two modes follow from that:

    - `"stamp"` (default) — overwrite `attrs["units"]` with `declared`, no checks.
      The only sound default, because a stale label is the norm rather than a
      symptom.
    - `"strict"` — a label that is present, parseable, and not equal to
      `declared` raises.

    **`"strict"` is only meaningful for bodies that maintain their own units** —
    pass-through and subsetting functions, or computations on pint-quantified
    arrays.  A body doing manual scalar arithmetic will fail it *whether or not
    it is correct*, so this is an opt-in for unit-aware code, not a general
    verification mode.  (If a body must do manual arithmetic and you still want
    `"strict"` elsewhere, have it drop the stale label — `attrs` cleared, or
    `keep_attrs=False` — since an absent label is always stamped.)

    **Pint-quantified returns are the exception, and are converted rather than
    stamped.**  A `Quantity`'s unit lives in the data, so — unlike `attrs` — it
    cannot be left behind by arithmetic; it is always a true description of the
    values.  Converting on the strength of it is therefore sound, and stamping
    on the strength of it would be actively wrong: writing `attrs["units"]`
    onto a `Quantity` that disagrees produces an array labelled two ways at
    once.  So for a quantified return:

    - `"stamp"` — converts to `declared` if it is not already there, and raises
      `pint.DimensionalityError` if it cannot.  This is the one case where
      `"stamp"` can raise, because here a mismatch is a real error rather than
      the expected consequence of an inert label.
    - `"strict"` — a unit other than `declared` raises, as for `attrs`.

    Either way the array is **returned still quantified**, with `attrs`
    untouched: it already describes itself, and dequantifying it purely to
    write a string would copy the whole buffer.

    Note what no mode can do: distinguish `flux * F` from `flux` when both are
    labelled the same.  Output *values* are never inspected, so a forgotten
    conversion factor is not detectable here — only at a consumer that declares
    the quantity and receives an array that was never stamped.

    An absent, over-long, or unparseable label is always stamped: absence is not
    evidence of a mismatch.

    Args:
        da: The returned `DataArray`.
        declared: The declared unit string (e.g. `"Pa"`).
        name: A label for the value, used in error messages.
        on_output: Override the on-output axis for this call (`None` defers to
            the active policy).
        qualname: A qualifier prepended to messages as `[qualname]`.

    Returns:
        For an `attrs`-labelled or unlabelled array, `da` itself with
        `attrs["units"]` set to `declared` (mutated in place).  For a quantified
        array, an array in `declared` with `attrs` untouched — the same object
        when it was already in `declared`, otherwise a converted new one.  Always
        use the return value rather than relying on mutation.

    Raises:
        pint.DimensionalityError: Under `"strict"`, when the returned array's own
            unit is dimensionally incompatible with `declared`; also under
            `"stamp"` for a quantified array that cannot be converted.
        ValueError: Under `"strict"`, when the returned array's unit is
            compatible with but not equal to `declared`.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units._check import apply_output_units
        >>> da = xr.DataArray([1.0])
        >>> apply_output_units(da, "Pa", "return").attrs["units"]
        'Pa'
    """
    pol = get_policy()
    if not pol.enabled:
        return da
    on_output = pol.on_output if on_output is None else _validate_on_output(on_output)
    prefix = f"[{qualname}] " if qualname else ""

    quantified = da.pint.units
    if quantified is not None:
        return _apply_output_quantified(
            da, quantified, declared, name, on_output, prefix
        )

    have = da.attrs.get("units")
    if on_output == "stamp" or have is None:
        da.attrs["units"] = declared
        return da
    if not isinstance(have, str) or len(have) > _MAX_UNIT_LEN:
        # Unverifiable for the same reasons as on the input path; stamp regardless.
        da.attrs["units"] = declared
        return da
    try:
        get_registry().Unit(have)
    except Exception:
        da.attrs["units"] = declared
        return da

    if not units_equal(have, declared):
        if not units_compatible(have, declared):
            try:
                get_registry().Quantity(1.0, have).to(declared)
            except pint.DimensionalityError as exc:
                raise pint.DimensionalityError(
                    exc.units1,
                    exc.units2,
                    exc.dim1,
                    exc.dim2,
                    extra_msg=(
                        f" ({prefix}output {name!r} is labelled {have!r} but declares "
                        f"{declared!r}, and on_output='strict' requires them to match; "
                        f"if the body converts by scalar arithmetic, clear its 'units' "
                        f"attribute or use on_output='stamp')"
                    ),
                ) from None
        else:
            raise ValueError(
                f"{prefix}output {name!r}: unit {have!r} differs from declared "
                f"{declared!r} and on_output='strict' requires them to match"
            )
    da.attrs["units"] = declared
    return da
