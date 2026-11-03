"""Runtime unit validation and conversion.

The "usage side" of the package: given a declared unit string (typically read
off a signature by `_annotations`), validate an actual `DataArray` against it
and, where the declaration opts in, convert it into the declared unit.
Conversion is delegated to pint-xarray's `.pint` accessor via the active
registry.

Every non-trivial action is logged at DEBUG to the `xarray_annotated.units`
logger: an opted-in input conversion, an output stamped over a different label,
a converted quantified output.
"""

import logging
from typing import Any

import pint
import xarray as xr
from pint_xarray.errors import PintExceptionGroup

from .._config import resolve_enabled
from ._annotations import Unit
from ._config import (
    DEFAULT_ON_OUTPUT,
    OnOutput,
    _validate_on_output,
)
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


def resolve_on_output(declared: str | Unit, on_output: OnOutput | None) -> OnOutput:
    """The on-output axis that applies to `declared` (`None` defers to the default).

    Resolves marker override > call keyword > built-in default.  The keyword is
    validated even when the marker overrides it, so a bad keyword is never
    masked by the declaration it happens to be used with.
    """
    if on_output is not None:
        _validate_on_output(on_output)
    marker = declared.on_output if isinstance(declared, Unit) else None
    return marker or on_output or DEFAULT_ON_OUTPUT


def _labels_differ(have: Any, declared: str) -> bool:
    """Whether a label names a different unit; unparseable labels count as different."""
    try:
        return not units_equal(have, declared)
    except Exception:
        return True


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
    differs (`"hPa"` != `"Pa"`).  This is the distinction `convert` turns
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
        >>> units_equal("hPa", "Pa")
        False
    """
    a = a.unit if isinstance(a, Unit) else a
    b = b.unit if isinstance(b, Unit) else b
    return get_registry().Unit(a) == get_registry().Unit(b)


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
        declaring a unit an array already has costs nothing.

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
        >>> out.values  # 10 * 100 = 1000
        array([101300., 100000.])
    """
    if not resolve_enabled():
        return da
    convert = isinstance(declared, Unit) and declared.convert
    declared = str(declared)

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
        raise ValueError(
            f"{prefix}input {name!r} has no 'units' attribute "
            f"(declared {declared!r}). {_ATTACH_UNIT}"
        )
    if isinstance(have, str) and len(have) > _MAX_UNIT_LEN:
        # An over-long units attribute can hang pint's parser (DoS); refuse it
        # *before* parsing. Never echo `have` in the message -- it may be huge.
        raise ValueError(
            f"{prefix}input {name!r} has an over-long 'units' attribute "
            f"({len(have)} chars > {_MAX_UNIT_LEN}; declared {declared!r}); "
            f"refusing to parse. {_ATTACH_UNIT}"
        )
    try:
        get_registry().Unit(have)
    except Exception as exc:
        # A units attribute that exists but the registry cannot parse can no more be
        # validated than a missing one; refuse it with a message rather than letting
        # an opaque parse error escape.
        raise ValueError(
            f"{prefix}input {name!r} has unparseable 'units' attribute {have!r} "
            f"(declared {declared!r}): {type(exc).__name__}: {exc}"
            f"{_cf_hint(have)}. {_ATTACH_UNIT}"
        ) from exc
    if quantified is None and units_equal(have, declared):
        # Same unit, differently spelled ("pascal" for "Pa", "1" for
        # "dimensionless"): nothing to convert, so the quantify/convert/dequantify
        # round-trip below would copy the entire buffer purely to rewrite a string
        # -- and would rewrite the *coordinates'* unit spellings on the way past.
        # A shallow copy shares the data and takes its own attrs dict, so the
        # caller's array is left exactly as it was.
        #
        # Quantified inputs still go the long way round: they are contracted to
        # come back dequantified, and only dequantify can do that.
        relabelled = da.copy(deep=False)
        relabelled.attrs["units"] = declared
        return relabelled
    if units_compatible(have, declared) and not units_equal(have, declared):
        # Dimensionally compatible but value-changing (e.g. hPa -> Pa).  Whether
        # that is correct depends on the quantity (an annual total is not a daily
        # mean), so it is only done when the declaration opts in.
        if not convert:
            raise ValueError(
                f"{prefix}input {name!r}: unit {have!r} differs from declared "
                f"{declared!r}. Converting would change the values; if that is "
                f"correct for this quantity, declare "
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
            `declared` — under `"strict"`, or under `"stamp"` or `"require"`
            from the conversion itself.
        ValueError: Under `"strict"`, when the unit is compatible with but not
            equal to `declared`; under `"require"`, when the active registry
            cannot parse the unit.
    """
    try:
        label = _quantified_label(have)
        equal = units_equal(label, declared)
    except Exception:
        # A unit undefined in the active registry (the array was quantified
        # under a different one). Unverifiable and unconvertible from here --
        # return it untouched rather than stamping attrs onto a Quantity and
        # creating the two-labels-at-once inconsistency this path exists to
        # avoid.  "require" refuses it, as it does an unparseable attrs label.
        if on_output == "require":
            raise ValueError(
                _require_message(
                    prefix, name, declared, f"an unparseable unit {str(have)!r}"
                )
            ) from None
        _log.info(
            "%soutput %r: quantified unit %r is unparseable in the active "
            "registry; returned unchecked (declared %r, on_output=%r)",
            prefix,
            name,
            str(have),
            declared,
            on_output,
        )
        return da
    if equal:
        return da
    # Only *compatible* units raise here; an incompatible label falls through to the
    # conversion below, which raises DimensionalityError for us.
    if on_output == "strict" and units_compatible(label, declared):
        raise ValueError(
            f"{prefix}output {name!r}: unit {label!r} differs from declared "
            f"{declared!r} and on_output='strict' requires them to match"
        )
    _log.debug(
        "%soutput %r: converting quantified %r -> declared %r",
        prefix,
        name,
        label,
        declared,
    )
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
    declared: str | Unit,
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

    Three modes follow from that:

    - `"stamp"` (default) — overwrite `attrs["units"]` with `declared`, no checks.
      The only sound default, because a stale label is the norm rather than a
      symptom.
    - `"strict"` — a label that is present, parseable, and not equal to
      `declared` raises.
    - `"require"` — as `"strict"`, and additionally an output with **no unit of
      its own** (no label, or one that cannot be parsed) raises instead of being
      stamped.  For wrappers around libraries that return bare arrays, where the
      declaration is a claim about someone else's convention: attach the unit the
      library actually returns (`.assign_attrs(units=...).pint.quantify()`) and
      the declared unit is checked and converted to.

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
    - `"require"` — as `"stamp"`: a quantified array carries its own unit, which
      is all `"require"` asks for.

    Either way the array is **returned still quantified**, with `attrs`
    untouched: it already describes itself, and dequantifying it purely to
    write a string would copy the whole buffer.

    Note what no mode can do: distinguish `flux * F` from `flux` when both are
    labelled the same.  Output *values* are never inspected, so a forgotten
    conversion factor is not detectable here.  Nor is it detectable downstream
    once the output is stamped: a consumer then compares one declaration against
    another, never against the values.  Only `"require"` keeps an output's unit
    tied to something other than its own declaration.

    Under `"stamp"` and `"strict"`, an absent, over-long, or unparseable label is
    stamped: absence is not evidence of a mismatch.

    Args:
        da: The returned `DataArray`.
        declared: The declared unit string (e.g. `"Pa"`), or a `Unit` marker,
            whose `on_output` override then applies.
        name: A label for the value, used in error messages.
        on_output: Set the on-output axis for this call (`None` takes the
            default, `"stamp"`); a marker's own override wins.
        qualname: A qualifier prepended to messages as `[qualname]`.

    Returns:
        For an `attrs`-labelled or unlabelled array, `da` itself with
        `attrs["units"]` set to `declared` (mutated in place).  For a quantified
        array, an array in `declared` with `attrs` untouched — the same object
        when it was already in `declared`, otherwise a converted new one.  Always
        use the return value rather than relying on mutation.

    Raises:
        pint.DimensionalityError: Under `"strict"` or `"require"`, when the
            returned array's own unit is dimensionally incompatible with
            `declared`; also under `"stamp"` or `"require"` for a quantified array
            that cannot be converted.
        ValueError: Under `"strict"`, when the returned array's unit is
            compatible with but not equal to `declared`; likewise under
            `"require"` for an `attrs`-labelled array (a quantified one is
            converted instead).  Under `"require"`, also when it carries no
            parseable unit of its own.

    Examples:
        >>> import xarray as xr
        >>> from xarray_annotated.units._check import apply_output_units
        >>> da = xr.DataArray([1.0])
        >>> apply_output_units(da, "Pa", "return").attrs["units"]
        'Pa'
    """
    if not resolve_enabled():
        return da
    on_output = resolve_on_output(declared, on_output)
    declared = str(declared)
    prefix = f"[{qualname}] " if qualname else ""

    quantified = da.pint.units
    if quantified is not None:
        return _apply_output_quantified(
            da, quantified, declared, name, on_output, prefix
        )

    have = da.attrs.get("units")
    if have is None:
        if on_output == "require":
            raise ValueError(_require_message(prefix, name, declared, "no unit"))
        da.attrs["units"] = declared
        return da
    if on_output == "stamp":
        if _log.isEnabledFor(logging.DEBUG) and _labels_differ(have, declared):
            _log.debug(
                "%soutput %r: stamping declared %r over label %r",
                prefix,
                name,
                declared,
                have,
            )
        da.attrs["units"] = declared
        return da
    if not isinstance(have, str) or len(have) > _MAX_UNIT_LEN:
        # Unverifiable for the same reasons as on the input path.
        if on_output == "require":
            raise ValueError(
                _require_message(prefix, name, declared, "an unverifiable unit label")
            )
        # Never echo `have`: it may be huge.
        _log.info(
            "%soutput %r: unverifiable unit label; stamped unchecked "
            "(declared %r, on_output=%r)",
            prefix,
            name,
            declared,
            on_output,
        )
        da.attrs["units"] = declared
        return da
    try:
        get_registry().Unit(have)
    except Exception:
        if on_output == "require":
            raise ValueError(
                _require_message(
                    prefix, name, declared, f"an unparseable unit label {have!r}"
                )
            ) from None
        _log.info(
            "%soutput %r: unparseable unit label %r; stamped unchecked "
            "(declared %r, on_output=%r)",
            prefix,
            name,
            have,
            declared,
            on_output,
        )
        da.attrs["units"] = declared
        return da

    # "strict" and "require" compare a parseable label alike.
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
                        f"{declared!r}, and on_output={on_output!r} requires them to "
                        f"match; if the body converts by scalar arithmetic, clear its "
                        f"'units' attribute or use on_output='stamp')"
                    ),
                ) from None
        else:
            raise ValueError(
                f"{prefix}output {name!r}: unit {have!r} differs from declared "
                f"{declared!r} and on_output={on_output!r} requires them to match"
            )
    da.attrs["units"] = declared
    return da


def require_dataarray(
    value: Any,
    declared: str | Unit,
    name: str,
    on_output: OnOutput | None = None,
    qualname: str | None = None,
) -> None:
    """Refuse a declared output that is not a `DataArray` under `"require"`.

    `apply_output_units` only sees `DataArray` values, so a wrapper returning a
    bare numpy array, `None`, or a container with the declared field missing
    would otherwise pass `"require"` unchecked.  Other modes let it through.

    Raises:
        ValueError: Under `"require"`, always.
    """
    if not resolve_enabled() or resolve_on_output(declared, on_output) != ("require"):
        return
    prefix = f"[{qualname}] " if qualname else ""
    carries = "no value" if value is None else f"a {type(value).__name__}"
    raise ValueError(
        _require_message(prefix, name, str(declared), f"{carries}, not a DataArray")
    )


def _require_message(prefix: str, name: str, declared: str, carries: str) -> str:
    return (
        f"{prefix}output {name!r} carries {carries} (declared {declared!r}), and "
        f"on_output='require' refuses to stamp a unit the values were not "
        f"computed in. Attach the unit the values are actually in where they "
        f"enter, e.g. `.assign_attrs(units=...).pint.quantify()`, and the declared "
        f"unit is then checked and converted to."
    )
