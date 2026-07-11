"""Comparing declared frequencies, and validating a DataArray's time axis.

Two legs, sharing one model of what a frequency *is*:

* `freq_compatible` — marker vs marker, with no array in hand (the counterpart to
  `schema.dims_compatible`), for a build-time check of a producer/consumer edge.
* `check_freq` — marker vs DataArray, the runtime leg.

**The model.** A declaration is compared along two independent keys:

* **spacing** — always compared.  Fixed-length offsets (ticks, `D`, and `W`, which is
  seven fixed days) reduce to a nanosecond count, so `"7D"` and `"W-WED"` have the
  *same* spacing; calendar offsets reduce to a month count, so `"QE"` and `"3ME"` do
  too.  The two families never compare equal (`"ME"` is not `"30D"`).  Anything else
  (business days, custom offsets) is opaque and compares by its exact string.
* **phase** — the `End`/`Begin` convention (`"ME"` vs `"MS"`), always compared; and the
  anchor (`"-WED"`, `"-MAR"`), compared only where **both** declarations determine it.

That last clause is the whole point.  pandas silently defaults an anchor the user did
not spell (`to_offset("W").freqstr == "W-SUN"`), so anchoredness is read off the *raw
declared string* — a `"-"` in it — never off the normalised offset.  `Freq("W")` thus
means "weekly, any weekday" (a deliberate divergence from pandas), while `Freq("W-SUN")`
means Sundays and will not accept a `W-WED` axis.  A marker's `anchored=` keyword
overrides that inference either way.

Known limitation: `Day` is DST-aware in pandas while `24h` is not, so treating them as
the same spacing is an approximation for timezone-aware axes.
"""

import warnings

import numpy as np
import xarray as xr
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import (
    BaseOffset,
    Day,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    Tick,
    Week,
    YearBegin,
    YearEnd,
)

from ._annotations import Freq
from ._config import (
    OnMismatch,
    OnUninferable,
    _validate_on_mismatch,
    _validate_on_uninferable,
    get_policy,
)

_NANOS_PER_DAY = 86_400_000_000_000


class FreqWarning(UserWarning):
    """Warning issued for a frequency mismatch or an uninferable time axis."""


class FreqError(Exception):
    """Raised for a frequency mismatch under ``on_mismatch="error"``.

    Deliberately **not** a ``ValueError``: a mismatch (the *data* fails the
    declaration) is a distinct event from a malformed *declaration* (an unparseable
    offset string), which raises ``ValueError`` from ``assert_valid_freq``.  Keeping
    them on separate hierarchies means catching one never silently swallows the other.
    """


def assert_valid_freq(marker: Freq, context: str) -> None:
    """Validate a `Freq` *declaration* itself, independent of any DataArray.

    Used at decoration time to fail fast on a malformed declaration (an unparseable
    offset string, an empty dim name) rather than only when the decorated function
    first runs.  pandas' own message is kept — it is the one that explains that the
    legacy aliases ``"M"`` and ``"H"`` are now spelled ``"ME"`` and ``"h"``.

    Args:
        marker: The declared `Freq`.
        context: A label for the error message (e.g. ``"f input 'x'"``).

    Raises:
        ValueError: If the declaration is malformed.
    """
    if marker.dim is not None and not marker.dim:
        raise ValueError(f"{context}: Freq dim must be a non-empty string.")
    try:
        to_offset(marker.freq)
    except ValueError as exc:
        raise ValueError(
            f"{context}: invalid frequency {marker.freq!r}. {exc}"
        ) from exc


def _spacing(off: BaseOffset) -> tuple[str, object]:
    """The comparable spacing of an offset: fixed nanoseconds, months, or opaque."""
    if isinstance(off, (Tick, Day)):  # `Day` is not a `Tick` in pandas 3
        return ("fixed", off.nanos)
    if isinstance(off, Week):
        return ("fixed", off.n * 7 * _NANOS_PER_DAY)  # what makes `7D` ≡ `W-*`
    if isinstance(off, (MonthEnd, MonthBegin)):
        return ("months", off.n)
    if isinstance(off, (QuarterEnd, QuarterBegin)):
        return ("months", 3 * off.n)
    if isinstance(off, (YearEnd, YearBegin)):
        return ("months", 12 * off.n)
    # Business days, custom offsets: no reduction we can trust, so compare exactly.
    return ("opaque", off.freqstr)


def _convention(off: BaseOffset) -> str | None:
    """The `"end"` / `"begin"` convention of a calendar offset, else `None`."""
    if isinstance(off, (MonthEnd, QuarterEnd, YearEnd)):
        return "end"
    if isinstance(off, (MonthBegin, QuarterBegin, YearBegin)):
        return "begin"
    return None


def _anchor(off: BaseOffset) -> int | None:
    """The anchor pandas assigned to an offset (weekday / month), else `None`.

    Note this is pandas' *normalised* anchor, which it fills in even where the user
    spelled none — `_phase` decides whether it is binding.  (The `type: ignore`s are
    because these attributes live on the Cython offset classes but not in the stubs.)
    """
    if isinstance(off, Week):
        return off.weekday  # type: ignore[attr-defined]
    if isinstance(off, (QuarterEnd, QuarterBegin)):
        return off.startingMonth  # type: ignore[attr-defined]
    if isinstance(off, (YearEnd, YearBegin)):
        return off.month
    return None


def _phase(marker: Freq) -> tuple[str | None, int | None]:
    """The comparable phase of a declaration: `(convention, anchor-or-None)`.

    The anchor is `None` — meaning *undetermined*, so it is not compared — unless the
    declaration determines it: either the raw string spells it out (`"W-WED"`) or the
    marker forces the question with `anchored=`.
    """
    off = to_offset(marker.freq)
    anchored = marker.anchored
    if anchored is None:
        anchored = "-" in marker.freq
    return _convention(off), _anchor(off) if anchored else None


def freq_compatible(a: Freq, b: Freq) -> bool:
    """Return whether two `Freq` declarations can describe the same time axis.

    The marker-vs-marker counterpart to `check_freq` (no array in hand), for a static
    / build-time check of a producer/consumer edge.  Two declarations are compatible
    when their **spacing** is the same and their **phase** does not conflict — where
    "conflict" needs *both* sides to determine the thing being compared, so a
    declaration that does not spell an anchor never conflicts on one.

    Args:
        a: A `Freq` marker.
        b: Another `Freq` marker.

    Returns:
        `True` unless the two are provably inconsistent.

    Raises:
        ValueError: If either frequency string is unparseable.

    Examples:
        >>> from xarray_annotated.temporal import Freq, freq_compatible
        >>> freq_compatible(Freq("7D"), Freq("W-WED"))  # same spacing, no anchor on 7D
        True
        >>> freq_compatible(Freq("W-SUN"), Freq("W-WED"))  # the resample-phase footgun
        False
        >>> freq_compatible(Freq("W"), Freq("W-WED"))  # "W" spells no anchor
        True
        >>> freq_compatible(Freq("QE"), Freq("3ME"))  # same spacing and convention
        True
        >>> freq_compatible(Freq("ME"), Freq("MS"))  # month-end is not month-start
        False
    """
    if _spacing(to_offset(a.freq)) != _spacing(to_offset(b.freq)):
        return False
    (conv_a, anchor_a), (conv_b, anchor_b) = _phase(a), _phase(b)
    if conv_a is not None and conv_b is not None and conv_a != conv_b:
        return False
    return not (anchor_a is not None and anchor_b is not None and anchor_a != anchor_b)


def _is_datetime_like(da: xr.DataArray, name: str) -> bool:
    """Whether a coordinate carries timestamps — numpy datetimes or cftime ones."""
    if np.issubdtype(da.coords[name].dtype, np.datetime64):
        return True
    return isinstance(da.indexes.get(name), xr.CFTimeIndex)


def _locate_axis(da: xr.DataArray, marker: Freq) -> tuple[str | None, str]:
    """Find the time coordinate a `Freq` declaration is about.

    Returns `(dim, "")` on success, or `(None, detail)` when the array offers no
    unambiguous answer — which is itself a mismatch (the declaration says the array
    has a time axis; it does not, or it has two).
    """
    candidates = [str(name) for name in da.coords if _is_datetime_like(da, str(name))]
    if marker.dim is not None:
        if marker.dim not in candidates:
            return None, (
                f"declared {marker!r} but {marker.dim!r} is not a datetime "
                f"coordinate; datetime coords: {candidates}"
            )
        return marker.dim, ""
    if not candidates:
        return None, f"declared {marker!r} but the array has no datetime coordinate"
    if len(candidates) > 1:
        return None, (
            f"declared {marker!r} but the time axis is ambiguous: datetime coords "
            f"{candidates}; name one with Freq(..., dim=...)"
        )
    return candidates[0], ""


def _infer(coord: xr.DataArray) -> str | None:
    """The frequency of a time coordinate, or `None` if it has none.

    `xarray.infer_freq` (not pandas') so a `CFTimeIndex` — a 360-day or noleap
    calendar — works too.  It *raises* on fewer than three points and *returns* `None`
    on irregular spacing; both mean the same thing here: uninferable.
    """
    try:
        return xr.infer_freq(coord)
    except ValueError:
        return None


def _report(ok: bool, severity: OnMismatch | OnUninferable, message: str) -> None:
    """Act on a check result under the resolved severity."""
    if ok or severity == "ignore":
        return
    if severity == "error":
        raise FreqError(message)
    warnings.warn(message, FreqWarning, stacklevel=3)


def check_freq(
    da: xr.DataArray,
    declared: Freq | list[Freq],
    name: str,
    on_mismatch: OnMismatch | None = None,
    on_uninferable: OnUninferable | None = None,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate `da`'s time axis against its declared frequency; return it unchanged.

    Locates the declared time coordinate (the array's sole datetime-like coordinate,
    unless the marker names one), infers its frequency with `xarray.infer_freq`, and
    compares that with the declaration via `freq_compatible`.  Because pandas always
    hands back a fully-anchored string (`"W-WED"`, never `"7D"`), the *declaration*
    alone decides how strict the comparison is — `Freq("7D")` accepts any weekday,
    `Freq("W-SUN")` accepts only Sundays.

    Two events, two axes.  A frequency that contradicts the declaration is a
    **mismatch**, as is an array with no (or an ambiguous) time axis.  An axis whose
    frequency cannot be inferred at all — fewer than three points, or irregular
    spacing — is **uninferable**: the declaration was not violated, it was never
    tested, and by default that warns rather than raises.

    Severity resolves per marker as: the marker's own `on_mismatch` override, else the
    `on_mismatch` argument, else the policy default.  When the master switch is off
    (`enabled=False`) this is a total no-op.

    Args:
        da: The DataArray to validate.
        declared: A single `Freq` or a list of them declared for `da`.
        name: The parameter/field name, for error messages.
        on_mismatch: Per-call default severity for a mismatch, overriding the policy
            default but overridden by a marker's own `on_mismatch`.
        on_uninferable: Per-call severity for an uninferable axis, overriding the
            policy default.
        qualname: Optional qualified function name, prefixed to messages.

    Returns:
        `da`, unchanged (validation never mutates).

    Raises:
        FreqError: On a mismatch when the effective severity is ``"error"``.
    """
    pol = get_policy()
    if not pol.enabled:
        return da
    if on_mismatch is not None:
        on_mismatch = _validate_on_mismatch(on_mismatch)
    if on_uninferable is not None:
        on_uninferable = _validate_on_uninferable(on_uninferable)

    markers = [declared] if isinstance(declared, Freq) else declared
    uninferable = on_uninferable or pol.on_uninferable
    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    for marker in markers:
        severity = marker.on_mismatch or on_mismatch or pol.on_mismatch
        dim, detail = _locate_axis(da, marker)
        if dim is None:
            _report(False, severity, f"{prefix} {detail}")
            continue
        inferred = _infer(da[dim])
        if inferred is None:
            _report(
                False,
                uninferable,
                f"{prefix} frequency of {dim!r} is uninferable (fewer than three "
                f"points, or irregular spacing), so {marker!r} was not checked",
            )
            continue
        ok = freq_compatible(marker, Freq(inferred))
        _report(
            ok,
            severity,
            f"{prefix} frequency mismatch on {dim!r}: expected {marker.freq!r}, "
            f"got {inferred!r}",
        )
    return da
