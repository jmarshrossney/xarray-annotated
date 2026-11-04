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

import logging
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
from xarray.indexes import PandasIndex

from .._config import resolve_enabled
from ._annotations import Freq
from ._config import (
    DEFAULT_ON_UNINFERABLE,
    OnUninferable,
    _validate_on_uninferable,
)

_log = logging.getLogger("xarray_annotated.temporal")

_NANOS_PER_DAY = 86_400_000_000_000


class FreqWarning(UserWarning):
    """Base class for frequency warnings."""


class UninferableFreqWarning(FreqWarning):
    """Warning issued when a time axis is too short or irregular to check.

    A warnings filter can silence short test fixtures:

        filterwarnings = ["ignore::xarray_annotated.temporal.UninferableFreqWarning"]
    """


class FreqError(Exception):
    """Raised for a frequency mismatch.

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
        # `QE-MAR` and `QE-DEC` are the same four dates, and `infer_freq` only ever
        # names one month of the three, even for `2QE`, which is why `2QE-MAR` and
        # `2QE-JUN` cannot be told apart here.
        return off.startingMonth % 3  # type: ignore[attr-defined]
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
    """Whether a coordinate is an indexed time axis, numpy datetimes or cftime ones.

    Only a `PandasIndex` qualifies, as it is in memory where a custom index may leave
    its coordinate lazy.  `da.indexes` is avoided: it casts every index to pandas, and
    raises on a custom one.
    """
    index = da.xindexes.get(name)
    if not isinstance(index, PandasIndex):
        return False
    is_numpy = np.issubdtype(da.coords[name].dtype, np.datetime64)
    return is_numpy or isinstance(index.index, xr.CFTimeIndex)


def _why_not_axis(da: xr.DataArray, name: str) -> str | None:
    """Why a numpy datetime coordinate is not a time axis, and how to make it one.

    Decided from its dtype, shape and index, so nothing is read.  That leaves cftime
    unidentified, as an unindexed one has dtype `object` until it is read.

    Returns:
        The reason, or `None` for a coordinate this cannot say anything about.
    """
    coord = da.coords[name]
    index = da.xindexes.get(name)
    if not np.issubdtype(coord.dtype, np.datetime64) or isinstance(index, PandasIndex):
        return None
    if coord.ndim == 0:
        return (
            f"{name!r} is a scalar, so keep its dimension with .expand_dims({name!r})"
        )
    if coord.ndim > 1:
        return f"{name!r} is {coord.ndim}-D, and a time axis must be 1-D"
    if index is not None:
        return (
            f"{name!r} has a {type(index).__name__}, not a pandas index, so replace it "
            f"with .drop_indexes({name!r}).set_xindex({name!r})"
        )
    return f"{name!r} has no index, so add one with .set_xindex({name!r})"


def _locate_axis(da: xr.DataArray, marker: Freq) -> tuple[str | None, str]:
    """Find the time coordinate a `Freq` declaration is about.

    Returns `(dim, "")` on success, or `(None, detail)` when the array offers no
    unambiguous answer — which is itself a mismatch (the declaration says the array
    has a time axis; it does not, or it has two).
    """
    names = [str(name) for name in da.coords]
    candidates = [name for name in names if _is_datetime_like(da, name)]
    if marker.dim is not None:
        if marker.dim not in candidates:
            reason = _why_not_axis(da, marker.dim) if marker.dim in names else None
            return None, (
                f"declared {marker!r} but {marker.dim!r} is not an indexed datetime "
                f"coordinate: {reason or f'indexed datetime coords: {candidates}'}"
            )
        return marker.dim, ""
    if not candidates:
        reasons = [r for name in names if (r := _why_not_axis(da, name))]
        return None, (
            f"declared {marker!r} but the array has no indexed datetime coordinate"
            + (
                f": {'; '.join(reasons)}"
                if reasons
                else "; unindexed cftime and undecoded times look like this too"
            )
        )
    if len(candidates) > 1:
        return None, (
            f"declared {marker!r} but the time axis is ambiguous: indexed datetime "
            f"coords {candidates}; name one with Freq(..., dim=...)"
        )
    return candidates[0], ""


def _infer(coord: xr.DataArray) -> str | None:
    """The frequency of a time coordinate, or `None` if it has none.

    `xarray.infer_freq` (not pandas') so a `CFTimeIndex` — a 360-day or noleap
    calendar — works too.  It *raises* on fewer than three points and *returns* `None`
    on irregular spacing; both mean the same thing here: uninferable.
    """
    try:
        # Type comes from pd.infer_freq's stub, but xarray sets
        # future.infer_freq_returns_offset=False, so it is always str | None.
        return xr.infer_freq(coord)  # pyright: ignore[reportReturnType]
    except ValueError:
        return None


def _report_uninferable(severity: OnUninferable, message: str) -> None:
    """Act on an uninferable axis under the resolved severity."""
    if severity == "ignore":
        _log.info("%s (ignored)", message)
        return
    if severity == "error":
        raise FreqError(message)
    # stacklevel=3 targets a direct check_freq call; via enforce it
    # lands on the wrapper line, still identified by the [qualname] prefix.
    warnings.warn(message, UninferableFreqWarning, stacklevel=3)


def check_freq(
    da: xr.DataArray,
    declared: Freq | list[Freq],
    name: str,
    on_uninferable: OnUninferable | None = None,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate `da`'s time axis against its declared frequency; return it unchanged.

    Locates the time coordinate, reading nothing from disk (see
    `docs/guides/declaring.md` for which coordinates count), infers its frequency with
    `xarray.infer_freq`, and
    compares that with the declaration via `freq_compatible`.  Because pandas always
    hands back a fully-anchored string (`"W-WED"`, never `"7D"`), the *declaration*
    alone decides how strict the comparison is — `Freq("7D")` accepts any weekday,
    `Freq("W-SUN")` accepts only Sundays.

    A frequency that contradicts the declaration is a **mismatch**, as is an array
    with no (or an ambiguous) time axis, and both raise.  An axis whose
    frequency cannot be inferred at all — fewer than three points, or irregular
    spacing — is **uninferable**: the declaration was not violated, it was never
    tested, and by default that warns rather than raises.

    `on_uninferable` defaults to `"warn"`, and an event passed under `"ignore"` is
    logged at INFO to `xarray_annotated.temporal`.  When xarray-annotated is switched
    off (`enabled(False)`) this is a total no-op.

    Args:
        da: The DataArray to validate.
        declared: A single `Freq` or a list of them declared for `da`.
        name: The parameter/field name, for error messages.
        on_uninferable: Per-call severity for an uninferable axis, overriding the
            default.
        qualname: Optional qualified function name, prefixed to messages.

    Returns:
        `da`, unchanged (validation never mutates).

    Raises:
        FreqError: On a mismatch, or an uninferable axis under ``"error"``.
    """
    if not resolve_enabled():
        return da
    if on_uninferable is not None:
        on_uninferable = _validate_on_uninferable(on_uninferable)

    markers = [declared] if isinstance(declared, Freq) else declared
    uninferable = on_uninferable or DEFAULT_ON_UNINFERABLE
    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    for marker in markers:
        dim, detail = _locate_axis(da, marker)
        if dim is None:
            raise FreqError(f"{prefix} {detail}")
        inferred = _infer(da[dim])
        if inferred is None:
            _report_uninferable(
                uninferable,
                f"{prefix} frequency of {dim!r} is uninferable (fewer than three "
                f"points, or irregular spacing), so {marker!r} was not checked",
            )
            continue
        if not freq_compatible(marker, Freq(inferred)):
            raise FreqError(
                f"{prefix} frequency mismatch on {dim!r}: expected {marker.freq!r}, "
                f"got {inferred!r}"
            )
    return da
