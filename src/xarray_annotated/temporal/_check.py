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
means Sundays and will not accept a `W-WED` axis.

Two kinds of declaration have no pandas frequency to infer, so they are compared step
by step instead.  `"month"` and `"year"` have a month spacing and no phase, as they
stamp on any day.  `gaps_ok` allows steps that are whole multiples of the declared one,
provided the smallest equals it; without that last clause `Freq("D", gaps_ok=True)`
would accept every axis stamped at midnight.  `gaps_ok` is not compared between
declarations, since a gappy axis and a regular one can be the same axis.

Known limitation: `Day` is DST-aware in pandas while `24h` is not, so treating them as
the same spacing is an approximation for timezone-aware axes.
"""

from typing import Any

import numpy as np
import pandas as pd
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
from .._errors import ContractError
from ._annotations import _ANY_DAY, Freq

_NANOS_PER_DAY = 86_400_000_000_000


class FreqError(ContractError):
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

    ``gaps_ok`` needs a positive step to take multiples of, so it is refused on business
    days and custom offsets, which would otherwise pass unchecked.

    Args:
        marker: The declared `Freq`.
        context: A label for the error message (e.g. ``"f input 'x'"``).

    Raises:
        ValueError: If the declaration is malformed.
    """
    if marker.dim is not None and not marker.dim:
        raise ValueError(f"{context}: Freq dim must be a non-empty string.")
    if not isinstance(marker.gaps_ok, bool):
        raise ValueError(f"{context}: gaps_ok must be a bool, got {marker.gaps_ok!r}.")
    if not isinstance(marker.freq, str):
        raise ValueError(f"{context}: frequency must be a string, got {marker.freq!r}.")
    try:
        kind, spacing = _declared_spacing(marker.freq)
    except ValueError as exc:
        raise ValueError(
            f"{context}: invalid frequency {marker.freq!r}. {exc}"
        ) from exc
    if marker.gaps_ok and (kind == "opaque" or spacing <= 0):
        raise ValueError(
            f"{context}: gaps_ok needs a positive step to compare, and {marker.freq!r} "
            f"does not have one."
        )


def _any_day_months(freq: str) -> int | None:
    """The months per step of an any-day spelling such as `"3month"`, else `None`.

    Raises:
        ValueError: If the multiple is zero.
    """
    match = _ANY_DAY.fullmatch(freq)
    if match is None:
        return None
    count = int(match[1] or 1)
    if count == 0:
        raise ValueError("A calendar step must be at least one.")
    return count * (12 if match[2] == "year" else 1)


def _declared_spacing(freq: str) -> tuple[str, Any]:
    """The comparable spacing of a declared string, any-day spellings included."""
    months = _any_day_months(freq)
    return ("months", months) if months is not None else _spacing(to_offset(freq))


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
    raw string spells it out (`"W-WED"`).  An any-day spelling has no phase at all.
    """
    if _any_day_months(marker.freq) is not None:
        return None, None
    off = to_offset(marker.freq)
    return _convention(off), _anchor(off) if "-" in marker.freq else None


def freq_compatible(a: Freq, b: Freq) -> bool:
    """Return whether two `Freq` declarations can describe the same time axis.

    The marker-vs-marker counterpart to `check_freq` (no array in hand), for a static
    / build-time check of a producer/consumer edge.  Two declarations are compatible
    when their **spacing** is the same and their **phase** does not conflict — where
    "conflict" needs *both* sides to determine the thing being compared, so a
    declaration that does not spell an anchor never conflicts on one.

    `"month"` has no convention, so it conflicts with neither `"ME"` nor `"MS"`, and
    `gaps_ok` is not compared.  So this does not prove an edge safe: a producer
    declaring `gaps_ok` is compatible with a consumer that does not, and it is
    `check_freq` that refuses an axis with holes when the data arrives, as it does for
    `"W"` into `"W-SUN"`.

    Two declarations naming different `dim`s describe different axes, which one array
    can have, so they never conflict.

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
        >>> freq_compatible(Freq("month"), Freq("MS"))  # "month" stamps on any day
        True
        >>> freq_compatible(Freq("D", gaps_ok=True), Freq("D"))  # gaps_ok not compared
        True
        >>> freq_compatible(Freq("D", dim="a"), Freq("7D", dim="b"))  # two axes
        True
    """
    spacing_a, spacing_b = _declared_spacing(a.freq), _declared_spacing(b.freq)
    if a.dim is not None and b.dim is not None and a.dim != b.dim:
        return True
    if spacing_a != spacing_b:
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
    """The frequency of a time coordinate of 3+ points, or `None` if it has none.

    `xarray.infer_freq` (not pandas') so a `CFTimeIndex` — a 360-day or noleap
    calendar — works too.  It returns `None` on irregular spacing.
    """
    # Type comes from pd.infer_freq's stub, but xarray sets
    # future.infer_freq_returns_offset=False, so it is always str | None.
    return xr.infer_freq(coord)  # pyright: ignore[reportReturnType]


def _on_declared_weekday(marker: Freq, coord: xr.DataArray) -> bool:
    """Whether a weekly axis falls on the weekday its declaration spells, if any."""
    _, anchor = _phase(marker)
    if anchor is None or not isinstance(to_offset(marker.freq), Week):
        return True
    return coord.dt.dayofweek.values[0] == anchor


def _pair(times: pd.Index, step: int) -> str:
    """The two timestamps either side of a step, for an error message."""
    return f"{times[step]} then {times[step + 1]}"


def _steps_mismatch(
    steps: Any, spacing: Any, marker: Freq, times: pd.Index
) -> str | None:
    """How steps contradict the declared spacing, as `timedelta`s or month counts."""
    zero = spacing * 0
    if marker.gaps_ok:
        wrong = (steps <= zero) | (steps % spacing != zero)
    else:
        wrong = steps != spacing
    if wrong.any():
        return _pair(times, int(np.argmax(wrong)))
    if marker.gaps_ok and steps.min() != spacing:
        return "no step that short"
    return None


def _calendar_phase_mismatch(marker: Freq, coord: xr.DataArray) -> str | None:
    """How timestamps with whole-month steps sit off the declared calendar phase.

    Every step is a multiple of the declared one by now, so a quarter or year anchor
    needs comparing on the first timestamp only.
    """
    times = coord.to_index()
    time_of_day = (coord - coord.dt.floor("D")).values
    convention, anchor = _phase(marker)
    expected_day = coord.dt.days_in_month.values if convention == "end" else 1
    off = (time_of_day != time_of_day[0]) | (coord.dt.day.values != expected_day)
    if off.any():
        return _pair(times, max(int(np.argmax(off)) - 1, 0))
    if anchor is None:
        return None
    month = int(coord.dt.month.values[0])
    is_quarterly = isinstance(to_offset(marker.freq), (QuarterEnd, QuarterBegin))
    return None if (month % 3 if is_quarterly else month) == anchor else _pair(times, 0)


def _step_mismatch(marker: Freq, coord: xr.DataArray) -> str | None:
    """Compare an axis step by step, where `xarray.infer_freq` has no answer.

    That is an axis of two points, as it needs three; a `gaps_ok` axis or an any-day
    one, where it returns `None`.  Business days and custom offsets have no reduction
    to compare, so they pass, which leaves only two points, as `check_freq` refuses
    `gaps_ok` on them.
    """
    kind, spacing = _declared_spacing(marker.freq)
    if kind == "opaque":
        return None
    times = coord.to_index()
    if kind == "fixed":
        steps = pd.TimedeltaIndex(times[1:] - times[:-1])
        got = _steps_mismatch(steps, pd.Timedelta(spacing, "ns"), marker, times)
        if got is not None or _on_declared_weekday(marker, coord):
            return got
        return _pair(times, 0)
    months = 12 * coord.dt.year.values + coord.dt.month.values
    got = _steps_mismatch(np.diff(months), spacing, marker, times)
    if got is not None or _any_day_months(marker.freq) is not None:
        return got
    return _calendar_phase_mismatch(marker, coord)


def _mismatch(marker: Freq, coord: xr.DataArray) -> str | None:
    """How a time axis contradicts a declaration, or `None` if it does not.

    An axis of zero or one point has no spacing, so nothing can contradict it.
    """
    if coord.size < 2:
        return None
    if coord.size == 2 or marker.gaps_ok or _any_day_months(marker.freq) is not None:
        return _step_mismatch(marker, coord)
    inferred = _infer(coord)
    if inferred is None:
        return "irregular spacing"
    if not freq_compatible(marker, Freq(inferred)):
        return repr(inferred)
    # On a cftime axis `infer_freq` says `7D` where pandas says `W-WED`.
    return (
        None if _on_declared_weekday(marker, coord) else f"{inferred!r} on another day"
    )


def _check_valid_freq(
    da: xr.DataArray, declared: Freq, name: str, qualname: str | None
) -> None:
    """`check_freq` on a validated declaration, whatever `enabled` says."""
    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    dim, detail = _locate_axis(da, declared)
    if dim is None:
        raise FreqError(f"{prefix} {detail}")
    got = _mismatch(declared, da[dim])
    if got is not None:
        raise FreqError(
            f"{prefix} frequency mismatch on {dim!r}: expected {declared.freq!r}, "
            f"got {got}"
        )


def check_freq(
    da: xr.DataArray,
    declared: Freq,
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate `da`'s time axis against its declared frequency; return it unchanged.

    Locates the time coordinate, reading nothing from disk (see `Freq` for which
    coordinates count), and compares it with the declaration via `freq_compatible`,
    using `xarray.infer_freq`.  Because pandas always hands back a fully-anchored string
    (`"W-WED"`, never `"7D"`), the *declaration* alone decides how strict the comparison
    is — `Freq("7D")` accepts any weekday, `Freq("W-SUN")` accepts only Sundays.

    An array with no (or an ambiguous) time axis is a mismatch, and so is irregular
    spacing, unless the declaration allows it.  Two points are compared directly, as
    `infer_freq` needs three, and so are `gaps_ok` and any-day axes, which it has no
    frequency for.  An axis of zero or one point always passes.  When xarray-annotated
    is switched off (`enabled(False)`) this is a total no-op.

    Args:
        da: The DataArray to validate.
        declared: The `Freq` declared for `da`.
        name: The parameter/field name, for error messages.
        qualname: Optional qualified function name, prefixed to messages.

    Returns:
        `da`, unchanged (validation never mutates).

    Raises:
        FreqError: On a mismatch.
        ValueError: If the declaration is malformed, including `gaps_ok` on a frequency
            with no step to compare, or `XARRAY_ANNOTATED_ENABLED` holds an unrecognised
            value.

    Examples:
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from xarray_annotated.temporal import Freq, FreqError, check_freq
        >>> time = pd.to_datetime(["2025-01-01", "2025-01-02", "2026-01-01"])
        >>> da = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims="time")
        >>> check_freq(da, Freq("D", gaps_ok=True), "x") is da
        True
        >>> try:
        ...     check_freq(da, Freq("D"), "x")
        ... except FreqError as exc:
        ...     print(exc)
        'x' frequency mismatch on 'time': expected 'D', got irregular spacing
    """
    if not resolve_enabled():
        return da

    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    assert_valid_freq(declared, prefix)
    _check_valid_freq(da, declared, name, qualname)
    return da
