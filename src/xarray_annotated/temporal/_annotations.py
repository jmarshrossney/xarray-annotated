"""The `Freq` marker declaring a DataArray's temporal frequency.

The temporal counterpart to ``schema._annotations``: where the schema markers
declare properties an array *carries* (its dims, coords, dtype), ``Freq`` declares
a property derived from the *values* of its datetime coordinate — the spacing (and
phase) of its time axis:

    Annotated[xr.DataArray, Unit("degC"), Freq("W-SUN")]

The declared string is kept **verbatim**: pandas silently defaults the anchor a user
did not spell (``"W"`` normalises to ``"W-SUN"``), so the raw spelling is the only
place the *anchoredness* of a declaration survives.  See ``_check`` for how it is
compared.

The markers are immutable (``__slots__``)
and hashable so an annotation carrying them stays hashable; ``eq``/``repr`` cover all
fields, so ``eval(repr(m)) == m``.
"""

import re

_ANY_DAY = re.compile(r"([0-9]*)(month|year)")


class Freq:
    """Declare the expected temporal frequency of a DataArray's time axis.

    `Freq("7D")` declares an array whose time axis advances in seven-day steps, and
    `Freq("W-SUN")` declares the same spacing and pins the phase to Sundays.

    **The time axis** is the array's one datetime coordinate with a pandas index, which
    a dimension coordinate has by default, so the check reads nothing from disk.  If
    there is no such coordinate, or there are two, the check raises `FreqError`, and
    names the fix where it can tell; pass `dim` to choose between two.  A scalar, 2-D or
    unindexed coordinate is not a time axis, so `time(obs)` needs `.set_xindex("time")`,
    and alongside a real one it is left unchecked.

    **Spacing** is always compared, by meaning rather than by string.  A regular axis
    has its frequency inferred with `xarray.infer_freq`, so `cftime` calendars work too.

    | Declared | Actual | Result | Why |
    |---|---|---|---|
    | `Freq("7D")` | `W-WED` | pass | Seven days either way |
    | `Freq("D")` | `24h` | pass | Same fixed spacing |
    | `Freq("QE")` | `3ME` | pass | A quarter is three months |
    | `Freq("ME")` | `30D` | fail | Calendar months are not fixed-length days |

    **Phase** is compared in two parts.  The end or begin convention (`"ME"` vs `"MS"`)
    is always compared, and the anchor only if you spell it: `Freq("W")` means weekly on
    any weekday, and `Freq("W-SUN")` weekly on Sundays only.  Unlike pandas, an
    unspelled anchor means "any" rather than a default.  A quarterly anchor names four
    dates rather than one month, so `"QE-MAR"` and `"QE-DEC"` are the same.

    **Every axis is tested**, whatever its length.  Irregular spacing raises, such as a
    missing or duplicated timestamp, unless `gaps_ok` allows it.  Two timestamps are
    compared directly, though business-day and custom frequencies have nothing to
    compare, so pass.  Zero or one timestamp always passes.

    **A malformed declaration** raises `ValueError` when `check_annotations` decorates
    the function, not at `Freq(...)`.  That covers a string pandas cannot parse,
    including the legacy aliases `"M"` and `"H"`, now spelled `"ME"` and `"h"`; a zero
    calendar step such as `"0month"`; an empty `dim`; and `gaps_ok` on a business-day,
    custom or negative frequency, which has no positive step to compare.

    Args:
        freq: A pandas frequency, or `"month"`, `"year"` or a multiple such as
            `"3month"`, for one value per calendar step, stamped on any day, such as
            monthly means stamped at midpoints.
        dim: The time coordinate, where an array has two.
        gaps_ok: Declare the sampling period without a gap-free axis, so
            `Freq("D", gaps_ok=True)` accepts January 2025 joined to January 2026.
            Timestamps must still increase, every step must be a whole multiple of the
            declared one, and the smallest step must equal it, so weekly values do not
            pass as daily ones.
    """

    _freq: str
    _dim: str | None
    _gaps_ok: bool
    __slots__ = ("_dim", "_freq", "_gaps_ok")

    def __init__(
        self, freq: str, *, dim: str | None = None, gaps_ok: bool = False
    ) -> None:
        self._freq = freq
        self._dim = dim
        self._gaps_ok = gaps_ok

    @property
    def freq(self) -> str:
        """The declared frequency string, exactly as spelled."""
        return self._freq

    @property
    def dim(self) -> str | None:
        """The declared time dimension, or `None` to auto-detect it."""
        return self._dim

    @property
    def gaps_ok(self) -> bool:
        """Whether the time axis may have holes."""
        return self._gaps_ok

    @property
    def any_day(self) -> bool:
        """Whether this is a calendar step on any day, such as `"month"`.

        Such a declaration is not a pandas frequency, so a tool that needs one, to
        resample say, cannot use it.  This reads the spelling alone, and validating the
        declaration is separate, so `Freq("0month").any_day` is `True`.
        """
        return (
            isinstance(self._freq, str) and _ANY_DAY.fullmatch(self._freq) is not None
        )

    def __repr__(self) -> str:
        parts = [repr(self._freq)]
        if self._dim is not None:
            parts.append(f"dim={self._dim!r}")
        if self._gaps_ok:
            parts.append(f"gaps_ok={self._gaps_ok!r}")
        return f"Freq({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Freq):
            return (self._freq, self._dim, self._gaps_ok) == (
                other._freq,
                other._dim,
                other._gaps_ok,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._freq, self._dim, self._gaps_ok))
