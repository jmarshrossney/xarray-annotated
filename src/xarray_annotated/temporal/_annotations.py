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


class Freq:
    """Declare the expected temporal frequency of a DataArray's time axis.

    ``Freq("7D")`` declares an array whose datetime coordinate advances in
    seven-day steps; ``Freq("W-SUN")`` declares the same spacing *and* pins the
    phase to Sundays.  Two independent comparisons follow from the declaration:

    * **spacing** — always compared (``"7D"`` and ``"W-WED"`` have the same spacing);
    * **phase** — the ``End``/``Begin`` convention (``"ME"`` vs ``"MS"``), always
      compared; and the anchor (``"-WED"``, ``"-MAR"``), compared only where the
      declaration *spells* it.  ``Freq("W")`` therefore means "weekly, any weekday",
      a deliberate divergence from pandas (which would default it to ``W-SUN``).

    Pass ``dim`` to name the time coordinate where an array has two; see
    ``docs/guides/declaring.md`` for which coordinates count.
    """

    _freq: str
    _dim: str | None
    __slots__ = ("_dim", "_freq")

    def __init__(self, freq: str, *, dim: str | None = None) -> None:
        self._freq = freq
        self._dim = dim

    @property
    def freq(self) -> str:
        """The declared frequency string, exactly as spelled."""
        return self._freq

    @property
    def dim(self) -> str | None:
        """The declared time dimension, or `None` to auto-detect it."""
        return self._dim

    def __repr__(self) -> str:
        parts = [repr(self._freq)]
        if self._dim is not None:
            parts.append(f"dim={self._dim!r}")
        return f"Freq({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Freq):
            return (self._freq, self._dim) == (other._freq, other._dim)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._freq, self._dim))
