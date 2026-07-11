"""The `Freq` marker declaring a DataArray's temporal frequency, and the reader.

The temporal counterpart to ``schema._annotations``: where the schema markers
declare properties an array *carries* (its dims, coords, dtype), ``Freq`` declares
a property derived from the *values* of its datetime coordinate — the spacing (and
phase) of its time axis::

    Annotated[xr.DataArray, "degC", Freq("W-SUN")]

The declared string is kept **verbatim**: pandas silently defaults the anchor a user
did not spell (``"W"`` normalises to ``"W-SUN"``), so the raw spelling is the only
place the *anchoredness* of a declaration survives.  See ``_check`` for how it is
compared.

There is deliberately **no bare-string shorthand** — a string in the metadata is a
unit or a description, never a frequency.  The markers are immutable (``__slots__``)
and hashable so an annotation carrying them stays hashable; ``eq``/``repr`` cover all
fields, so ``eval(repr(m)) == m``.
"""

from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
)

from .._annotations import _is_dataarray_type, walk_signature
from ._config import OnMismatch, _validate_on_mismatch


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

    Pass ``anchored=True`` to opt in to pandas' default anchor anyway
    (``Freq("W", anchored=True)`` means ``W-SUN`` and means it), or ``anchored=False``
    to suppress the anchor comparison of a spelled-out anchor
    (``Freq("W-SUN", anchored=False)`` = "weekly, any weekday").

    Pass ``dim`` to name the time coordinate explicitly; by default the array's sole
    datetime-like coordinate is used, and an array carrying two is ambiguous.
    """

    _freq: str
    _dim: str | None
    _anchored: bool | None
    _on_mismatch: OnMismatch | None
    __slots__ = ("_freq", "_dim", "_anchored", "_on_mismatch")

    def __init__(
        self,
        freq: str,
        *,
        dim: str | None = None,
        anchored: bool | None = None,
        on_mismatch: OnMismatch | None = None,
    ) -> None:
        self._freq = freq
        self._dim = dim
        self._anchored = anchored
        self._on_mismatch = (
            None if on_mismatch is None else _validate_on_mismatch(on_mismatch)
        )

    @property
    def freq(self) -> str:
        """The declared frequency string, exactly as spelled."""
        return self._freq

    @property
    def dim(self) -> str | None:
        """The declared time dimension, or `None` to auto-detect it."""
        return self._dim

    @property
    def anchored(self) -> bool | None:
        """Whether the anchor is binding, or `None` to infer it from the spelling."""
        return self._anchored

    @property
    def on_mismatch(self) -> OnMismatch | None:
        """Per-marker severity override, or `None` to use the policy default."""
        return self._on_mismatch

    def __repr__(self) -> str:
        parts = [repr(self._freq)]
        if self._dim is not None:
            parts.append(f"dim={self._dim!r}")
        if self._anchored is not None:
            parts.append(f"anchored={self._anchored!r}")
        if self._on_mismatch is not None:
            parts.append(f"on_mismatch={self._on_mismatch!r}")
        return f"Freq({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Freq):
            return (self._freq, self._dim, self._anchored, self._on_mismatch) == (
                other._freq,
                other._dim,
                other._anchored,
                other._on_mismatch,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._freq, self._dim, self._anchored, self._on_mismatch))


def annotated_freq(hint: Any) -> Freq | None:
    """Return the `Freq` marker carried by an `Annotated` hint, or `None`.

    The first `Freq` marker in the metadata, and only when the annotated base type
    is a `DataArray` (incl. `DataArray | None`).  Non-`Annotated` hints, hints on
    non-DataArray types, and hints with no `Freq` marker return `None` (so
    `walk_signature`'s "no declaration" filter drops them).

    Args:
        hint: A type hint, typically from `get_type_hints(..., include_extras=True)`.

    Returns:
        The declared `Freq`, or `None`.
    """
    if get_origin(hint) is not Annotated:
        return None
    args = get_args(hint)
    if not _is_dataarray_type(args[0]):
        return None
    return next((m for m in args[1:] if isinstance(m, Freq)), None)


def freq_from_signature(
    func: object,
) -> tuple[dict[str, Freq], dict[str, Freq] | Freq | None]:
    """Extract declared frequencies from a function's type annotations.

    Args:
        func: A callable whose hints carry `Annotated[DataArray, Freq(...)]`.

    Returns:
        An `(inputs, output)` pair, where each declaration is that parameter's
        `Freq`; `output` is a per-field dict for a `TypedDict` or dataclass return,
        a single `Freq` for one declared `DataArray` return, or `None`.
    """
    return walk_signature(func, annotated_freq)
