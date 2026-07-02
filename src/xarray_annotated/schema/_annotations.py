"""Typed markers declaring a DataArray's structural properties, and the reader.

The structural counterpart to ``units._annotations``: the properties every
``DataArray`` possesses regardless of physical units — its dimensions
(``Dims``), coordinates (``Coords``), and dtype (``Dtype``).  Used inside
``Annotated`` exactly as ``Unit`` is, and — unlike units — **several at once**::

    Annotated[xr.DataArray, Dims("time", "lat", "lon")]
    Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")]

Each marker carries the *strictness* of its own check (dims order, exact dtype)
and an optional per-marker ``on_mismatch`` severity override, because the three
properties have different natural defaults and a wrong dtype is often less severe
than a wrong set of dims.  The markers are immutable (``__slots__``) and hashable
so an annotation carrying them stays hashable/cacheable; ``eq``/``repr`` cover all
fields, so ``eval(repr(m)) == m``.

There is deliberately **no bare-string shorthand** (a string in the metadata is a
description, never a declaration): only these typed markers are read.
"""

from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
)

from .._annotations import _is_dataarray_type, walk_signature
from ._config import OnMismatch, _validate_on_mismatch


def _check_on_mismatch(value: OnMismatch | None) -> OnMismatch | None:
    """Validate a marker's optional ``on_mismatch`` override at construction."""
    return None if value is None else _validate_on_mismatch(value)


class Dims:
    """Declare the expected dimension names of a DataArray.

    ``Dims("time", "lat", "lon")`` declares an array over exactly those dims.
    By default the *set* of dims must match (extra or missing dims fail) but their
    order is free — xarray operations are order-independent until you drop to
    numpy.  Pass ``ordered=True`` to also pin the order (e.g. before ``.values``,
    ``.stack`` or ``apply_ufunc``).
    """

    _names: tuple[str, ...]
    _ordered: bool
    _on_mismatch: OnMismatch | None
    __slots__ = ("_names", "_ordered", "_on_mismatch")

    def __init__(
        self,
        *names: str,
        ordered: bool = False,
        on_mismatch: OnMismatch | None = None,
    ) -> None:
        self._names = tuple(names)
        self._ordered = ordered
        self._on_mismatch = _check_on_mismatch(on_mismatch)

    @property
    def names(self) -> tuple[str, ...]:
        """The declared dimension names, in the order given."""
        return self._names

    @property
    def ordered(self) -> bool:
        """Whether dim order (not just the set) must match."""
        return self._ordered

    @property
    def on_mismatch(self) -> OnMismatch | None:
        """Per-marker severity override, or `None` to use the policy default."""
        return self._on_mismatch

    def __repr__(self) -> str:
        parts = [repr(n) for n in self._names]
        if self._ordered:
            parts.append("ordered=True")
        if self._on_mismatch is not None:
            parts.append(f"on_mismatch={self._on_mismatch!r}")
        return f"Dims({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dims):
            return (self._names, self._ordered, self._on_mismatch) == (
                other._names,
                other._ordered,
                other._on_mismatch,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._names, self._ordered, self._on_mismatch))


class Coords:
    """Declare coordinate variables a DataArray must carry.

    ``Coords("time", "lat")`` declares that those coordinates are present (as
    *labels*, not merely dims — a dim can exist without coordinate values).
    Extra coordinates are allowed; only the declared ones must be present.
    """

    _names: tuple[str, ...]
    _on_mismatch: OnMismatch | None
    __slots__ = ("_names", "_on_mismatch")

    def __init__(
        self,
        *names: str,
        on_mismatch: OnMismatch | None = None,
    ) -> None:
        self._names = tuple(names)
        self._on_mismatch = _check_on_mismatch(on_mismatch)

    @property
    def names(self) -> tuple[str, ...]:
        """The declared coordinate names."""
        return self._names

    @property
    def on_mismatch(self) -> OnMismatch | None:
        """Per-marker severity override, or `None` to use the policy default."""
        return self._on_mismatch

    def __repr__(self) -> str:
        parts = [repr(n) for n in self._names]
        if self._on_mismatch is not None:
            parts.append(f"on_mismatch={self._on_mismatch!r}")
        return f"Coords({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coords):
            return (self._names, self._on_mismatch) == (
                other._names,
                other._on_mismatch,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._names, self._on_mismatch))


class Dtype:
    """Declare the expected dtype of a DataArray, e.g. ``Dtype("float64")``.

    By default the check is by *kind*: any float satisfies ``Dtype("float64")``,
    any integer satisfies ``Dtype("int32")`` — enough to catch an int/float or
    bool/float mix-up without firing on ``f8`` vs ``f4``.  Pass ``exact=True`` to
    require the precise dtype (e.g. to pin memory footprint or a typed sink).
    """

    _dtype: str
    _exact: bool
    _on_mismatch: OnMismatch | None
    __slots__ = ("_dtype", "_exact", "_on_mismatch")

    def __init__(
        self,
        dtype: str,
        *,
        exact: bool = False,
        on_mismatch: OnMismatch | None = None,
    ) -> None:
        self._dtype = dtype
        self._exact = exact
        self._on_mismatch = _check_on_mismatch(on_mismatch)

    @property
    def dtype(self) -> str:
        """The declared dtype string."""
        return self._dtype

    @property
    def exact(self) -> bool:
        """Whether the exact dtype (not just its kind) must match."""
        return self._exact

    @property
    def on_mismatch(self) -> OnMismatch | None:
        """Per-marker severity override, or `None` to use the policy default."""
        return self._on_mismatch

    def __repr__(self) -> str:
        parts = [repr(self._dtype)]
        if self._exact:
            parts.append("exact=True")
        if self._on_mismatch is not None:
            parts.append(f"on_mismatch={self._on_mismatch!r}")
        return f"Dtype({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dtype):
            return (self._dtype, self._exact, self._on_mismatch) == (
                other._dtype,
                other._exact,
                other._on_mismatch,
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._dtype, self._exact, self._on_mismatch))


#: The marker types the reader recognises in `Annotated` metadata.
SchemaMarker = Dims | Coords | Dtype
_SCHEMA_MARKERS = (Dims, Coords, Dtype)


def annotated_schema(hint: Any) -> list[SchemaMarker] | None:
    """Return every schema marker carried by an `Annotated` hint, or `None`.

    Collects *all* `Dims` / `Coords` / `Dtype` markers in the metadata (a hint may
    declare several structural properties at once), but only when the annotated
    base type is a `DataArray` (incl. `DataArray | None`).  Non-`Annotated` hints,
    hints on non-DataArray types, and hints with no schema marker return `None`
    (so `walk_signature`'s "no declaration" filter drops them).

    Args:
        hint: A type hint, typically from `get_type_hints(..., include_extras=True)`.

    Returns:
        A non-empty list of markers in annotation order, or `None`.
    """
    if get_origin(hint) is not Annotated:
        return None
    args = get_args(hint)
    if not _is_dataarray_type(args[0]):
        return None
    markers = [m for m in args[1:] if isinstance(m, _SCHEMA_MARKERS)]
    return markers or None


def schema_from_signature(
    func: object,
) -> tuple[
    dict[str, list[SchemaMarker]],
    dict[str, list[SchemaMarker]] | list[SchemaMarker] | None,
]:
    """Extract declared schema markers from a function's type annotations.

    Args:
        func: A callable whose hints carry `Annotated[DataArray, <markers>]`.

    Returns:
        An `(inputs, output)` pair, where each declaration is the list of markers
        on that parameter/field; `output` is a per-field dict for a `TypedDict` or
        dataclass return, a single list for one declared `DataArray` return, or
        `None`.
    """
    return walk_signature(func, annotated_schema)
