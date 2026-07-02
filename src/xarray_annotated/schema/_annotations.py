"""Typed markers declaring a DataArray's structural properties (stub).

The counterpart to ``units._annotations.Unit`` for the *structural* domain: the
properties every ``DataArray`` possesses independent of physical units — its
dimensions, coordinates, and dtype.  Used inside ``Annotated`` exactly as
``Unit`` is::

    Annotated[xr.DataArray, Dims("time", "lat", "lon")]
    Annotated[xr.DataArray, Coords("time"), Dtype("float64")]

These markers currently only *carry* their declaration; validation
(``schema._check.check_schema``) is not yet implemented.  They mirror ``Unit``'s
shape (immutable, ``__slots__``, hashable) so an annotation carrying them stays
hashable and the eventual reader can resolve them the same way ``annotated_unit``
resolves a ``Unit``.
"""


class Dims:
    """Declare the expected dimension names of a DataArray, in order.

    ``Dims("time", "lat", "lon")`` declares a 3-D array over those dims.
    """

    __slots__ = ("_names",)

    def __init__(self, *names: str) -> None:
        self._names = tuple(names)

    @property
    def names(self) -> tuple[str, ...]:
        """The declared dimension names, in order."""
        return self._names

    def __repr__(self) -> str:
        return f"Dims({', '.join(repr(n) for n in self._names)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dims):
            return self._names == other._names
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._names)


class Coords:
    """Declare the coordinate variables a DataArray must carry.

    ``Coords("time", "lat")`` declares that those coordinates must be present.
    """

    __slots__ = ("_names",)

    def __init__(self, *names: str) -> None:
        self._names = tuple(names)

    @property
    def names(self) -> tuple[str, ...]:
        """The declared coordinate names."""
        return self._names

    def __repr__(self) -> str:
        return f"Coords({', '.join(repr(n) for n in self._names)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coords):
            return self._names == other._names
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._names)


class Dtype:
    """Declare the expected dtype of a DataArray, e.g. ``Dtype("float64")``."""

    __slots__ = ("_dtype",)

    def __init__(self, dtype: str) -> None:
        self._dtype = dtype

    @property
    def dtype(self) -> str:
        """The declared dtype string."""
        return self._dtype

    def __repr__(self) -> str:
        return f"Dtype({self._dtype!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dtype):
            return self._dtype == other._dtype
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._dtype)
