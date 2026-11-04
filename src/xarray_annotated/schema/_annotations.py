"""Typed markers declaring a DataArray's structural properties.

The structural counterpart to ``units._annotations``: the properties every
``DataArray`` possesses regardless of physical units — its dimensions
(``Dims``), coordinates (``Coords``), and dtype (``Dtype``).  Used inside
``Annotated`` exactly as ``Unit`` is, and — unlike units — **several at once**:

    Annotated[xr.DataArray, Dims("time", "lat", "lon")]
    Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")]

Each marker carries the *strictness* of its own check (dims order, exact dtype).
The markers are immutable (``__slots__``) and hashable so an annotation carrying
them stays hashable/cacheable; ``eq``/``repr`` cover all fields, so
``eval(repr(m)) == m``.
"""


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
    __slots__ = ("_names", "_ordered")

    def __init__(self, *names: str, ordered: bool = False) -> None:
        self._names = tuple(names)
        self._ordered = ordered

    @property
    def names(self) -> tuple[str, ...]:
        """The declared dimension names, in the order given."""
        return self._names

    @property
    def ordered(self) -> bool:
        """Whether dim order (not just the set) must match."""
        return self._ordered

    def __repr__(self) -> str:
        parts = [repr(n) for n in self._names]
        if self._ordered:
            parts.append("ordered=True")
        return f"Dims({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dims):
            return (self._names, self._ordered) == (other._names, other._ordered)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._names, self._ordered))


class Coords:
    """Declare coordinate variables a DataArray must carry.

    ``Coords("time", "lat")`` declares that those coordinates are present (as
    *labels*, not merely dims — a dim can exist without coordinate values).
    Extra coordinates are allowed; only the declared ones must be present.
    """

    _names: tuple[str, ...]
    __slots__ = ("_names",)

    def __init__(self, *names: str) -> None:
        self._names = tuple(names)

    @property
    def names(self) -> tuple[str, ...]:
        """The declared coordinate names."""
        return self._names

    def __repr__(self) -> str:
        parts = [repr(n) for n in self._names]
        return f"Coords({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Coords):
            return self._names == other._names
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._names)


class Dtype:
    """Declare the expected dtype of a DataArray, e.g. ``Dtype("float64")``.

    By default the check is by numpy *kind*: any float satisfies
    ``Dtype("float64")`` and any signed integer ``Dtype("int32")`` — enough to catch
    an int/float or bool/float mix-up without firing on ``f8`` vs ``f4``.  Pass
    ``exact=True`` to
    require the precise dtype (e.g. to pin memory footprint or a typed sink).
    """

    _dtype: str
    _exact: bool
    __slots__ = ("_dtype", "_exact")

    def __init__(self, dtype: str, *, exact: bool = False) -> None:
        self._dtype = dtype
        self._exact = exact

    @property
    def dtype(self) -> str:
        """The declared dtype string."""
        return self._dtype

    @property
    def exact(self) -> bool:
        """Whether the exact dtype (not just its kind) must match."""
        return self._exact

    def __repr__(self) -> str:
        parts = [repr(self._dtype)]
        if self._exact:
            parts.append("exact=True")
        return f"Dtype({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dtype):
            return (self._dtype, self._exact) == (other._dtype, other._exact)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._dtype, self._exact))


#: Any one schema marker.
SchemaMarker = Dims | Coords | Dtype
