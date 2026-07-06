"""Reading declared units off `typing.Annotated` type hints.

Pure `typing` introspection with no dependency on the active registry or
configuration: this is the "declaration side" of the package, turning
`Annotated[DataArray, Unit("degC")]` metadata into plain unit strings that the
checking layer later validates.

The canonical declaration is the self-identifying `Unit` marker.  A bare unit
string is accepted as a shorthand under a **unit-first** convention: the unit
must precede any descriptive metadata, e.g.
`Annotated[DataArray, "m s-1", "description"]`.
"""

from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
)

from .._annotations import (
    _is_dataarray_type,
    unwrap_annotated,
    walk_signature,
)

# Re-exported so `units._annotations.unwrap_annotated` stays a stable import.
__all__ = ["Unit", "annotated_unit", "units_from_signature", "unwrap_annotated"]


class Unit:
    """Typed marker declaring the expected unit of a DataArray.

    This is the **recommended** way to declare a unit.  Inside ``Annotated`` it is
    self-identifying and independent of metadata ordering, so it composes cleanly
    with schema markers or other ``Annotated``-based tooling::

        Annotated[xr.DataArray, Unit("degC"), SomeOtherMarker(...)]

    A bare string ``Annotated[xr.DataArray, "degC"]`` is also accepted as a
    shorthand (the first string in metadata, by convention) and resolves to the
    same unit.  Prefer the marker whenever the annotation is shared: it removes
    both the order dependence and the ambiguity a bare string has against a
    description string.

    The unit string is *not* validated here — parsing is deferred to
    `assert_valid_unit` at decoration time, exactly as for the bare-string form.
    """

    __slots__ = ("_unit",)

    def __init__(self, unit: str) -> None:
        self._unit = unit

    @property
    def unit(self) -> str:
        """The declared unit string."""
        return self._unit

    def __repr__(self) -> str:
        return f"Unit({self._unit!r})"

    def __str__(self) -> str:
        # The bare unit string, for readable interpolation (e.g. f"expected {m}").
        return self._unit

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._unit == other._unit
        return NotImplemented

    def __hash__(self) -> int:
        # Defining __eq__ drops the default __hash__; restore it so that an
        # Annotated[...] hint carrying a Unit marker stays hashable (typing and
        # downstream tools cache/hash annotations).
        return hash(self._unit)


def annotated_unit(hint: Any) -> str | None:
    """Return the declared unit carried by an `Annotated` type hint, or `None`.

    The unit is the first `Unit(...)` marker in the `Annotated` metadata, and —
    falling back — the first `str`.  So both spellings resolve identically:

        `Annotated[DataArray, Unit("degC")]` → `"degC"`
        `Annotated[DataArray, "degC"]`       → `"degC"`

    The typed `Unit` marker is preferred because it is *self-identifying* and
    order-independent: it owns its slot regardless of what other metadata
    (descriptions, other typed markers) shares the annotation, so

        `Annotated[DataArray, "note", Unit("Pa")]` → `"Pa"` (marker wins)

    The bare-string form keeps the extensible *unit-first* convention: the unit
    may be followed by free-form annotations that are ignored here, so

        `Annotated[DataArray, "m s-1", "z component of velocity"]` → `"m s-1"`

    A description placed before the unit (and with no `Unit` marker present)
    would be mis-read as the unit — but `assert_valid_unit` rejects it unless the
    description itself parses as a valid unit, so the failure is loud.  Non-string
    metadata (ints, unrelated markers) is skipped regardless of position.

    The metadata is only interpreted as a unit when the annotated base type
    is a `DataArray` (the only type that carries units); a `Unit` marker or
    descriptive string on a non-`DataArray` parameter (e.g.
    `Annotated[bool, "toggles X"]`) is *not* a unit and yields `None`.
    Non-`Annotated` hints, or `Annotated` hints whose metadata holds neither a
    `Unit` marker nor a string, also return `None`.

    Args:
        hint: A type hint, typically from `get_type_hints(..., include_extras=True)`.

    Returns:
        The declared unit (first `Unit` marker, else first `str`) if the base
        type is a `DataArray`; `None` otherwise.

    Examples:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> from xarray_annotated.units._annotations import Unit, annotated_unit
        >>> annotated_unit(Annotated[xr.DataArray, Unit("degC")])
        'degC'
        >>> annotated_unit(Annotated[xr.DataArray, "degC"])  # shorthand
        'degC'
        >>> annotated_unit(Annotated[bool, "toggles X"]) is None
        True
    """
    if get_origin(hint) is not Annotated:
        return None
    # get_args(Annotated[T, m1, m2, ...]) == (T, m1, m2, ...); skip the base type.
    args = get_args(hint)
    if not _is_dataarray_type(args[0]):
        return None
    metadata = args[1:]
    # The typed marker wins over a bare string, regardless of relative order.
    for meta in metadata:
        if isinstance(meta, Unit):
            return meta.unit
    for meta in metadata:
        if isinstance(meta, str):
            return meta
    return None


def units_from_signature(
    func: object,
) -> tuple[dict[str, str], dict[str, str] | str | None]:
    """Extract declared units from a function's type annotations.

    Reads `get_type_hints(func, include_extras=True)` and interprets a
    `Unit(...)` marker (or its bare-string shorthand) in the `Annotated`
    metadata as a unit declaration.

    Args:
        func: A callable whose type hints carry `Annotated[DataArray, Unit(...)]`
            metadata (or the bare-string shorthand).

    Returns:
        A `(input_units, output_units)` pair:

        * `input_units` — `dict[str, str]` mapping parameter names to their
            declared unit strings.  Only parameters whose hint carries a unit
            declaration contribute.
        * `output_units` — a `dict[str, str]` (per-field units) if the
            return hint is a `TypedDict` or dataclass; a bare `str` if the return
            is a single `Annotated[DataArray, Unit(...)]`; `None` otherwise.
    """
    return walk_signature(func, annotated_unit)
