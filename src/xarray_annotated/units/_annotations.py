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

from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    get_args,
    get_origin,
)

from .._annotations import (
    _is_dataarray_type,
    _single_marker,
    unwrap_annotated,
    walk_signature,
)
from ._config import (
    OnInexact,
    OnMissing,
    OnOutput,
    _validate_on_inexact,
    _validate_on_missing,
    _validate_on_output,
)

# Re-exported so `units._annotations.unwrap_annotated` stays a stable import.
__all__ = ["Unit", "annotated_unit", "units_from_signature", "unwrap_annotated"]


def _optional[T: str](validate: Callable[[str], T], value: T | None) -> T | None:
    """Validate an optional override at construction; `None` means no override."""
    return None if value is None else validate(value)


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

    Each units axis can be set **for this declaration**, so the intent sits next
    to the parameter that depends on it::

        tair: Annotated[xr.DataArray, Unit("degC", on_inexact="convert")]
        total: Annotated[xr.DataArray, Unit("g m-2 yr-1", on_inexact="error")]

    A marker override beats the decorator keyword, which beats the default;
    switching the package off (`enabled(False)`) still disables everything.
    `on_missing` and `on_inexact` apply where the marker declares an input,
    `on_output` where it declares an output; an override that does not apply
    where the marker sits is ignored, so one marker can be reused on both sides.

    Equality and hashing include the overrides, as for every other marker:
    `Unit("Pa", on_output="strict") != Unit("Pa")`.  To compare only the units,
    compare `.unit`, or use `units_equal` to compare spellings by meaning.

    Args:
        unit: The declared unit string.
        on_missing: Override the on-missing axis for this declaration.
        on_inexact: Override the on-inexact axis for this declaration.
        on_output: Override the on-output axis for this declaration.

    Raises:
        ValueError: If an override is not a valid value for its axis.
    """

    _unit: str
    _on_missing: OnMissing | None
    _on_inexact: OnInexact | None
    _on_output: OnOutput | None
    __slots__ = ("_on_inexact", "_on_missing", "_on_output", "_unit")

    def __init__(
        self,
        unit: str,
        *,
        on_missing: OnMissing | None = None,
        on_inexact: OnInexact | None = None,
        on_output: OnOutput | None = None,
    ) -> None:
        self._unit = unit
        self._on_missing = _optional(_validate_on_missing, on_missing)
        self._on_inexact = _optional(_validate_on_inexact, on_inexact)
        self._on_output = _optional(_validate_on_output, on_output)

    @property
    def unit(self) -> str:
        """The declared unit string."""
        return self._unit

    @property
    def on_missing(self) -> OnMissing | None:
        """This declaration's on-missing override, or `None`."""
        return self._on_missing

    @property
    def on_inexact(self) -> OnInexact | None:
        """This declaration's on-inexact override, or `None`."""
        return self._on_inexact

    @property
    def on_output(self) -> OnOutput | None:
        """This declaration's on-output override, or `None`."""
        return self._on_output

    def _key(self) -> tuple[str, str | None, str | None, str | None]:
        return (self._unit, self._on_missing, self._on_inexact, self._on_output)

    def __repr__(self) -> str:
        parts = [repr(self._unit)]
        for axis in ("on_missing", "on_inexact", "on_output"):
            value = getattr(self, f"_{axis}")
            if value is not None:
                parts.append(f"{axis}={value!r}")
        return f"Unit({', '.join(parts)})"

    def __str__(self) -> str:
        # The bare unit string, for readable interpolation (e.g. f"expected {m}").
        return self._unit

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._key() == other._key()
        return NotImplemented

    def __hash__(self) -> int:
        # Defining __eq__ drops the default __hash__; restore it so that an
        # Annotated[...] hint carrying a Unit marker stays hashable (typing and
        # downstream tools cache/hash annotations).
        return hash(self._key())


def annotated_unit(hint: Any) -> str | None:
    """Return the declared unit carried by an `Annotated` type hint, or `None`.

    The unit is the `Unit(...)` marker in the `Annotated` metadata (at most one), and
    — falling back — the first `str`.  So both spellings resolve identically:

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
        The declared unit (the `Unit` marker, else the first `str`) if the base
        type is a `DataArray`; `None` otherwise.

    Raises:
        ValueError: If the hint carries more than one `Unit` marker.

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
    marker = _unit_marker(hint)
    return None if marker is None else marker.unit


def _unit_marker(hint: Any) -> Unit | None:
    """`annotated_unit`, but returning the marker so its overrides survive.

    A bare-string shorthand comes back as `Unit(string)`.  Used by the
    marker-carrying readers (`declarations_from_signature`, and so `enforce`);
    `annotated_unit` and `units_from_signature` return the plain string.
    """
    if get_origin(hint) is not Annotated:
        return None
    # get_args(Annotated[T, m1, m2, ...]) == (T, m1, m2, ...); skip the base type.
    args = get_args(hint)
    if not _is_dataarray_type(args[0]):
        return None
    metadata = args[1:]
    # The typed marker wins over a bare string, regardless of relative order.
    marker = _single_marker(metadata, Unit)
    if marker is not None:
        return marker
    for meta in metadata:
        if isinstance(meta, str):
            return Unit(meta)
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
