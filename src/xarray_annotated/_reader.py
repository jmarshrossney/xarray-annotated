"""Reading *all* declared facets off a hint or a signature — the one reader.

Where the writer `annotate` turns facet values into an `Annotated` hint,
`declaration_from_hint` and `declarations_from_signature` turn a hint, or a whole
signature, back into a facet value per declared DataArray.  They are the read-side
counterpart to `annotate`, and their round-trip is exact.

The unified value `Declared` is deliberately *homogeneous*: every slot holds a
marker (`Unit` / `Dims` / `Dtype` / `Coords` / `Freq`) or `None`.  Markers are the
package's public currency, since they carry per-declaration settings a plain string
could not, so a facet-generic consumer gets one marker per slot.

One slot per marker type is lossless because a hint may declare each facet at most
once: `declaration_from_hint` rejects a second `Unit` / `Dims` / `Dtype` / `Coords` /
`Freq` on one hint as a malformed declaration (`ValueError`).

Like `annotate`, this is cross-domain — it assembles every domain's markers into one
value — so it lives at the package root rather than in any one domain.
"""

import types
import typing
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    NotRequired,
    Required,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from ._annotations import _is_dataarray_type, _single_marker, walk_signature
from .schema import Coords, Dims, Dtype
from .schema._annotations import SchemaMarker
from .temporal import Freq
from .units import Unit

__all__ = ["Declared", "declaration_from_hint", "declarations_from_signature"]

#: TypedDict qualifiers, which `get_type_hints(..., include_extras=True)` keeps.
_QUALIFIERS = tuple(
    q for q in (Required, NotRequired, getattr(typing, "ReadOnly", None)) if q
)


@dataclass(frozen=True, slots=True)
class Declared:
    """Every facet declared on one DataArray hint, one marker (or `None`) per slot.

    The uniform value produced by `declaration_from_hint` and
    `declarations_from_signature`: a homogeneous bag of markers, so facet-generic
    code can treat each slot identically.  Each slot holds the corresponding marker
    or `None` when that facet was not declared; `.unit.unit` recovers the unit
    string.

    A hint carrying two markers of one type never reaches here: reading it raises
    `ValueError`, so no declaration is dropped.

    The field order (`unit`, `dims`, `dtype`, `coords`, `freq`) matches `annotate`'s
    keyword arguments, so `annotate(unit=d.unit, dims=d.dims, dtype=d.dtype,
    coords=d.coords, freq=d.freq)` rebuilds the hint this was read from.
    """

    unit: Unit | None = None
    dims: Dims | None = None
    dtype: Dtype | None = None
    coords: Coords | None = None
    freq: Freq | None = None

    @property
    def schema(self) -> list[SchemaMarker]:
        """The declared schema markers, as `check_schema` takes them."""
        return [m for m in (self.dims, self.dtype, self.coords) if m is not None]


def declaration_from_hint(hint: Any) -> Declared | None:
    """Read every facet declared on one type hint into a `Declared`.

    The single-hint counterpart to `declarations_from_signature`, and an extractor
    `walk_signature` can drive.  A non-`Annotated` hint, a base type that is not a
    `DataArray`, and a hint with no marker all declare nothing.

    Reads through a union (`Annotated[DataArray, ...] | None`) and the `TypedDict`
    qualifiers (`NotRequired[Annotated[DataArray, ...]]`, or the other way round), so
    every way of spelling an optional value declares the same thing, and through a
    `type` alias.

    Args:
        hint: A type hint, typically from `get_type_hints(..., include_extras=True)`.

    Returns:
        The declared facets, or `None` when the hint declares none.

    Raises:
        ValueError: If the hint carries two markers of the same type, or is a union
            of two declared members, since a value could not say which one applies.

    Examples:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> from xarray_annotated import Dims, Unit, declaration_from_hint
        >>> declared = declaration_from_hint(
        ...     Annotated[xr.DataArray, Unit("Pa"), Dims("time"), "surface pressure"]
        ... )
        >>> declared.unit, declared.schema
        (Unit('Pa'), [Dims('time')])
        >>> declaration_from_hint(Annotated[xr.DataArray, "surface pressure"]) is None
        True
    """
    if isinstance(hint, TypeAliasType):
        return declaration_from_hint(hint.__value__)
    origin = get_origin(hint)
    if origin in _QUALIFIERS:
        return declaration_from_hint(get_args(hint)[0])
    if origin in (Union, types.UnionType):
        members = [declaration_from_hint(arg) for arg in get_args(hint)]
        declared = [member for member in members if member is not None]
        if len(declared) > 1:
            raise ValueError(
                f"one union declares {len(declared)} DataArray hints "
                f"({', '.join(map(repr, declared))}); declare at most one"
            )
        return declared[0] if declared else None
    if origin is not Annotated:
        return None
    base, *metadata = get_args(hint)
    if get_origin(base) in _QUALIFIERS:
        base = get_args(base)[0]
    if not _is_dataarray_type(base):
        return None
    declared = Declared(
        unit=_single_marker(metadata, Unit),
        dims=_single_marker(metadata, Dims),
        dtype=_single_marker(metadata, Dtype),
        coords=_single_marker(metadata, Coords),
        freq=_single_marker(metadata, Freq),
    )
    return None if declared == Declared() else declared


def declarations_from_signature(
    func: object,
) -> tuple[dict[str, Declared], dict[str, Declared] | Declared | None]:
    """Read every declared facet off a function's signature in one uniform shape.

    Each payload is a `Declared` carrying the unit, dims, dtype, coords and freq
    declared on one DataArray hint.

    Args:
        func: A callable whose parameters/return may carry
            `Annotated[DataArray, ...]` metadata.

    Returns:
        An `(inputs, output)` pair, mirroring `walk_signature` with a `Declared`
        payload:

        * `inputs` — `dict[str, Declared]` mapping each declared parameter to its
            facets.  Parameters carrying no declaration are omitted.
        * `output` — a `dict[str, Declared]` (per-field) for a `TypedDict` or
            dataclass return, a single `Declared` for one declared DataArray
            return, or `None` when the return carries no declaration.
    """
    return walk_signature(func, declaration_from_hint)
