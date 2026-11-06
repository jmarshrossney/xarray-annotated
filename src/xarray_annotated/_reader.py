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
from dataclasses import dataclass, field, replace
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

from ._annotations import _is_dataarray_type, _single_marker, _walk_signature
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

    `others` holds what the hint's union allows besides a `DataArray`, such as
    `NoneType` for `Annotated[DataArray, ...] | None`.  It is not a marker, so it takes
    no part in equality.
    """

    unit: Unit | None = None
    dims: Dims | None = None
    dtype: Dtype | None = None
    coords: Coords | None = None
    freq: Freq | None = None
    others: tuple[Any, ...] = field(default=(), compare=False, repr=False)

    @property
    def schema(self) -> list[SchemaMarker]:
        """The declared schema markers, as `check_schema` takes them."""
        return [m for m in (self.dims, self.dtype, self.coords) if m is not None]


def _enter_alias(alias: TypeAliasType, path: frozenset[Any]) -> frozenset[Any]:
    """Add an alias to the path being read.

    Raises:
        ValueError: If the path already holds it, since it then refers to itself and
            could never resolve to a declaration.  A self-reference inside a
            container is fine: `type R = Annotated[...] | list[R]` is not entered.
    """
    if alias in path:
        raise ValueError(
            f"type alias {alias.__name__} refers to itself, so it declares nothing"
        )
    return path | {alias}


def _flatten_aliases(
    base: Any, metadata: list[Any], path: frozenset[Any]
) -> tuple[Any, list[Any]]:
    """Unwrap `type` aliases in `Annotated`'s base slot, keeping their metadata.

    `Annotated` flattens a nested `Annotated`, inner metadata first, but cannot see
    inside a lazy alias, so this does the same for one.

    Returns:
        The first base that is not a `type` alias, and the merged metadata.
    """
    if not isinstance(base, TypeAliasType):
        return base, metadata
    path = _enter_alias(base, path)
    value = base.__value__
    if get_origin(value) is Annotated:
        value, *inner = get_args(value)
        metadata = [*inner, *metadata]
    return _flatten_aliases(value, metadata, path)


def _read_hint(hint: Any, path: frozenset[Any] = frozenset()) -> Declared | None:
    """Read one hint, tracking the `type` aliases entered on the way to it."""
    if isinstance(hint, TypeAliasType):
        return _read_hint(hint.__value__, _enter_alias(hint, path))
    origin = get_origin(hint)
    if origin in _QUALIFIERS:
        return _read_hint(get_args(hint)[0], path)
    if origin in (Union, types.UnionType):
        members = [_read_hint(arg, path) for arg in get_args(hint)]
        declared = [member for member in members if member is not None]
        if len(declared) > 1:
            raise ValueError(
                f"one union declares {len(declared)} DataArray hints "
                f"({', '.join(map(repr, declared))}); declare at most one"
            )
        if not declared:
            return None
        others = tuple(
            arg
            for arg, member in zip(get_args(hint), members, strict=True)
            if member is None
        )
        return replace(declared[0], others=(*declared[0].others, *others))
    if origin is not Annotated:
        return None
    base, *metadata = get_args(hint)
    if get_origin(base) in _QUALIFIERS:
        base = get_args(base)[0]
    base, metadata = _flatten_aliases(base, metadata, path)
    if not _is_dataarray_type(base):
        return None
    declared = Declared(
        unit=_single_marker(metadata, Unit),
        dims=_single_marker(metadata, Dims),
        dtype=_single_marker(metadata, Dtype),
        coords=_single_marker(metadata, Coords),
        freq=_single_marker(metadata, Freq),
    )
    if declared == Declared():
        return None
    others = (
        [arg for arg in get_args(base) if not _is_dataarray_type(arg)]
        if get_origin(base) in (Union, types.UnionType)
        else []
    )
    if others:
        names = " | ".join(
            "None" if arg is types.NoneType else getattr(arg, "__name__", repr(arg))
            for arg in others
        )
        raise ValueError(
            f"{hint!r} reads as declaring its markers for {names} too, but they apply "
            f"only to a DataArray. Write Annotated[DataArray, ...] | {names}."
        )
    return declared


def declaration_from_hint(hint: Any) -> Declared | None:
    """Read every facet declared on one type hint into a `Declared`.

    The single-hint counterpart to `declarations_from_signature`, and an extractor
    `_walk_signature` can drive.  A non-`Annotated` hint, a base type that is not a
    `DataArray`, and a hint with no marker all declare nothing.

    Reads through a union (`Annotated[DataArray, ...] | None`) and the `TypedDict`
    qualifiers (`NotRequired[Annotated[DataArray, ...]]`, or the other way round), and
    through a `type` alias, whose markers merge with any written around it
    (`Annotated[Alias, Dims("x")]`).

    Args:
        hint: A type hint, typically from `get_type_hints(..., include_extras=True)`.

    Returns:
        The declared facets, or `None` when the hint declares none.

    Raises:
        ValueError: If the hint carries two markers of the same type, or is a union
            of two declared members, since a value could not say which one applies,
            or puts markers on a union with another type, as in
            `Annotated[DataArray | None, ...]`, where the union belongs outside the
            `Annotated`, or is a `type` alias that refers to itself.

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
    return _read_hint(hint)


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
        An `(inputs, output)` pair, mirroring `_walk_signature` with a `Declared`
        payload:

        * `inputs` — `dict[str, Declared]` mapping each declared parameter to its
            facets.  Parameters carrying no declaration are omitted.
        * `output` — a `dict[str, Declared]` (per-field) for a `TypedDict` or
            dataclass return, a single `Declared` for one declared DataArray
            return, or `None` when the return carries no declaration.
    """
    return _walk_signature(func, declaration_from_hint)
