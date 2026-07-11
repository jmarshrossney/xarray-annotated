"""Reading *all* declared facets off a signature — the unified reader.

Every domain ships its own reader (`units_from_signature`, `schema_from_signature`,
`freq_from_signature`) returning a domain-shaped payload; this is the one
cross-domain reader that collapses them into a single uniform value.  Where the
writer `annotate` turns facet values into an `Annotated` hint,
`declarations_from_signature` turns a whole signature back into a facet value per
declared DataArray — it is the read-side counterpart to `annotate`, and their
round-trip is exact.

The unified value `Declared` is deliberately *homogeneous*: every slot holds a
marker (`Unit` / `Dims` / `Dtype` / `Coords` / `Freq`) or `None`.  Markers are the
package's public currency — the schema and temporal markers are load-bearing (they
carry strictness that has no bare-string form) — so a facet-generic consumer wants
one marker per slot rather than a mix of strings and objects.  A bare-string unit
shorthand is normalised to a `Unit`, so the shape does not depend on how the unit
was spelled.

Like `annotate`, this is cross-domain — it assembles every domain's markers into one
value — so it lives at the package root rather than in any one domain.
"""

from dataclasses import dataclass
from typing import Any

from ._annotations import walk_signature
from .schema import Coords, Dims, Dtype
from .schema._annotations import annotated_schema
from .temporal import Freq
from .temporal._annotations import annotated_freq
from .units import Unit
from .units._annotations import annotated_unit

__all__ = ["Declared", "declarations_from_signature"]


@dataclass(frozen=True, slots=True)
class Declared:
    """Every facet declared on one DataArray hint, one marker (or `None`) per slot.

    The uniform value produced by `declarations_from_signature`: a homogeneous
    bag of markers, so facet-generic code can treat each slot identically.  Each
    slot holds the corresponding marker or `None` when that facet was not
    declared; the unit is always a `Unit` (its bare-string shorthand is
    normalised on read), so `.unit.unit` recovers the string.

    The field order (`unit`, `dims`, `dtype`, `coords`, `freq`) matches `annotate`'s
    keyword arguments, so `annotate(unit=d.unit, dims=d.dims, dtype=d.dtype,
    coords=d.coords, freq=d.freq)` rebuilds the hint this was read from.
    """

    unit: Unit | None = None
    dims: Dims | None = None
    dtype: Dtype | None = None
    coords: Coords | None = None
    freq: Freq | None = None


def _annotated_declaration(hint: Any) -> Declared | None:
    """Combine the per-domain extractors into one `Declared`, or `None`.

    Delegates to the existing `annotated_unit`, `annotated_schema` and
    `annotated_freq` so the "is this an `Annotated[DataArray, ...]` hint?" logic is
    not duplicated, then folds their payloads into a single homogeneous value.
    Returns `None` when the hint declares no facet, which is `walk_signature`'s "no
    declaration" signal.
    """
    unit = annotated_unit(hint)
    markers = annotated_schema(hint) or ()
    dims = next((m for m in markers if isinstance(m, Dims)), None)
    dtype = next((m for m in markers if isinstance(m, Dtype)), None)
    coords = next((m for m in markers if isinstance(m, Coords)), None)
    freq = annotated_freq(hint)
    if all(facet is None for facet in (unit, dims, dtype, coords, freq)):
        return None
    return Declared(
        unit=Unit(unit) if unit is not None else None,
        dims=dims,
        dtype=dtype,
        coords=coords,
        freq=freq,
    )


def declarations_from_signature(
    func: object,
) -> tuple[dict[str, Declared], dict[str, Declared] | Declared | None]:
    """Read every declared facet off a function's signature in one uniform shape.

    The all-facets counterpart to `units_from_signature` / `schema_from_signature`:
    a single reader whose payload is a `Declared` carrying the unit, dims, dtype,
    and coords declared on a DataArray hint at once.  A consumer that would
    otherwise call both per-domain readers and merge their differently-shaped
    results can call this once and route a uniform value instead.

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
    return walk_signature(func, _annotated_declaration)
