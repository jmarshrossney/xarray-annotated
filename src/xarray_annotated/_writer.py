"""Writing declarations *onto* an annotation — the inverse of the readers.

Every domain ships a reader (`Annotated → declaration`: `units_from_signature`,
`schema_from_signature`); this is the one shared *writer* (`declaration →
Annotated`).  A tool that builds functions dynamically (codegen, graph nodes)
needs to stamp a declared contract onto a generated signature; `annotate` returns
a real `Annotated` object it can assign to `fn.__annotations__["return"]`, so the
ordinary `declare_units` / `declare_schema` decorators then read it straight back.

It is deliberately cross-domain — it assembles both the units `Unit` marker and
the schema `Dims` / `Dtype` / `Coords` markers — so it lives at the package root
rather than in either domain, next to the domain-agnostic annotation helpers.
"""

from collections.abc import Iterable
from typing import Annotated, Any

import xarray as xr

from .schema import Coords, Dims, Dtype
from .units import Unit

__all__ = ["annotate"]


def annotate(
    base: Any = xr.DataArray,
    *,
    unit: str | Unit | None = None,
    dims: Iterable[str] | Dims | None = None,
    dtype: str | Dtype | None = None,
    coords: Iterable[str] | Coords | None = None,
) -> Any:
    """Build an `Annotated[base, <markers>]` hint from declared facet values.

    The inverse of the `*_from_signature` readers: given facet values it returns a
    real `Annotated` object carrying the corresponding markers, in a fixed order
    (unit, dims, dtype, coords).  Assign it to a function's return/parameter
    annotation and the `declare_units` / `declare_schema` decorators read it back
    exactly as they would a hand-written one.

    Each facet accepts either a raw value or an already-built marker, so a caller
    holding a mix (e.g. a `Unit` object but bare dim-name tuples) can pass both
    without unwrapping:

        * `unit`   — a unit string (`"Pa"`) or a `Unit`.
        * `dims`   — an iterable of dim names (`("time", "x")`) or a `Dims`.
        * `dtype`  — a dtype string (`"float64"`) or a `Dtype`.
        * `coords` — an iterable of coord names or a `Coords`.

    A facet left as `None` contributes no marker.  When no facet is given, `base`
    is returned unchanged (no `Annotated` wrapper), so `annotate()` is a safe
    no-op default.

    Args:
        base: The base type to annotate (default `xarray.DataArray`).
        unit: Declared unit, or `None`.
        dims: Declared dimensions, or `None`.
        dtype: Declared dtype, or `None`.
        coords: Declared coordinates, or `None`.

    Returns:
        `Annotated[base, <markers>]` if any facet was given; otherwise `base`.

    Examples:
        >>> from typing import Annotated, get_args, get_origin
        >>> import xarray as xr
        >>> from xarray_annotated import annotate
        >>> from xarray_annotated.units import Unit
        >>> hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64")
        >>> get_origin(hint) is Annotated
        True
        >>> Unit("Pa") in get_args(hint)
        True
        >>> annotate() is xr.DataArray
        True
    """
    markers: list[Any] = []
    if unit is not None:
        markers.append(unit if isinstance(unit, Unit) else Unit(unit))
    if dims is not None:
        markers.append(dims if isinstance(dims, Dims) else Dims(*dims))
    if dtype is not None:
        markers.append(dtype if isinstance(dtype, Dtype) else Dtype(dtype))
    if coords is not None:
        markers.append(coords if isinstance(coords, Coords) else Coords(*coords))
    return Annotated[(base, *markers)] if markers else base
