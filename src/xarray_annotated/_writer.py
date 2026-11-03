"""Writing declarations *onto* an annotation — the inverse of the readers.

Where `declaration_from_hint` reads `Annotated → declaration`, this is the one
shared *writer* (`declaration → Annotated`).  A tool that builds functions
dynamically (codegen, graph nodes) needs to stamp a declared contract onto a
generated signature;
`annotate` returns a real `Annotated` object it can assign to
`fn.__annotations__["return"]`, so the ordinary `enforce` decorator then reads it
straight back.

It is deliberately cross-domain — it assembles the units `Unit` marker, the schema
`Dims` / `Dtype` / `Coords` markers, and the temporal `Freq` marker — so it lives at
the package root rather than in any one domain, next to the domain-agnostic
annotation helpers.
"""

from collections.abc import Iterable
from typing import Annotated, Any

import xarray as xr

from .schema import Coords, Dims, Dtype
from .temporal import Freq
from .units import Unit

__all__ = ["annotate"]


def annotate(
    base: Any = xr.DataArray,
    *,
    unit: str | Unit | None = None,
    dims: Iterable[str] | Dims | None = None,
    dtype: str | Dtype | None = None,
    coords: Iterable[str] | Coords | None = None,
    freq: str | Freq | None = None,
) -> Any:
    """Build an `Annotated[base, <markers>]` hint from declared facet values.

    The inverse of `declaration_from_hint`: given facet values it returns a
    real `Annotated` object carrying the corresponding markers, in a fixed order
    (unit, dims, dtype, coords, freq).  Assign it to a function's return/parameter
    annotation and the `enforce` decorator reads it back exactly as it would a
    hand-written one.

    Each facet accepts either a raw value or an already-built marker, so a caller
    holding a mix (e.g. a `Unit` object but bare dim-name tuples) can pass both
    without unwrapping:

        * `unit`   — a unit string (`"Pa"`) or a `Unit`.
        * `dims`   — a dim name, an iterable of them (`("time", "x")`), or a `Dims`.
        * `dtype`  — a dtype string (`"float64"`) or a `Dtype`.
        * `coords` — a coord name, an iterable of them, or a `Coords`.
        * `freq`   — an offset string (`"7D"`) or a `Freq`.

    A facet left as `None` contributes no marker.  When no facet is given, `base`
    is returned unchanged (no `Annotated` wrapper), so `annotate()` is a safe
    no-op default.

    Args:
        base: The base type to annotate (default `xarray.DataArray`).
        unit: Declared unit, or `None`.
        dims: Declared dimensions, or `None`.
        dtype: Declared dtype, or `None`.
        coords: Declared coordinates, or `None`.
        freq: Declared temporal frequency, or `None`.

    Returns:
        `Annotated[base, <markers>]` if any facet was given; otherwise `base`.

    Examples:
        >>> from typing import Annotated, get_args, get_origin
        >>> import xarray as xr
        >>> from xarray_annotated import Unit, annotate
        >>> hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64")
        >>> get_origin(hint) is Annotated
        True
        >>> Unit("Pa") in get_args(hint)
        True
        >>> annotate() is xr.DataArray
        True
    """
    if isinstance(dims, str):
        dims = (dims,)
    if isinstance(coords, str):
        coords = (coords,)
    markers: list[Any] = []
    if unit is not None:
        markers.append(unit if isinstance(unit, Unit) else Unit(unit))
    if dims is not None:
        markers.append(dims if isinstance(dims, Dims) else Dims(*dims))
    if dtype is not None:
        markers.append(dtype if isinstance(dtype, Dtype) else Dtype(dtype))
    if coords is not None:
        markers.append(coords if isinstance(coords, Coords) else Coords(*coords))
    if freq is not None:
        markers.append(freq if isinstance(freq, Freq) else Freq(freq))
    return Annotated[(base, *markers)] if markers else base
