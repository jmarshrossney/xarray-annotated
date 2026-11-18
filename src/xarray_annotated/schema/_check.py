"""Runtime structural validation of a DataArray against declared markers.

The "usage side" of the schema domain: given a `DataArray` and the `Dims` /
`Coords` / `Dtype` markers declared for it, check each one.
Unlike the units domain this **never mutates** — it raises on a mismatch, or returns
the array unchanged, so a decorator can drop it into an argument-validation pass
symmetrically with `check_units`.

The per-property checks are independent functions dispatched by marker type, so a
new structural property is added by writing one checker and one dispatch entry —
nothing else here changes.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from .._config import resolve_enabled
from .._errors import ContractError, Kind, Role
from ._annotations import Coords, Dims, Dtype, SchemaMarker

#: Dtype names meaning any dtype of that kind.  numpy reads each as one dtype, whose
#: kind is the one meant: `np.dtype("float")` is `float64`, of kind `f`.
_KIND_NAMES = frozenset(
    {"float", "int", "uint", "complex", "str", "bytes", "datetime64", "timedelta64"}
)


class SchemaError(ContractError):
    """Raised for a structural mismatch.

    Deliberately **not** a ``ValueError``: a structural mismatch (the *data* fails
    the declaration) is a distinct event from a malformed *declaration* (an
    unparseable dtype, duplicate dim names), which raises ``ValueError`` from
    ``assert_valid_schema``. Keeping them on separate hierarchies means catching
    one never silently swallows the other.
    """


def assert_valid_schema(marker: SchemaMarker, context: str) -> None:
    """Validate a marker *declaration* itself, independent of any DataArray.

    Used at decoration time to fail fast on a malformed declaration (e.g. an
    unparseable dtype string or duplicate dim names) rather than only when the
    decorated function first runs.

    Args:
        marker: The declared `Dims`, `Coords`, or `Dtype` marker.
        context: A label for the error message (e.g. ``"f input 'x'"``).

    Raises:
        ValueError: If the declaration is malformed.
    """
    kind = type(marker).__name__
    if isinstance(marker, Dtype):
        try:
            np.dtype(marker.dtype)
        except TypeError as exc:
            raise ValueError(f"{context}: invalid dtype {marker.dtype!r}.") from exc
    else:  # Dims or Coords: a tuple of names
        for name in marker.names:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{context}: {kind} names must be non-empty strings, "
                    f"got {marker.names!r}."
                )
        if len(set(marker.names)) != len(marker.names):
            raise ValueError(f"{context}: {kind} has duplicate names {marker.names!r}.")


class _Outcome(NamedTuple):
    """What a schema checker found: whether it passed, and if not, why."""

    ok: bool
    detail: str
    kind: Kind
    hint: str | None = None


def _check_dims(da: xr.DataArray, marker: Dims) -> _Outcome:
    """Check a DataArray's dims against a `Dims` marker."""
    actual = tuple(da.dims)
    same_set = set(actual) == set(marker.names)
    if marker.ordered:
        detail = f"dims order mismatch: expected {marker.names}, got {actual}"
        kind = "dims_order" if same_set else "dims_mismatch"
        return _Outcome(actual == marker.names, detail, kind)
    detail = f"dims mismatch: expected {marker.names} in any order, got {actual}"
    return _Outcome(same_set, detail, "dims_mismatch")


def _check_coords(da: xr.DataArray, marker: Coords) -> _Outcome:
    """Check a DataArray carries the declared coords."""
    actual = set(map(str, da.coords))
    missing = [n for n in marker.names if n not in actual]
    detail = f"missing coords {missing}; present: {sorted(actual)}"
    return _Outcome(not missing, detail, "coords_missing")


def _kind(marker: Dtype) -> str:
    """The numpy kind a `Dtype` marker declares."""
    return np.dtype(marker.dtype).kind


def _is_exact(marker: Dtype) -> bool:
    """Whether a `Dtype` marker names a precise dtype, rather than a kind."""
    return marker.dtype not in _KIND_NAMES


def _same_dtype(a: np.dtype, b: np.dtype) -> bool:
    """Whether two dtypes are the same, ignoring byte order.

    Byte order says nothing about what the array is.  xarray makes data native when it
    opens a file, but an array built from raw bytes, or opened with `decode_cf=False`,
    may not be.
    """
    return a.newbyteorder("=") == b.newbyteorder("=")


def _check_dtype(da: xr.DataArray, marker: Dtype) -> _Outcome:
    """Check a DataArray's dtype against a `Dtype` marker."""
    actual = da.dtype
    if not _is_exact(marker):
        detail = f"dtype mismatch: expected any {marker.dtype}, got {actual}"
        return _Outcome(actual.kind == _kind(marker), detail, "dtype_kind")
    declared = np.dtype(marker.dtype)
    detail = f"dtype mismatch: expected {declared}, got {actual}"
    if actual.kind != declared.kind:
        return _Outcome(False, detail, "dtype_kind")
    kind_name = next((k for k in _KIND_NAMES if np.dtype(k).kind == actual.kind), None)
    hint = None
    if kind_name is not None:
        hint = f"Dtype({kind_name!r}) would accept any width"
        detail += f"; {hint}"
    return _Outcome(_same_dtype(actual, declared), detail, "dtype_width", hint)


def dims_compatible(a: Dims, b: Dims) -> bool:
    """Return whether two `Dims` declarations can describe the same array.

    The marker-vs-marker counterpart to validating an array against a `Dims`
    declaration (no array in hand), for a static / build-time check of a
    producer/consumer edge: two declarations are
    *provably inconsistent* only if their dim **sets** differ, or if both pin the
    order (`ordered=True`) and the orders disagree.  A loose declaration on either
    side can always be satisfied, so it never conflicts.

    Args:
        a: A `Dims` marker.
        b: Another `Dims` marker.

    Returns:
        `True` unless the two are provably inconsistent.

    Examples:
        >>> from xarray_annotated.schema import Dims, dims_compatible
        >>> dims_compatible(Dims("x", "y"), Dims("y", "x"))
        True
        >>> dims_compatible(Dims("x"), Dims("x", "y"))
        False
        >>> dims_compatible(Dims("x", "y", ordered=True), Dims("y", "x", ordered=True))
        False
    """
    if set(a.names) != set(b.names):
        return False
    return not (a.ordered and b.ordered and a.names != b.names)


def dtype_compatible(a: Dtype, b: Dtype) -> bool:
    """Return whether two `Dtype` declarations can describe the same array.

    The marker-vs-marker counterpart to validating an array against a `Dtype`
    declaration (no array in hand): two declarations are *provably inconsistent* if
    their numpy **kinds** differ (float vs int), or if both name a precise dtype and
    those differ (`f8` vs `f4`).  A declaration of a kind (`Dtype("float")`) matches any
    width, so it never conflicts on width alone.

    Args:
        a: A `Dtype` marker.
        b: Another `Dtype` marker.

    Returns:
        `True` unless the two are provably inconsistent.

    Raises:
        ValueError: If either dtype is invalid.

    Examples:
        >>> from xarray_annotated.schema import Dtype, dtype_compatible
        >>> dtype_compatible(Dtype("float"), Dtype("float32"))
        True
        >>> dtype_compatible(Dtype("float64"), Dtype("float32"))
        False
        >>> dtype_compatible(Dtype("float"), Dtype("int"))
        False
    """
    assert_valid_schema(a, "dtype_compatible")
    assert_valid_schema(b, "dtype_compatible")
    if _kind(a) != _kind(b):
        return False
    if not (_is_exact(a) and _is_exact(b)):
        return True
    return _same_dtype(np.dtype(a.dtype), np.dtype(b.dtype))


# There is deliberately no `coords_compatible`: two `Coords` declarations are
# lower bounds (each merely requires its names be present, extras allowed), so
# they can never be *proven* inconsistent — a build-time edge check skips coords.


_CHECKERS: dict[type, Callable[[xr.DataArray, Any], _Outcome]] = {
    Dims: _check_dims,
    Coords: _check_coords,
    Dtype: _check_dtype,
}

_FACETS = {Dims: "dims", Coords: "coords", Dtype: "dtype"}


def _found(da: xr.DataArray, marker: SchemaMarker) -> str | tuple[str, ...]:
    """What an array has for the facet a marker declares, for a `Violation`."""
    if isinstance(marker, Dims):
        return tuple(map(str, da.dims))
    if isinstance(marker, Coords):
        return tuple(sorted(map(str, da.coords)))
    return str(da.dtype)


def _check_valid_schema(
    da: xr.DataArray,
    declared: SchemaMarker | list[SchemaMarker],
    name: str,
    qualname: str | None,
    role: Role = "input",
) -> None:
    """`check_schema` on validated markers, whatever `enabled` says."""
    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    for marker in _as_list(declared):
        outcome = _CHECKERS[type(marker)](da, marker)
        if not outcome.ok:
            raise SchemaError(
                f"{prefix} {outcome.detail}",
                name=name,
                qualname=qualname,
                role=role,
                facet=_FACETS[type(marker)],
                kind=outcome.kind,
                declared=marker,
                found=_found(da, marker),
                hint=outcome.hint,
            )


def _as_list(declared: SchemaMarker | list[SchemaMarker]) -> list[SchemaMarker]:
    return [declared] if isinstance(declared, (Dims, Coords, Dtype)) else declared


def check_schema(
    da: xr.DataArray,
    declared: SchemaMarker | list[SchemaMarker],
    name: str,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate `da` against its declared schema marker(s); return it unchanged.

    Runs each marker's checker in turn.  When xarray-annotated is switched off
    (`enabled(False)`) this is a total no-op.

    Args:
        da: The DataArray to validate.
        declared: A single marker or a list of markers declared for `da`.
        name: The parameter/field name, for error messages.
        qualname: Optional qualified function name, prefixed to messages.

    Returns:
        `da`, unchanged (validation never mutates).

    Raises:
        SchemaError: On a mismatch.
        ValueError: If a marker is malformed, or `XARRAY_ANNOTATED_ENABLED` holds an
            unrecognised value.
    """
    if not resolve_enabled():
        return da

    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    for marker in _as_list(declared):
        assert_valid_schema(marker, prefix)
    _check_valid_schema(da, declared, name, qualname)
    return da
