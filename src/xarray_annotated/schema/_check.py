"""Runtime structural validation of a DataArray against declared markers.

The "usage side" of the schema domain: given a `DataArray` and the `Dims` /
`Coords` / `Dtype` markers declared for it, check each one under the active policy.
Unlike the units domain this **never mutates** — it asserts and (per `on_mismatch`)
raises, warns, or ignores, then returns the array unchanged, so a decorator can
drop it into an argument-validation pass symmetrically with `check_units`.

The per-property checks are independent functions dispatched by marker type, so a
new structural property is added by writing one checker and one dispatch entry —
nothing else here changes.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from ._annotations import Coords, Dims, Dtype, SchemaMarker
from ._config import OnMismatch, _validate_on_mismatch, get_policy


class SchemaWarning(UserWarning):
    """Warning issued for a structural mismatch under ``on_mismatch="warn"``."""


class SchemaError(Exception):
    """Raised for a structural mismatch under ``on_mismatch="error"``.

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


def check_dims(da: xr.DataArray, marker: Dims) -> tuple[bool, str]:
    """Check a DataArray's dims against a `Dims` marker. Returns (ok, detail)."""
    actual = tuple(da.dims)
    if marker.ordered:
        ok = actual == marker.names
        detail = f"dims order mismatch: expected {marker.names}, got {actual}"
    else:
        ok = set(actual) == set(marker.names)
        detail = f"dims mismatch: expected {marker.names} in any order, got {actual}"
    return ok, detail


def check_coords(da: xr.DataArray, marker: Coords) -> tuple[bool, str]:
    """Check a DataArray carries the declared coords. Returns (ok, detail)."""
    actual = set(map(str, da.coords))
    missing = [n for n in marker.names if n not in actual]
    detail = f"missing coords {missing}; present: {sorted(actual)}"
    return not missing, detail


def check_dtype(da: xr.DataArray, marker: Dtype) -> tuple[bool, str]:
    """Check a DataArray's dtype against a `Dtype` marker. Returns (ok, detail)."""
    declared = np.dtype(marker.dtype)
    actual = da.dtype
    if marker.exact:
        ok = actual == declared
        detail = f"dtype mismatch: expected exactly {declared}, got {actual}"
    else:
        ok = actual.kind == declared.kind
        detail = (
            f"dtype kind mismatch: expected kind {declared.kind!r} "
            f"(like {declared}), got {actual}"
        )
    return ok, detail


_CHECKERS: dict[type, Callable[[xr.DataArray, Any], tuple[bool, str]]] = {
    Dims: check_dims,
    Coords: check_coords,
    Dtype: check_dtype,
}


def _report(ok: bool, severity: OnMismatch, message: str) -> None:
    """Act on a check result under the resolved severity."""
    if ok or severity == "ignore":
        return
    if severity == "error":
        raise SchemaError(message)
    warnings.warn(message, SchemaWarning, stacklevel=3)


def check_schema(
    da: xr.DataArray,
    declared: SchemaMarker | list[SchemaMarker],
    name: str,
    on_mismatch: OnMismatch | None = None,
    qualname: str | None = None,
) -> xr.DataArray:
    """Validate `da` against its declared schema marker(s); return it unchanged.

    Runs each marker's checker under the effective severity, resolved per marker
    as: the marker's own `on_mismatch` override, else the `on_mismatch` argument,
    else the policy default (`get_policy().on_mismatch`).  When the master switch
    is off (`enabled=False`) this is a total no-op.

    Args:
        da: The DataArray to validate.
        declared: A single marker or a list of markers declared for `da`.
        name: The parameter/field name, for error messages.
        on_mismatch: Per-call default severity, overriding the policy default but
            overridden by a marker's own `on_mismatch`.
        qualname: Optional qualified function name, prefixed to messages.

    Returns:
        `da`, unchanged (validation never mutates).

    Raises:
        SchemaError: On a mismatch when the effective severity is ``"error"``.
    """
    pol = get_policy()
    if not pol.enabled:
        return da
    if on_mismatch is not None:
        on_mismatch = _validate_on_mismatch(on_mismatch)

    markers = [declared] if isinstance(declared, (Dims, Coords, Dtype)) else declared
    prefix = f"[{qualname}] {name!r}" if qualname else f"{name!r}"
    for marker in markers:
        ok, detail = _CHECKERS[type(marker)](da, marker)
        severity = marker.on_mismatch or on_mismatch or pol.on_mismatch
        _report(ok, severity, f"{prefix} {detail}")
    return da
