"""`enforce`: apply every facet declared on a signature, in one decorator.

The single cross-domain decorator.  It reads a function's
`Annotated[DataArray, ...]` hints once, at decoration time, and on every call
applies each declared facet in a fixed order:

1. **inputs** — units are converted (and restamped) *first*, so the structural
   and temporal checks then see the array the function body will see;
2. the function runs;
3. **outputs** — the declared unit is applied *first* (stamped, or converted for
   a pint-quantified array), so the structural and temporal checks see the
   value the caller will receive.

That order is the reason this decorator exists: the per-domain decorators it
replaces could be stacked in any order, and only one order was correct.

Like the per-domain decorators before it, this is deliberately the *convenience*
layer over public primitives — the `declarations_from_signature` reader and the
`check_units` / `apply_output_units` / `check_schema` / `check_freq` checkers.  A
tool needing different behaviour should build its own consumer from those.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Collection
from typing import Any, get_args

import xarray as xr

from ._config import resolve_enabled
from ._reader import Declared, declarations_from_signature
from .schema import _config as _schema
from .schema._annotations import SchemaMarker
from .schema._check import assert_valid_schema, check_schema
from .temporal import _config as _temporal
from .temporal._check import assert_valid_freq, check_freq
from .units import _config as _units
from .units._check import (
    apply_output_units,
    assert_valid_unit,
    check_units,
    require_dataarray,
)

__all__ = ["enforce"]

#: The `Declared` slots each domain owns.
_SLOTS = {
    "units": ("unit",),
    "schema": ("dims", "dtype", "coords"),
    "temporal": ("freq",),
}


def _restrict(declared: Declared, domains: Collection[str]) -> Declared | None:
    """Blank the slots of domains not in `domains`; `None` if nothing is left."""
    blank = {
        slot: None for d, slots in _SLOTS.items() if d not in domains for slot in slots
    }
    restricted = dataclasses.replace(declared, **blank) if blank else declared
    return None if restricted == Declared() else restricted


def _schema_markers(declared: Declared) -> list[SchemaMarker]:
    return [
        m for m in (declared.dims, declared.dtype, declared.coords) if m is not None
    ]


def _assert_valid(declared: Declared, context: str) -> None:
    """Fail fast on a malformed declaration (independent of policy)."""
    if declared.unit is not None:
        assert_valid_unit(declared.unit, context)
    for marker in _schema_markers(declared):
        assert_valid_schema(marker, context)
    if declared.freq is not None:
        assert_valid_freq(declared.freq, context)


def _set_field(result: Any, name: str, value: xr.DataArray, qualname: str) -> None:
    """Write a converted output back onto a dataclass return value.

    Only reached when applying the declared unit produced a *new* array, which
    for a dataclass means a pint-quantified field that needed converting — a
    plain field is stamped in place and never comes through here.

    A frozen dataclass cannot take the write. `setattr` raises
    `FrozenInstanceError` (an `AttributeError`, despite reading like a type
    error), which on its own says nothing about units; re-raise it with the
    cause and the way out.

    Raises:
        dataclasses.FrozenInstanceError: If `result` is frozen.
    """
    try:
        setattr(result, name, value)
    except dataclasses.FrozenInstanceError as exc:
        raise dataclasses.FrozenInstanceError(
            f"[{qualname}] output {name!r} is pint-quantified and needs converting "
            f"to its declared unit, but the returned dataclass is frozen so the "
            f"converted array cannot be written back. Dequantify the field in the "
            f"function body (`.pint.dequantify()`), or drop `frozen=True`."
        ) from exc


def _enforce(
    func: Callable[..., Any] | None,
    *,
    domains: Collection[str],
    units_on_missing: _units.OnMissing | None = None,
    units_on_inexact: _units.OnInexact | None = None,
    units_on_output: _units.OnOutput | None = None,
    schema_on_mismatch: _schema.OnMismatch | None = None,
    temporal_on_mismatch: _temporal.OnMismatch | None = None,
    temporal_on_uninferable: _temporal.OnUninferable | None = None,
) -> Callable[..., Any]:
    """`enforce`, restricted to `domains` (the deprecated decorators' kernel)."""
    for keyword, value, allowed in (
        ("units_on_missing", units_on_missing, _units.OnMissing),
        ("units_on_inexact", units_on_inexact, _units.OnInexact),
        ("units_on_output", units_on_output, _units.OnOutput),
        ("schema_on_mismatch", schema_on_mismatch, _schema.OnMismatch),
        ("temporal_on_mismatch", temporal_on_mismatch, _temporal.OnMismatch),
        ("temporal_on_uninferable", temporal_on_uninferable, _temporal.OnUninferable),
    ):
        # Checked here rather than by the domain validators, which name the
        # unprefixed axis (`on_missing`) instead of the keyword the caller passed.
        if value is not None and value not in get_args(allowed):
            raise ValueError(
                f"Invalid {keyword} {value!r}. "
                f"Choose one of {sorted(get_args(allowed))}."
            )

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        read_inputs, read_output = declarations_from_signature(fn)
        inputs = {
            name: d
            for name, raw in read_inputs.items()
            if (d := _restrict(raw, domains)) is not None
        }
        output: dict[str, Declared] | Declared | None
        if isinstance(read_output, dict):
            output = {
                name: d
                for name, raw in read_output.items()
                if (d := _restrict(raw, domains)) is not None
            }
        elif isinstance(read_output, Declared):
            output = _restrict(read_output, domains)
        else:
            output = None
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        for name, decl in inputs.items():
            _assert_valid(decl, f"{qualname} input {name!r}")
        if isinstance(output, Declared):
            _assert_valid(output, f"{qualname} output")
        elif isinstance(output, dict):
            for name, decl in output.items():
                _assert_valid(decl, f"{qualname} output {name!r}")

        def _apply_output(value: Any, decl: Declared, name: str) -> Any:
            # Units first, so structure and frequency are checked on what the
            # caller receives.  apply_output_units may return a *new* array (a
            # converted quantified one), so its result is always used.
            if not isinstance(value, xr.DataArray):
                if decl.unit is not None:
                    require_dataarray(value, decl.unit, name, units_on_output, qualname)
                return value
            if decl.unit is not None:
                value = apply_output_units(
                    value, decl.unit, name, units_on_output, qualname
                )
            if markers := _schema_markers(decl):
                check_schema(value, markers, name, schema_on_mismatch, qualname)
            if decl.freq is not None:
                check_freq(
                    value,
                    decl.freq,
                    name,
                    temporal_on_mismatch,
                    temporal_on_uninferable,
                    qualname,
                )
            return value

        def _outputs(result: Any) -> Any:
            if isinstance(output, Declared):
                return _apply_output(result, output, "return")
            if isinstance(output, dict):
                if isinstance(result, dict):
                    for name, decl in output.items():
                        # A missing key is checked as None, so "require" sees it.
                        applied = _apply_output(result.get(name), decl, name)
                        if name in result:
                            result[name] = applied
                elif dataclasses.is_dataclass(result):
                    for name, decl in output.items():
                        value = getattr(result, name, None)
                        applied = _apply_output(value, decl, name)
                        if applied is not value:
                            _set_field(result, name, applied, qualname)
            return result

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not resolve_enabled():
                # Switched off: a total no-op, conversion and stamping included.
                return fn(*args, **kwargs)
            if inputs:
                bound = sig.bind_partial(*args, **kwargs)
                # bind_partial omits parameters left at their default; apply
                # them so a declared default is checked (and converted) too.
                bound.apply_defaults()
                for name, val in list(bound.arguments.items()):
                    decl = inputs.get(name)
                    if decl is None or not isinstance(val, xr.DataArray):
                        continue
                    if decl.unit is not None:
                        val = check_units(
                            val,
                            decl.unit,
                            name,
                            units_on_missing,
                            units_on_inexact,
                            qualname,
                        )
                        bound.arguments[name] = val
                    if markers := _schema_markers(decl):
                        check_schema(
                            val,
                            markers,
                            name,
                            schema_on_mismatch,
                            qualname,
                        )
                    if decl.freq is not None:
                        check_freq(
                            val,
                            decl.freq,
                            name,
                            temporal_on_mismatch,
                            temporal_on_uninferable,
                            qualname,
                        )
                args, kwargs = bound.args, bound.kwargs
            return _outputs(fn(*args, **kwargs))

        wrapper.__annotations__ = dict(fn.__annotations__)
        return wrapper

    return decorate if func is None else decorate(func)


def enforce(
    func: Callable[..., Any] | None = None,
    *,
    units_on_missing: _units.OnMissing | None = None,
    units_on_inexact: _units.OnInexact | None = None,
    units_on_output: _units.OnOutput | None = None,
    schema_on_mismatch: _schema.OnMismatch | None = None,
    temporal_on_mismatch: _temporal.OnMismatch | None = None,
    temporal_on_uninferable: _temporal.OnUninferable | None = None,
) -> Callable[..., Any]:
    """Apply every facet declared on a function's signature at runtime.

    Reads the decorated function's type hints once: parameters annotated
    `Annotated[DataArray, ...]` declare inputs, and a `TypedDict` / dataclass
    return (or a bare `Annotated[DataArray, ...]` return) declares outputs.  Any
    mix of `Unit` (or its bare-string shorthand), `Dims` / `Coords` / `Dtype`,
    and `Freq` markers may sit on one hint.

    On each call the wrapper:

    1. converts every declared input to its unit (`check_units`), then validates
       its structure (`check_schema`) and frequency (`check_freq`) on the
       converted array;
    2. runs the wrapped function;
    3. applies each declared output unit (`apply_output_units`: stamped, or
       converted when pint-quantified), then validates structure and frequency
       on the value returned to the caller.

    Only `DataArray` values are touched; anything else passes through, except
    that under `on_output="require"` a declared output that is not a
    `DataArray` (including `None` or a missing key) raises.  When
    xarray-annotated is switched off (`xra.enabled(False)` or
    `XARRAY_ANNOTATED_ENABLED=false`) the wrapper is a total no-op: inputs are
    not converted and outputs are not stamped.  Every declaration is validated
    **at decoration time**, so a malformed unit, marker or offset string fails
    at import even when switched off.

    Each keyword sets one axis for this function, and is named
    `<domain>_<axis>`.  Per axis, a marker's own override beats the keyword,
    which beats the built-in default.

    Usable bare (`@xra.enforce`) or called (`@xra.enforce(units_on_missing=...)`).

    Args:
        func: The function to decorate (when used bare); `None` when called with
            keyword arguments, in which case a parametrised decorator is returned.
        units_on_missing: Overrides `units` `on_missing`.
        units_on_inexact: Overrides `units` `on_inexact`.
        units_on_output: Overrides `units` `on_output`.
        schema_on_mismatch: Overrides `schema` `on_mismatch`.
        temporal_on_mismatch: Overrides `temporal` `on_mismatch`.
        temporal_on_uninferable: Overrides `temporal` `on_uninferable`.

    Returns:
        The wrapped function, or a decorator when `func` is `None`.

    Raises:
        ValueError: If a keyword or a declaration is malformed (at decoration).

    Example:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> from xarray_annotated import Dims, Unit
        >>> @xra.enforce
        ... def to_pa(
        ...     p: Annotated[xr.DataArray, Unit("Pa"), Dims("x")],
        ... ) -> Annotated[xr.DataArray, Unit("Pa")]:
        ...     return p
        >>> to_pa(xr.DataArray([1.0], dims="x", attrs={"units": "hPa"})).values
        array([100.])
    """
    return _enforce(
        func,
        domains=_SLOTS.keys(),
        units_on_missing=units_on_missing,
        units_on_inexact=units_on_inexact,
        units_on_output=units_on_output,
        schema_on_mismatch=schema_on_mismatch,
        temporal_on_mismatch=temporal_on_mismatch,
        temporal_on_uninferable=temporal_on_uninferable,
    )
