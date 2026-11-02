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
layer over public primitives — the `*_from_signature` readers and the
`check_units` / `apply_output_units` / `check_schema` / `check_freq` checkers.  A
tool needing different behaviour should build its own consumer from those.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Collection
from typing import Any, NamedTuple

import xarray as xr

from ._annotations import walk_signature
from ._config import resolve_enabled
from .schema import _config as _schema
from .schema._annotations import SchemaMarker, annotated_schema
from .schema._check import assert_valid_schema, check_schema
from .temporal import _config as _temporal
from .temporal._annotations import Freq, annotated_freq
from .temporal._check import assert_valid_freq, check_freq
from .units import _config as _units
from .units._annotations import annotated_unit
from .units._check import apply_output_units, assert_valid_unit, check_units

__all__ = ["enforce"]

_DOMAINS = frozenset({"units", "schema", "temporal"})


class _Facets(NamedTuple):
    """Every facet one hint declares, in each checker's own payload shape.

    Not `Declared`: that folds schema markers into one slot per marker type, and
    `check_schema` must see every marker on the hint.
    """

    unit: str | None
    schema: list[SchemaMarker] | None
    freq: Freq | None


def _extractor(domains: Collection[str]) -> Callable[[Any], _Facets | None]:
    """Build a `walk_signature` extractor reading only `domains`' facets."""

    def extract(hint: Any) -> _Facets | None:
        facets = _Facets(
            unit=annotated_unit(hint) if "units" in domains else None,
            schema=annotated_schema(hint) if "schema" in domains else None,
            freq=annotated_freq(hint) if "temporal" in domains else None,
        )
        return None if facets == (None, None, None) else facets

    return extract


def _assert_valid(facets: _Facets, context: str) -> None:
    """Fail fast on a malformed declaration (independent of policy)."""
    if facets.unit is not None:
        assert_valid_unit(facets.unit, context)
    for marker in facets.schema or ():
        assert_valid_schema(marker, context)
    if facets.freq is not None:
        assert_valid_freq(facets.freq, context)


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
    for value, validate in (
        (units_on_missing, _units._validate_on_missing),
        (units_on_inexact, _units._validate_on_inexact),
        (units_on_output, _units._validate_on_output),
        (schema_on_mismatch, _schema._validate_on_mismatch),
        (temporal_on_mismatch, _temporal._validate_on_mismatch),
        (temporal_on_uninferable, _temporal._validate_on_uninferable),
    ):
        if value is not None:
            validate(value)

    extract = _extractor(domains)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        inputs, output = walk_signature(fn, extract)
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        for name, facets in inputs.items():
            _assert_valid(facets, f"{qualname} input {name!r}")
        if isinstance(output, _Facets):
            _assert_valid(output, f"{qualname} output")
        elif isinstance(output, dict):
            for name, facets in output.items():
                _assert_valid(facets, f"{qualname} output {name!r}")

        def _apply_output(value: Any, facets: _Facets, name: str) -> Any:
            # Units first, so structure and frequency are checked on what the
            # caller receives.  apply_output_units may return a *new* array (a
            # converted quantified one), so its result is always used.
            if not isinstance(value, xr.DataArray):
                return value
            if facets.unit is not None:
                on_output = units_on_output or _units.get_policy().on_output
                value = apply_output_units(
                    value, facets.unit, name, on_output, qualname
                )
            if facets.schema is not None:
                on_mismatch = schema_on_mismatch or _schema.get_policy().on_mismatch
                check_schema(value, facets.schema, name, on_mismatch, qualname)
            if facets.freq is not None:
                pol = _temporal.get_policy()
                check_freq(
                    value,
                    facets.freq,
                    name,
                    temporal_on_mismatch or pol.on_mismatch,
                    temporal_on_uninferable or pol.on_uninferable,
                    qualname,
                )
            return value

        def _outputs(result: Any) -> Any:
            if isinstance(output, _Facets):
                return _apply_output(result, output, "return")
            if isinstance(output, dict):
                if isinstance(result, dict):
                    for name, facets in output.items():
                        if name in result:
                            result[name] = _apply_output(result[name], facets, name)
                elif dataclasses.is_dataclass(result):
                    for name, facets in output.items():
                        value = getattr(result, name, None)
                        applied = _apply_output(value, facets, name)
                        if applied is not value:
                            _set_field(result, name, applied, qualname)
            return result

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not resolve_enabled():
                # Master switch off: a total no-op.
                return fn(*args, **kwargs)
            if inputs:
                bound = sig.bind_partial(*args, **kwargs)
                # bind_partial omits parameters left at their default; apply
                # them so a declared default is checked (and converted) too.
                bound.apply_defaults()
                for name, val in list(bound.arguments.items()):
                    facets = inputs.get(name)
                    if facets is None or not isinstance(val, xr.DataArray):
                        continue
                    if facets.unit is not None:
                        pol = _units.get_policy()
                        val = check_units(
                            val,
                            facets.unit,
                            name,
                            units_on_missing or pol.on_missing,
                            units_on_inexact or pol.on_inexact,
                            qualname,
                        )
                        bound.arguments[name] = val
                    if facets.schema is not None:
                        check_schema(
                            val,
                            facets.schema,
                            name,
                            schema_on_mismatch or _schema.get_policy().on_mismatch,
                            qualname,
                        )
                    if facets.freq is not None:
                        pol = _temporal.get_policy()
                        check_freq(
                            val,
                            facets.freq,
                            name,
                            temporal_on_mismatch or pol.on_mismatch,
                            temporal_on_uninferable or pol.on_uninferable,
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

    Only `DataArray` values are touched; anything else passes through.  When
    the package-wide master switch is off (`enabled=False`) the wrapper is a
    total no-op.  Every declaration is validated **at decoration time**, so a
    malformed unit, marker or offset string fails at import regardless of
    policy.

    Each keyword overrides one policy axis for this function, and is named after
    its domain's axis (and environment variable).  Per axis, a marker's own
    override beats the keyword, which beats the domain's `get_policy()`.

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
        >>> from xarray_annotated.schema import Dims
        >>> from xarray_annotated.units import Unit
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
        domains=_DOMAINS,
        units_on_missing=units_on_missing,
        units_on_inexact=units_on_inexact,
        units_on_output=units_on_output,
        schema_on_mismatch=schema_on_mismatch,
        temporal_on_mismatch=temporal_on_mismatch,
        temporal_on_uninferable=temporal_on_uninferable,
    )
