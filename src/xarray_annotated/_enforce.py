"""`enforce`: apply every facet declared on a signature, in one decorator.

The single cross-domain decorator.  It reads a function's
`Annotated[DataArray, ...]` hints once, at decoration time, and on every call
applies each declared facet in a fixed order:

1. **inputs** — units are converted (and restamped) *first*, so the structural
   and temporal checks then see the array the function body will see;
2. the function runs;
3. **outputs** — the declared unit is checked *first*, then structure and frequency,
   on the value the caller will receive.

That order is the reason there is one decorator rather than one per domain:
stackable per-domain decorators could be applied in any order, and only one
order was correct.

This is deliberately the *convenience* layer over public primitives — the
`declarations_from_signature` reader and the `check_units` /
`apply_output_units` / `check_schema` / `check_freq` checkers.  A tool needing
different behaviour should build its own consumer from those.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeAliasType, get_args, get_origin

import xarray as xr

from ._annotations import walk_signature
from ._config import resolve_enabled
from ._reader import Declared, declaration_from_hint, declarations_from_signature
from .schema import Coords, Dims, Dtype
from .schema._check import assert_valid_schema, check_schema
from .temporal import Freq
from .temporal import _config as _temporal
from .temporal._check import assert_valid_freq, check_freq
from .units import Unit
from .units._check import (
    apply_output_units,
    assert_valid_unit,
    check_units,
)

__all__ = ["enforce"]


def _assert_valid(declared: Declared, context: str) -> None:
    """Fail fast on a malformed declaration."""
    if declared.unit is not None:
        assert_valid_unit(declared.unit, context)
    for marker in declared.schema:
        assert_valid_schema(marker, context)
    if declared.freq is not None:
        assert_valid_freq(declared.freq, context)


def _markers_in(hint: Any, seen: set[TypeAliasType] | None = None) -> list[Any]:
    """Return every marker anywhere inside a hint, however deeply nested.

    Each `type` alias is entered once, since one may refer to itself.
    """
    seen = set() if seen is None else seen
    found = []
    for alias in (hint, get_origin(hint)):
        if isinstance(alias, TypeAliasType) and alias not in seen:
            seen.add(alias)
            found.extend(_markers_in(alias.__value__, seen))
    for arg in get_args(hint):
        if isinstance(arg, Unit | Dims | Dtype | Coords | Freq):
            found.append(arg)
        else:
            found.extend(_markers_in(arg, seen))
    return found


def _refuse_stray_markers(hint: Any) -> None:
    """Raise if a hint carries markers that declare nothing, so would go unchecked.

    Raises:
        ValueError: If `hint` holds a marker but `declaration_from_hint` reads no
            declaration from it.
    """
    if declaration_from_hint(hint) is None and (stray := _markers_in(hint)):
        raise ValueError(
            f"{hint!r} carries {stray!r}, which enforce cannot apply: only an "
            f"Annotated[DataArray, ...] parameter, return, or TypedDict or dataclass "
            f"field declares anything. For several outputs, return a TypedDict or "
            f"dataclass."
        )


def _with_fields(result: Any, changes: dict[str, xr.DataArray]) -> Any:
    """Write applied outputs back onto a dataclass return value.

    Written only once every field has passed, so a failed check leaves `result` as
    it was.  A frozen dataclass cannot take the write, so it is rebuilt instead,
    which runs its `__post_init__` again.

    Returns:
        `result`, or its frozen replacement.
    """
    try:
        for name, value in changes.items():
            setattr(result, name, value)
    except dataclasses.FrozenInstanceError:
        return dataclasses.replace(result, **changes)
    return result


def enforce(
    func: Callable[..., Any] | None = None,
    *,
    temporal_on_uninferable: _temporal.OnUninferable | None = None,
) -> Callable[..., Any]:
    """Apply every facet declared on a function's signature at runtime.

    Reads the decorated function's type hints once: parameters annotated
    `Annotated[DataArray, ...]` declare inputs, and a `TypedDict` / dataclass
    return (or a bare `Annotated[DataArray, ...]` return) declares outputs.  Any
    mix of `Unit`, `Dims` / `Coords` / `Dtype`, and `Freq` markers may sit on one
    hint.

    On each call the wrapper:

    1. converts every declared input to its unit (`check_units`), then validates
       its structure (`check_schema`) and frequency (`check_freq`) on the
       converted array;
    2. runs the wrapped function;
    3. checks each declared output against its unit (`apply_output_units`), then
       validates structure and frequency on the value returned to the caller.

    Only `DataArray` values are touched; anything else passes through.  When
    xarray-annotated is switched off (`xra.enabled(False)` or
    `XARRAY_ANNOTATED_ENABLED=false`) the wrapper is a total no-op: inputs are
    not converted and outputs are not checked.  Every declaration is validated
    **at decoration time**, so a malformed unit, marker or offset string fails
    at import even when switched off.

    Each keyword sets one axis for this function, and is named
    `<domain>_<axis>`.  Per axis, a marker's own override beats the keyword,
    which beats the built-in default.  Mismatch severity and unit conversion
    have no keyword: loosen a structural or frequency check, or allow a
    conversion, on the marker it applies to (`Dims(..., on_mismatch="warn")`,
    `Unit("Pa", convert=True)`).

    Usable bare (`@xra.enforce`) or called
    (`@xra.enforce(temporal_on_uninferable=...)`).
    On an `async def`, the checks run when the coroutine is awaited, and outputs
    are applied to the awaited result.

    Args:
        func: The function to decorate (when used bare); `None` when called with
            keyword arguments, in which case a parametrised decorator is returned.
        temporal_on_uninferable: Overrides `temporal` `on_uninferable`.

    Returns:
        The wrapped function, or a decorator when `func` is `None`.

    Raises:
        ValueError: If a keyword or a declaration is malformed, a `*args` or
            `**kwargs` parameter carries a declaration, or a hint carries a marker
            that cannot apply, such as in a tuple or on a `Dataset` (at decoration).

    Example:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> from xarray_annotated import Dims, Unit
        >>> @xra.enforce
        ... def to_pa(
        ...     p: Annotated[xr.DataArray, Unit("Pa", convert=True), Dims("x")],
        ... ) -> Annotated[xr.DataArray, Unit("Pa")]:
        ...     return p
        >>> to_pa(xr.DataArray([1.0], dims="x", attrs={"units": "hPa"})).values
        array([100.])
    """
    # Checked here rather than by the domain validator, which names the unprefixed
    # axis (`on_uninferable`) instead of the keyword the caller passed.
    allowed = get_args(_temporal.OnUninferable)
    if temporal_on_uninferable is not None and temporal_on_uninferable not in allowed:
        raise ValueError(
            f"Invalid temporal_on_uninferable {temporal_on_uninferable!r}. "
            f"Choose one of {sorted(allowed)}."
        )

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        walk_signature(fn, _refuse_stray_markers)
        inputs, output = declarations_from_signature(fn)
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        for name, decl in inputs.items():
            _assert_valid(decl, f"{qualname} input {name!r}")
            if sig.parameters[name].kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"{qualname} input {name!r}: enforce cannot check a declaration "
                    f"on a variadic parameter, since it binds a tuple or dict rather "
                    f"than a DataArray. Use named parameters, or call the checkers "
                    f"on each element in the function body."
                )
        if isinstance(output, Declared):
            _assert_valid(output, f"{qualname} output")
        elif isinstance(output, dict):
            for name, decl in output.items():
                _assert_valid(decl, f"{qualname} output {name!r}")

        def _apply_output(value: Any, decl: Declared, name: str) -> Any:
            # Units first, so structure and frequency are checked on what the
            # caller receives.  apply_output_units returns a *new* labelled array,
            # so its result is always used.
            if not isinstance(value, xr.DataArray):
                return value
            if decl.unit is not None:
                value = apply_output_units(value, decl.unit, name, qualname)
            if markers := decl.schema:
                check_schema(value, markers, name, qualname=qualname)
            if decl.freq is not None:
                check_freq(
                    value,
                    decl.freq,
                    name,
                    on_uninferable=temporal_on_uninferable,
                    qualname=qualname,
                )
            return value

        def _outputs(result: Any) -> Any:
            if isinstance(output, Declared):
                return _apply_output(result, output, "return")
            if isinstance(output, dict):
                if isinstance(result, dict):
                    for name, decl in output.items():
                        if name in result:
                            result[name] = _apply_output(result[name], decl, name)
                elif dataclasses.is_dataclass(result):
                    changes = {}
                    for name, decl in output.items():
                        value = getattr(result, name, None)
                        applied = _apply_output(value, decl, name)
                        if applied is not value:
                            changes[name] = applied
                    return _with_fields(result, changes)
            return result

        def _inputs(args: Any, kwargs: Any) -> tuple[Any, Any]:
            if not inputs:
                return args, kwargs
            bound = sig.bind_partial(*args, **kwargs)
            # bind_partial omits parameters left at their default; apply
            # them so a declared default is checked (and converted) too.
            bound.apply_defaults()
            for name, val in list(bound.arguments.items()):
                decl = inputs.get(name)
                if decl is None or not isinstance(val, xr.DataArray):
                    continue
                if decl.unit is not None:
                    val = check_units(val, decl.unit, name, qualname=qualname)
                    bound.arguments[name] = val
                if markers := decl.schema:
                    check_schema(val, markers, name, qualname=qualname)
                if decl.freq is not None:
                    check_freq(
                        val,
                        decl.freq,
                        name,
                        on_uninferable=temporal_on_uninferable,
                        qualname=qualname,
                    )
            return bound.args, bound.kwargs

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not resolve_enabled():
                # Switched off: a total no-op, conversion included.
                return fn(*args, **kwargs)
            args, kwargs = _inputs(args, kwargs)
            return _outputs(fn(*args, **kwargs))

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not resolve_enabled():
                return await fn(*args, **kwargs)
            args, kwargs = _inputs(args, kwargs)
            return _outputs(await fn(*args, **kwargs))

        chosen = async_wrapper if inspect.iscoroutinefunction(fn) else wrapper
        chosen.__annotations__ = dict(fn.__annotations__)
        return chosen

    return decorate if func is None else decorate(func)
