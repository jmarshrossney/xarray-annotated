"""`enforce`: apply every facet declared on a signature, in one decorator.

The single cross-domain decorator.  It reads a function's
`Annotated[DataArray, ...]` hints once, at decoration time, and on every call
applies each declared facet in a fixed order:

1. **inputs** — units are checked *first*, then structure and frequency;
2. the function runs;
3. **outputs** — the declared unit is checked *first*, then structure and frequency.

These are checks only: the body receives what the caller passed, and the caller
receives what the body returned.

That order is the reason there is one decorator rather than one per domain:
stackable per-domain decorators could be applied in any order, and only one
order was correct.

This is deliberately the *convenience* layer over public primitives — the
`declarations_from_signature` reader and the `check_declared` /
`check_output_units` / `check_schema` / `check_freq` checkers.  A tool needing
different behaviour should build its own consumer from those.
"""

import contextlib
import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import (
    Any,
    TypeAliasType,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import xarray as xr

from ._annotations import _record_return, _walk_signature
from ._config import resolve_enabled
from ._contract import _assert_valid, _refuse_other_value, check_declared
from ._reader import Declared, declaration_from_hint, declarations_from_signature
from .schema import Coords, Dims, Dtype
from .schema._check import check_schema
from .temporal import Freq
from .temporal._check import check_freq
from .units import Unit
from .units._check import check_output_units

__all__ = ["enforce"]


def _nested_hints(hint: Any, seen: set[Any]) -> list[Any]:
    """Return what sits directly inside a hint.

    That is its type arguments, with a `Callable`'s parameter list flattened, since
    `get_args` returns that as a plain list, and the value of a `type` alias.  Each
    alias is entered once, since one may refer to itself.
    """
    nested = []
    for alias in (hint, get_origin(hint)):
        if isinstance(alias, TypeAliasType) and alias not in seen:
            seen.add(alias)
            nested.append(alias.__value__)
    for arg in get_args(hint):
        if isinstance(arg, list):
            nested.extend(arg)
        else:
            nested.append(arg)
    return nested


def _markers_in(hint: Any, seen: set[Any] | None = None) -> list[Any]:
    """Return every marker anywhere inside a hint, however deeply nested."""
    seen = set() if seen is None else seen
    found = []
    for nested in _nested_hints(hint, seen):
        if isinstance(nested, Unit | Dims | Dtype | Coords | Freq):
            found.append(nested)
        else:
            found.extend(_markers_in(nested, seen))
    return found


def _is_record(tp: Any) -> bool:
    """Return whether `tp` is a `TypedDict`, dataclass or `NamedTuple` class."""
    return isinstance(tp, type) and (
        is_typeddict(tp)
        or dataclasses.is_dataclass(tp)
        or (issubclass(tp, tuple) and hasattr(tp, "_fields"))
    )


def _record_fields_in(hint: Any, seen: set[Any] | None = None) -> list[Any]:
    """Return the field hints of every record class inside a hint.

    Each class is entered once, since one may refer to itself.  A class with an
    unresolvable forward reference is skipped: one that declares fields through it
    raises when read, and any other has nothing to do with this package.
    """
    seen = set() if seen is None else seen
    found = []
    record = get_origin(hint) or hint
    if _is_record(record) and record not in seen:
        seen.add(record)
        with contextlib.suppress(NameError):
            found.extend(get_type_hints(record, include_extras=True).values())
        for field in list(found):
            found.extend(_record_fields_in(field, seen))
    for nested in _nested_hints(hint, seen):
        found.extend(_record_fields_in(nested, seen))
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


def _refuse_unread_fields(hint: Any) -> None:
    """Raise if a hint holds a record whose declared fields go unread.

    Raises:
        ValueError: If a class inside `hint` has fields carrying markers: `hint` is
            a parameter, a return that is not read field by field, or a field of one
            that is, and fields are read one level deep.
    """
    stray = [m for field in _record_fields_in(hint) for m in _markers_in(field)]
    if stray:
        raise ValueError(
            f"{hint!r} holds fields carrying {stray!r}, which enforce cannot apply: "
            f"only one TypedDict or dataclass return, optionally `| None`, is read "
            f"field by field, and not any class nested in its fields or any parameter. "
            f"Declare a parameter's arrays as parameters of their own."
        )


def enforce(func: Callable[..., Any]) -> Callable[..., Any]:
    """Apply every facet declared on a function's signature at runtime.

    Reads the decorated function's type hints once: parameters annotated
    `Annotated[DataArray, ...]` declare inputs, and a `TypedDict` / dataclass
    return (or a bare `Annotated[DataArray, ...]` return) declares outputs.  Any
    mix of `Unit`, `Dims` / `Coords` / `Dtype`, and `Freq` markers may sit on one
    hint, with at most one of each kind, so write `Coords("time", "lat")`.

    A `TypedDict` or dataclass return may also be written `R | None`, `R[int]` or
    `Annotated[R, "description"]`, or through a `type` alias.  The caller receives a
    shallow copy of it, which shares its arrays, and `__post_init__` does not run again.

    On each call the wrapper:

    1. checks every declared input against its unit (`check_units`), then validates
       its structure (`check_schema`) and frequency (`check_freq`);
    2. runs the wrapped function;
    3. checks each declared output against its unit (`check_output_units`), then
       validates structure and frequency on the value returned to the caller.

    A value that is not a `DataArray` raises, unless its hint allows it, as `| None`
    allows `None` and `| float` allows anything else.  When xarray-annotated is switched
    off (`xra.enabled(False)` or
    `XARRAY_ANNOTATED_ENABLED=false`) the wrapper is a total no-op.  Every declaration
    is validated **at decoration time**, so a malformed unit, marker or offset string
    fails at import even when switched off.

    There are no options: every failed check raises, and nothing is changed: the body
    receives the objects the caller passed, quantified or not, and the caller receives
    the object the body returned.  Applied bare, as `@xra.enforce`.

    On an `async def`, the checks run when the coroutine is awaited, and outputs
    are applied to the awaited result.

    Args:
        func: The function to decorate.

    Returns:
        The wrapped function.

    Raises:
        ValueError: If a declaration is malformed, a `*args` or `**kwargs` parameter
            carries a declaration, a hint carries two markers of one kind, or a hint
            carries a marker that cannot apply, such as in a tuple, on a `Dataset`, on a
            field of a `NamedTuple` return or of a class nested in a field, or on a
            field of a parameter (at decoration).
        TypeError: If a declared value is not a `DataArray` and its hint does not allow
            it, or a `TypedDict` return lacks a required declared field (when called).

    Example:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> from xarray_annotated import Dims, Unit
        >>> @xra.enforce
        ... def double(
        ...     p: Annotated[xr.DataArray, Unit("Pa"), Dims("x")],
        ... ) -> Annotated[xr.DataArray, Unit("Pa")]:
        ...     return p * 2
        >>> double(xr.DataArray([1.0], dims="x", attrs={"units": "Pa"})).values
        array([2.])
    """
    _walk_signature(func, _refuse_stray_markers)
    inputs, output = declarations_from_signature(func)
    hints = get_type_hints(func, include_extras=True)
    return_hint = hints.pop("return", None)
    for param_hint in hints.values():
        _refuse_unread_fields(param_hint)
    record, record_optional = _record_return(return_hint)
    if record is None:
        _refuse_unread_fields(return_hint)
    else:
        for field in get_type_hints(record, include_extras=True).values():
            _refuse_unread_fields(field)
    sig = inspect.signature(func)
    qualname = getattr(func, "__qualname__", repr(func))

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

    def _check_output(value: Any, decl: Declared, name: str) -> None:
        if not isinstance(value, xr.DataArray):
            _refuse_other_value(value, decl, f"[{qualname}] output {name!r}")
            return
        if decl.unit is not None:
            check_output_units(value, decl.unit, name, qualname)
        if markers := decl.schema:
            check_schema(value, markers, name, qualname=qualname)
        if decl.freq is not None:
            check_freq(value, decl.freq, name, qualname=qualname)

    def _check_record(result: Any, cls: Any, fields: dict[str, Declared]) -> None:
        is_dict = is_typeddict(cls)
        if not isinstance(result, dict if is_dict else cls):
            raise TypeError(
                f"[{qualname}] output: declared {cls.__name__}, got "
                f"{type(result).__name__}."
            )
        for name, decl in fields.items():
            if is_dict and name not in result:
                if name in cls.__required_keys__:
                    raise TypeError(
                        f"[{qualname}] output {name!r}: a required field of "
                        f"{cls.__name__} is missing."
                    )
                continue
            value = result[name] if is_dict else getattr(result, name, None)
            _check_output(value, decl, name)

    def _check_outputs(result: Any) -> None:
        if isinstance(output, Declared):
            _check_output(result, output, "return")
        elif output and not (result is None and record_optional):
            _check_record(result, record, output)

    def _check_inputs(args: Any, kwargs: Any) -> None:
        if not inputs:
            return
        bound = sig.bind_partial(*args, **kwargs)
        # bind_partial omits parameters left at their default; apply
        # them so a declared default is checked too.
        bound.apply_defaults()
        for name, val in bound.arguments.items():
            decl = inputs.get(name)
            if decl is not None:
                check_declared(val, decl, name, qualname)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not resolve_enabled():
            return func(*args, **kwargs)
        _check_inputs(args, kwargs)
        result = func(*args, **kwargs)
        _check_outputs(result)
        return result

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        if not resolve_enabled():
            return await func(*args, **kwargs)
        _check_inputs(args, kwargs)
        result = await func(*args, **kwargs)
        _check_outputs(result)
        return result

    chosen = async_wrapper if inspect.iscoroutinefunction(func) else wrapper
    chosen.__annotations__ = dict(func.__annotations__)
    return chosen
