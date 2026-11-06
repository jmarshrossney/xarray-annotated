"""Shared `typing.Annotated` introspection for xarray-annotated.

Pure `typing` machinery with no dependency on any domain's registry or config —
the "declaration side" common to every domain. An *extractor* turns one
`Annotated[...]` hint into a declaration payload;
`_walk_signature` drives that extractor over a whole function signature, handling
the `TypedDict` / dataclass / single-value return shapes once, for everyone.
"""

import dataclasses
import types
from collections.abc import Callable, Iterable
from typing import (
    Annotated,
    Any,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import xarray as xr


def _single_marker[M](metadata: Iterable[Any], marker_type: type[M]) -> M | None:
    """Return the one `marker_type` marker in `metadata`, or `None` if absent.

    A hint declares each facet at most once.  Two markers of one type are either
    contradictory (`Dims("x")` with `Dims("y")`) or redundant, so a second one is a
    malformed declaration rather than something to resolve by position.

    Raises:
        ValueError: If `metadata` holds more than one `marker_type` marker.
    """
    found = [m for m in metadata if isinstance(m, marker_type)]
    if len(found) > 1:
        raise ValueError(
            f"one Annotated hint declares {len(found)} {marker_type.__name__} "
            f"markers ({', '.join(map(repr, found))}); declare at most one"
        )
    return found[0] if found else None


def _is_dataarray_type(tp: Any) -> bool:
    """Return whether `tp` is `DataArray` (possibly wrapped or in a Union).

    Accepts a bare `DataArray` as well as `DataArray | None` /
    `Optional[DataArray]` (and any other union that includes `DataArray`), so the
    reader can refuse markers on such a union by name, rather than treating it as an
    unrelated type.  Anything else (scalars, `str`, `bool`, `Dataset`, …) is not a
    declaration-bearing type.

    Also accepts any *array-type wrapper* that exposes the underlying array class
    as an `array_type` attribute — notably jaxtyping's `Float[DataArray, "t x"]`,
    which occupies the base-type slot of an `Annotated` hint rather than competing
    with it.  This is a duck-typed check: jaxtyping is never imported, and any
    other library following the same convention composes for free.  See
    `docs/tools/jaxtyping.md`.

    Args:
        tp: A type annotation to inspect.

    Returns:
        `True` if `tp` is, wraps, or contains `DataArray`.
    """
    if tp is xr.DataArray:
        return True
    inner = getattr(tp, "array_type", None)
    if isinstance(inner, type) and issubclass(inner, xr.DataArray):
        return True
    if get_origin(tp) in (Union, types.UnionType):
        return any(_is_dataarray_type(arg) for arg in get_args(tp))
    return False


def _strip_return_hint(
    hint: Any, path: frozenset[Any] = frozenset()
) -> tuple[Any, bool]:
    """Strip `type` aliases, `Annotated` and `| None` off a return hint.

    An alias that refers to itself is left in place: it resolves to nothing.  The reader
    raises on one, except a generic one, as in `type G[T] = G[T] | None`, whose `G[int]`
    it does not enter.

    Returns:
        What remains, and whether a `| None` was stripped on the way to it.
    """
    origin = get_origin(hint)
    alias = next((a for a in (hint, origin) if isinstance(a, TypeAliasType)), None)
    if alias is not None:
        if alias in path:
            return hint, False
        return _strip_return_hint(alias.__value__, path | {alias})
    if origin is Annotated:
        return _strip_return_hint(get_args(hint)[0], path)
    if origin in (Union, types.UnionType):
        members = [arg for arg in get_args(hint) if arg is not types.NoneType]
        if len(members) == 1:
            return _strip_return_hint(members[0], path)[0], True
    return hint, False


def _record_return(hint: Any) -> tuple[Any, bool]:
    """Read a return hint's `TypedDict` or dataclass, and whether it may be `None`.

    Returns:
        The class, or `None` if the hint resolves to no such class, and whether the
        hint allows `None`.
    """
    base, optional = _strip_return_hint(hint)
    record = get_origin(base) or base
    is_record = is_typeddict(record) or dataclasses.is_dataclass(record)
    return (record if isinstance(record, type) and is_record else None), optional


def _walk_signature[T](
    func: object,
    extract: Callable[[Any], T | None],
) -> tuple[dict[str, T], dict[str, T] | T | None]:
    """Extract per-parameter and return declarations from a function's hints.

    Reads `get_type_hints(func, include_extras=True)` and maps each hint through
    `extract`, which returns a declaration payload for that hint or `None` when
    the hint carries no declaration.  `extract` also sees a whole return hint before
    any of its fields, so it can refuse what it finds there.

    Args:
        func: A callable whose parameters/return may carry `Annotated` metadata.
        extract: Turns one hint into a declaration payload, or `None`.  For every
            built-in facet at once this is `declaration_from_hint` (payload
            `Declared`).

    Returns:
        An `(inputs, output)` pair:

        * `inputs` — `dict[str, T]` mapping each declared parameter to its payload.
        * `output` — a `dict[str, T]` (per-field) if the return hint is a
            `TypedDict` or dataclass, read through a `type` alias, `Annotated`,
            `| None` and generic parameters; a single `T` if the return is one
            declared `Annotated[DataArray, ...]`; `None` otherwise.
    """
    qualname = getattr(func, "__qualname__", repr(func))

    def _extract(hint: Any, where: str) -> T | None:
        try:
            return extract(hint)
        except ValueError as exc:
            exc.add_note(f"in {qualname} {where}")
            raise

    hints = get_type_hints(func, include_extras=True)
    ret = hints.pop("return", None)

    inputs = {
        name: payload
        for name, hint in hints.items()
        if (payload := _extract(hint, f"input {name!r}")) is not None
    }

    output: dict[str, T] | T | None = _extract(ret, "output")
    record, _ = _record_return(ret)
    if is_typeddict(record):
        ret_hints = get_type_hints(record, include_extras=True)
        output = {
            name: payload
            for name, hint in ret_hints.items()
            if (payload := _extract(hint, f"output {name!r}")) is not None
        }
    elif record is not None:
        ret_hints = get_type_hints(record, include_extras=True)
        output = {
            f.name: payload
            for f in dataclasses.fields(record)
            if (payload := _extract(ret_hints.get(f.name), f"output {f.name!r}"))
            is not None
        }

    return inputs, output
