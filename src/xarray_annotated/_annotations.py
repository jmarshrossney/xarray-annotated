"""Shared `typing.Annotated` introspection for xarray-annotated.

Pure `typing` machinery with no dependency on any domain's registry or config —
the "declaration side" common to every domain. An *extractor* turns one
`Annotated[...]` hint into a declaration payload;
`walk_signature` drives that extractor over a whole function signature, handling
the `TypedDict` / dataclass / single-value return shapes once, for everyone.
"""

import dataclasses
import types
from collections.abc import Callable, Iterable
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import xarray as xr


def unwrap_annotated(hint: Any) -> Any:
    """Return the underlying type of an `Annotated` hint, else the hint itself.

    `Annotated[DataArray, ...]` → `DataArray`; a non-`Annotated` hint is returned
    unchanged.  Lets type comparisons (e.g. `t is DataArray`) see through the
    metadata that declarations attach to parameters.

    Args:
        hint: A type hint, possibly `Annotated`.

    Returns:
        The base type if `hint` is `Annotated`; otherwise `hint` unchanged.
    """
    return get_args(hint)[0] if get_origin(hint) is Annotated else hint


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
    `Optional[DataArray]` (and any other union that includes `DataArray`), so an
    optional DataArray parameter can still carry a declaration.  Anything else
    (scalars, `str`, `bool`, `Dataset`, …) is not a declaration-bearing type.

    Also accepts any *array-type wrapper* that exposes the underlying array class
    as an `array_type` attribute — notably jaxtyping's `Float[DataArray, "t x"]`,
    which occupies the base-type slot of an `Annotated` hint rather than competing
    with it.  This is a duck-typed check: jaxtyping is never imported, and any
    other library following the same convention composes for free.  See
    `docs/guides/jaxtyping.md`.

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


def walk_signature[T](
    func: object,
    extract: Callable[[Any], T | None],
) -> tuple[dict[str, T], dict[str, T] | T | None]:
    """Extract per-parameter and return declarations from a function's hints.

    Reads `get_type_hints(func, include_extras=True)` and maps each hint through
    `extract`, which returns a declaration payload for that hint or `None` when
    the hint carries no declaration.

    Args:
        func: A callable whose parameters/return may carry `Annotated` metadata.
        extract: Turns one hint into a declaration payload, or `None`.  For every
            built-in facet at once this is `declaration_from_hint` (payload
            `Declared`); a third-party facet supplies its own.

    Returns:
        An `(inputs, output)` pair:

        * `inputs` — `dict[str, T]` mapping each declared parameter to its payload.
        * `output` — a `dict[str, T]` (per-field) if the return hint is a
            `TypedDict` or dataclass; a single `T` if the return is one declared
            `Annotated[DataArray, ...]`; `None` otherwise.
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

    output: dict[str, T] | T | None
    if is_typeddict(ret):
        ret_hints = get_type_hints(ret, include_extras=True)
        output = {
            name: payload
            for name, hint in ret_hints.items()
            if (payload := _extract(hint, f"output {name!r}")) is not None
        }
    elif dataclasses.is_dataclass(ret):
        ret_hints = get_type_hints(ret, include_extras=True)
        output = {
            f.name: payload
            for f in dataclasses.fields(ret)
            if (payload := _extract(ret_hints.get(f.name), f"output {f.name!r}"))
            is not None
        }
    else:
        output = _extract(ret, "output")

    return inputs, output
