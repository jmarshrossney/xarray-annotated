"""Shared `typing.Annotated` introspection for xarray-annotated.

Pure `typing` machinery with no dependency on any domain's registry or config —
the "declaration side" common to every domain. A domain supplies an *extractor*
that turns one `Annotated[...]` hint into its own declaration payload (a unit
string, a list of schema markers, …); `walk_signature` drives that extractor over
a whole function signature, handling the `TypedDict` / dataclass / single-value
return shapes once, for everyone.
"""

import dataclasses
import types
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import xarray as xr

_T = TypeVar("_T")


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


def _is_dataarray_type(tp: Any) -> bool:
    """Return whether `tp` is `DataArray` (possibly wrapped in a Union).

    Accepts a bare `DataArray` as well as `DataArray | None` /
    `Optional[DataArray]` (and any other union that includes `DataArray`), so an
    optional DataArray parameter can still carry a declaration.  Anything else
    (scalars, `str`, `bool`, `Dataset`, …) is not a declaration-bearing type.

    Args:
        tp: A type annotation to inspect.

    Returns:
        `True` if `tp` is or contains `DataArray`.
    """
    if tp is xr.DataArray:
        return True
    if get_origin(tp) in (Union, types.UnionType):
        return any(_is_dataarray_type(arg) for arg in get_args(tp))
    return False


def walk_signature(
    func: object,
    extract: Callable[[Any], _T | None],
) -> tuple[dict[str, _T], dict[str, _T] | _T | None]:
    """Extract per-parameter and return declarations from a function's hints.

    Reads `get_type_hints(func, include_extras=True)` and maps each hint through
    `extract`, which returns a domain's declaration payload for that hint or
    `None` when the hint carries no declaration.  The `is not None` filter is the
    single "no declaration" rule shared by every domain.

    Args:
        func: A callable whose parameters/return may carry `Annotated` metadata.
        extract: Turns one hint into a declaration payload, or `None`.  For units
            this is `annotated_unit` (payload `str`); for schema `annotated_schema`
            (payload `list[marker]`).

    Returns:
        An `(inputs, output)` pair:

        * `inputs` — `dict[str, T]` mapping each declared parameter to its payload.
        * `output` — a `dict[str, T]` (per-field) if the return hint is a
            `TypedDict` or dataclass; a single `T` if the return is one declared
            `Annotated[DataArray, ...]`; `None` otherwise.
    """
    hints = get_type_hints(func, include_extras=True)
    ret = hints.pop("return", None)

    inputs = {
        name: payload
        for name, hint in hints.items()
        if (payload := extract(hint)) is not None
    }

    output: dict[str, _T] | _T | None
    if is_typeddict(ret):
        ret_hints = get_type_hints(ret, include_extras=True)
        output = {
            name: payload
            for name, hint in ret_hints.items()
            if (payload := extract(hint)) is not None
        }
    elif dataclasses.is_dataclass(ret):
        ret_hints = get_type_hints(ret, include_extras=True)
        output = {
            f.name: payload
            for f in dataclasses.fields(ret)
            if (payload := extract(ret_hints.get(f.name))) is not None
        }
    else:
        output = extract(ret)

    return inputs, output
