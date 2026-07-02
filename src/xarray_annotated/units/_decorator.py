"""Signature-driven unit declaration decorator.

`declare_units` wires the declaration side (`_annotations`) to the checking side
(`_check`): it reads a function's `Annotated[DataArray, "<unit>"]` hints once, at
decoration time, then on every call validates/converts the declared inputs and
stamps the declared outputs.

This is deliberately the *convenience* layer, and deliberately a plain function
decorator with no subclassing hierarchy. A tool that needs different behaviour
(build-time/static checks, custom value types, alternative policies) should build
its own consumer from the same public primitives — `units_from_signature`,
`check_units`, `assert_valid_unit` — which is exactly, and all, that this decorator
does. The two knobs most callers want are exposed as keyword arguments below.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Any

import xarray as xr

from ._annotations import units_from_signature
from ._check import assert_valid_unit, check_units
from ._config import (
    OnInexact,
    OnMissing,
    _validate_on_inexact,
    _validate_on_missing,
    get_policy,
)


def declare_units(
    func: Callable[..., Any] | None = None,
    *,
    on_missing: OnMissing | None = None,
    on_inexact: OnInexact | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared units at runtime.

    Reads the decorated function's own type annotations once, via
    `units_from_signature`: parameters annotated `Annotated[DataArray, "<unit>"]`
    declare input units, and a `TypedDict` return (or a bare
    `Annotated[DataArray, "<unit>"]` return) declares output units.  Those
    annotations are the single source of truth, so a unit is never written twice.

    On each call, under the active `Policy` (`get_policy`), the wrapper:

    1. validates/converts every declared `DataArray` input to its unit via
       `check_units`;
    2. runs the wrapped function;
    3. stamps each declared output `DataArray` with its unit (a `dict` return is
       stamped per key; a single `DataArray` return takes the bare declared unit).

    Only `DataArray` values are touched; other arguments and returns pass
    through unchanged.  When the policy is **disabled** (`enabled=False`) the
    wrapper is a total no-op: inputs are not converted and outputs are not
    stamped.

    Usable bare (`@declare_units`) or called
    (`@declare_units(on_missing="error")`).

    Every declared unit string is checked against the registry **at decoration
    time**, so a malformed or undefined unit fails fast at import — regardless
    of policy — rather than only when the function first runs.

    Args:
        func: The function to decorate (when used bare).  `None` when called
            with keyword arguments, in which case a parametrised decorator is
            returned.
        on_missing: Override the on-missing axis for the decorated function.
            When `None` (default) resolved per call from `get_policy`.
        on_inexact: Override the on-inexact axis for the decorated function.
            When `None` (default) resolved per call from `get_policy`.

    Returns:
        A wrapped function that validates inputs and stamps outputs according
        to the active policy.
    """
    if on_missing is not None:
        _validate_on_missing(on_missing)
    if on_inexact is not None:
        _validate_on_inexact(on_inexact)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        input_units, output_units = units_from_signature(fn)
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        # Fail fast on unparseable declarations (independent of policy).
        for name, unit in input_units.items():
            assert_valid_unit(unit, f"{qualname} input {name!r}")
        if isinstance(output_units, str):
            assert_valid_unit(output_units, f"{qualname} output")
        elif isinstance(output_units, dict):
            for name, unit in output_units.items():
                assert_valid_unit(unit, f"{qualname} output {name!r}")

        def _stamp(result: Any) -> Any:
            if isinstance(output_units, str):
                if isinstance(result, xr.DataArray):
                    result.attrs["units"] = output_units
            elif isinstance(output_units, dict):
                if isinstance(result, dict):
                    for name, value in result.items():
                        declared = output_units.get(name)
                        if declared is not None and isinstance(value, xr.DataArray):
                            value.attrs["units"] = declared
                elif dataclasses.is_dataclass(result):
                    for name, declared in output_units.items():
                        value = getattr(result, name)
                        if isinstance(value, xr.DataArray):
                            value.attrs["units"] = declared
            return result

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pol = get_policy()
            if not pol.enabled:
                # Master switch off: a total no-op (no conversion, no stamping).
                return fn(*args, **kwargs)
            if input_units:
                eff_missing = on_missing if on_missing is not None else pol.on_missing
                eff_inexact = on_inexact if on_inexact is not None else pol.on_inexact
                bound = sig.bind_partial(*args, **kwargs)
                for name, val in list(bound.arguments.items()):
                    declared = input_units.get(name)
                    if declared is not None and isinstance(val, xr.DataArray):
                        bound.arguments[name] = check_units(
                            val, declared, name, eff_missing, eff_inexact, qualname
                        )
                args, kwargs = bound.args, bound.kwargs
            return _stamp(fn(*args, **kwargs))

        return wrapper

    # @declare_units  →  func is the decorated function;
    # @declare_units(on_missing=...)  →  func is None, return the parametrised decorator.
    return decorate if func is None else decorate(func)
