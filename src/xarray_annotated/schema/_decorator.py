"""Signature-driven structural-declaration decorator.

`declare_schema` wires the declaration side (`_annotations`) to the checking side
(`_check`): it reads a function's `Annotated[DataArray, <markers>]` hints once, at
decoration time, then on every call validates the declared inputs and outputs.

Like `units.declare_units` this is deliberately the *convenience* layer — a plain
function decorator over the public primitives (`schema_from_signature`,
`check_schema`, `assert_valid_schema`) with no subclassing hierarchy.  Unlike it,
schema validation never mutates: inputs and outputs are *checked* and passed
through unchanged.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Any

import xarray as xr

from ._annotations import SchemaMarker, schema_from_signature
from ._check import assert_valid_schema, check_schema
from ._config import OnMismatch, _validate_on_mismatch, get_policy


def declare_schema(
    func: Callable[..., Any] | None = None,
    *,
    on_mismatch: OnMismatch | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared structure at runtime.

    Reads the decorated function's own type annotations once, via
    `schema_from_signature`: parameters annotated `Annotated[DataArray, <markers>]`
    declare input structure, and a `TypedDict`/dataclass return (or a bare
    `Annotated[DataArray, <markers>]` return) declares output structure.

    On each call, under the active `Policy` (`get_policy`), the wrapper validates
    every declared `DataArray` input via `check_schema`, runs the wrapped function,
    then validates every declared output the same way.  Nothing is mutated; a
    mismatch raises/warns/ignores per the effective `on_mismatch` (a marker's own
    override wins over this decorator's `on_mismatch`, which wins over the policy
    default).  When the policy is **disabled** the wrapper is a total no-op.

    Usable bare (`@declare_schema`) or called (`@declare_schema(on_mismatch=...)`).
    Every marker declaration is validated **at decoration time** (`assert_valid_schema`),
    so a malformed declaration fails fast at import.

    Args:
        func: The function to decorate (when used bare); `None` when parametrised.
        on_mismatch: Default severity for this function, overriding the policy
            default. `None` (default) resolves per call from `get_policy`.

    Returns:
        A wrapped function that validates declared inputs and outputs.
    """
    if on_mismatch is not None:
        _validate_on_mismatch(on_mismatch)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        input_markers, output_markers = schema_from_signature(fn)
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        # Fail fast on malformed declarations (independent of policy).
        for pname, markers in input_markers.items():
            for marker in markers:
                assert_valid_schema(marker, f"{qualname} input {pname!r}")
        _assert_output_declarations(output_markers, qualname)

        def _check_output(result: Any, eff: OnMismatch) -> Any:
            if isinstance(output_markers, list):
                if isinstance(result, xr.DataArray):
                    check_schema(result, output_markers, "return", eff, qualname)
            elif isinstance(output_markers, dict):
                for key, markers in output_markers.items():
                    if isinstance(result, dict):
                        value = result.get(key)
                    elif dataclasses.is_dataclass(result):
                        value = getattr(result, key, None)
                    else:
                        continue
                    if isinstance(value, xr.DataArray):
                        check_schema(value, markers, key, eff, qualname)
            return result

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pol = get_policy()
            if not pol.enabled:
                # Master switch off: a total no-op (no validation at all).
                return fn(*args, **kwargs)
            # Per-call default severity (a marker's own override still wins in check_schema).
            eff = on_mismatch if on_mismatch is not None else pol.on_mismatch
            if input_markers:
                bound = sig.bind_partial(*args, **kwargs)
                for pname, val in list(bound.arguments.items()):
                    markers = input_markers.get(pname)
                    if markers is not None and isinstance(val, xr.DataArray):
                        check_schema(val, markers, pname, eff, qualname)
            return _check_output(fn(*args, **kwargs), eff)

        return wrapper

    return decorate if func is None else decorate(func)


def _assert_output_declarations(
    output_markers: dict[str, list[SchemaMarker]] | list[SchemaMarker] | None,
    qualname: str,
) -> None:
    """Fail-fast-validate declared output markers at decoration time."""
    if isinstance(output_markers, list):
        for marker in output_markers:
            assert_valid_schema(marker, f"{qualname} output")
    elif isinstance(output_markers, dict):
        for key, markers in output_markers.items():
            for marker in markers:
                assert_valid_schema(marker, f"{qualname} output {key!r}")
