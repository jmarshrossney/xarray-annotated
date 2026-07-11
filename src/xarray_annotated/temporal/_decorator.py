"""Signature-driven frequency-declaration decorator.

`declare_freq` wires the declaration side (`_annotations`) to the checking side
(`_check`): it reads a function's `Annotated[DataArray, Freq(...)]` hints once, at
decoration time, then on every call validates the declared inputs and outputs.

Like `schema.declare_schema` — whose scaffold this mirrors — it is the *convenience*
layer over the public primitives (`freq_from_signature`, `check_freq`,
`assert_valid_freq`), and it never mutates: inputs and outputs are *checked* and
passed through unchanged.
"""

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Any

import xarray as xr

from ._annotations import Freq, freq_from_signature
from ._check import assert_valid_freq, check_freq
from ._config import (
    OnMismatch,
    OnUninferable,
    _validate_on_mismatch,
    _validate_on_uninferable,
    get_policy,
)


def declare_freq(
    func: Callable[..., Any] | None = None,
    *,
    on_mismatch: OnMismatch | None = None,
    on_uninferable: OnUninferable | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared frequencies at runtime.

    Reads the decorated function's own type annotations once, via
    `freq_from_signature`: parameters annotated `Annotated[DataArray, Freq(...)]`
    declare an input's temporal frequency, and a `TypedDict`/dataclass return (or a
    bare `Annotated[DataArray, Freq(...)]` return) declares an output's.

    On each call, under the active `Policy` (`get_policy`), the wrapper validates every
    declared `DataArray` input via `check_freq`, runs the wrapped function, then
    validates every declared output the same way.  Nothing is mutated; a mismatch
    raises/warns/ignores per the effective `on_mismatch` (a marker's own override wins
    over this decorator's, which wins over the policy default), and an axis whose
    frequency cannot be inferred at all is reported under `on_uninferable`.  When the
    policy is **disabled** the wrapper is a total no-op.

    Usable bare (`@declare_freq`) or called (`@declare_freq(on_mismatch=...)`).  Every
    declaration is validated **at decoration time** (`assert_valid_freq`), so a
    malformed offset string fails fast at import.

    Args:
        func: The function to decorate (when used bare); `None` when parametrised.
        on_mismatch: Default mismatch severity for this function, overriding the policy
            default. `None` (default) resolves per call from `get_policy`.
        on_uninferable: Severity for an uninferable time axis, overriding the policy
            default. `None` (default) resolves per call from `get_policy`.

    Returns:
        A wrapped function that validates declared inputs and outputs.
    """
    if on_mismatch is not None:
        _validate_on_mismatch(on_mismatch)
    if on_uninferable is not None:
        _validate_on_uninferable(on_uninferable)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        input_freqs, output_freqs = freq_from_signature(fn)
        sig = inspect.signature(fn)
        qualname = getattr(fn, "__qualname__", repr(fn))

        # Fail fast on malformed declarations (independent of policy).
        for pname, marker in input_freqs.items():
            assert_valid_freq(marker, f"{qualname} input {pname!r}")
        _assert_output_declarations(output_freqs, qualname)

        def _check_output(result: Any, eff: OnMismatch, unin: OnUninferable) -> Any:
            if isinstance(output_freqs, Freq):
                if isinstance(result, xr.DataArray):
                    check_freq(result, output_freqs, "return", eff, unin, qualname)
            elif isinstance(output_freqs, dict):
                for key, marker in output_freqs.items():
                    if isinstance(result, dict):
                        value = result.get(key)
                    elif dataclasses.is_dataclass(result):
                        value = getattr(result, key, None)
                    else:
                        continue
                    if isinstance(value, xr.DataArray):
                        check_freq(value, marker, key, eff, unin, qualname)
            return result

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            pol = get_policy()
            if not pol.enabled:
                # Master switch off: a total no-op (no validation at all).
                return fn(*args, **kwargs)
            # Per-call defaults (a marker's own override still wins in check_freq).
            eff = on_mismatch if on_mismatch is not None else pol.on_mismatch
            unin = on_uninferable if on_uninferable is not None else pol.on_uninferable
            if input_freqs:
                bound = sig.bind_partial(*args, **kwargs)
                for pname, val in list(bound.arguments.items()):
                    marker = input_freqs.get(pname)
                    if marker is not None and isinstance(val, xr.DataArray):
                        check_freq(val, marker, pname, eff, unin, qualname)
            return _check_output(fn(*args, **kwargs), eff, unin)

        return wrapper

    return decorate if func is None else decorate(func)


def _assert_output_declarations(
    output_freqs: dict[str, Freq] | Freq | None,
    qualname: str,
) -> None:
    """Fail-fast-validate declared output markers at decoration time."""
    if isinstance(output_freqs, Freq):
        assert_valid_freq(output_freqs, f"{qualname} output")
    elif isinstance(output_freqs, dict):
        for key, marker in output_freqs.items():
            assert_valid_freq(marker, f"{qualname} output {key!r}")
