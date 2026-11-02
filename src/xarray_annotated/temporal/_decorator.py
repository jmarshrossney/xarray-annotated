"""Deprecated: `declare_freq`, superseded by `xarray_annotated.enforce`."""

import warnings
from collections.abc import Callable
from typing import Any

from ._config import (
    OnMismatch,
    OnUninferable,
    _validate_on_mismatch,
    _validate_on_uninferable,
)


def declare_freq(
    func: Callable[..., Any] | None = None,
    *,
    on_mismatch: OnMismatch | None = None,
    on_uninferable: OnUninferable | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared frequencies at runtime.

    !!! warning "Deprecated"

        Use `xarray_annotated.enforce`, which applies every facet in the correct
        order.  `on_mismatch` / `on_uninferable` become `temporal_on_mismatch` /
        `temporal_on_uninferable`.

    Behaves as `enforce` restricted to the temporal facet.

    Args:
        func: The function to decorate (when used bare); `None` when parametrised.
        on_mismatch: Default mismatch severity for this function.
        on_uninferable: Severity for an uninferable time axis for this function.

    Returns:
        The wrapped function, or a decorator when `func` is `None`.
    """
    warnings.warn(
        "declare_freq is deprecated; use xarray_annotated.enforce (keywords become "
        "temporal_on_mismatch / temporal_on_uninferable)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Validated here so errors name the keywords this caller used.
    if on_mismatch is not None:
        _validate_on_mismatch(on_mismatch)
    if on_uninferable is not None:
        _validate_on_uninferable(on_uninferable)
    from .._enforce import _enforce  # deferred: _enforce imports this package

    return _enforce(
        func,
        domains={"temporal"},
        temporal_on_mismatch=on_mismatch,
        temporal_on_uninferable=on_uninferable,
    )
