"""Deprecated: `declare_units`, superseded by `xarray_annotated.enforce`."""

import warnings
from collections.abc import Callable
from typing import Any

from ._config import (
    OnInexact,
    OnMissing,
    OnOutput,
    _validate_on_inexact,
    _validate_on_missing,
    _validate_on_output,
)


def declare_units(
    func: Callable[..., Any] | None = None,
    *,
    on_missing: OnMissing | None = None,
    on_inexact: OnInexact | None = None,
    on_output: OnOutput | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared units at runtime.

    !!! warning "Deprecated"

        Use `xarray_annotated.enforce`, which applies every facet in the correct
        order.  `on_missing` / `on_inexact` / `on_output` become
        `units_on_missing` / `units_on_inexact` / `units_on_output`.

    Behaves as `enforce` restricted to the units facet.

    Args:
        func: The function to decorate (when used bare); `None` when parametrised.
        on_missing: Overrides the on-missing axis for this function.
        on_inexact: Overrides the on-inexact axis for this function.
        on_output: Overrides the on-output axis for this function.

    Returns:
        The wrapped function, or a decorator when `func` is `None`.
    """
    warnings.warn(
        "declare_units is deprecated; use xarray_annotated.enforce (keywords "
        "become units_on_missing / units_on_inexact / units_on_output)",
        DeprecationWarning,
        stacklevel=2,
    )
    # Validated here so errors name the keywords this caller used.
    for value, validate in (
        (on_missing, _validate_on_missing),
        (on_inexact, _validate_on_inexact),
        (on_output, _validate_on_output),
    ):
        if value is not None:
            validate(value)
    from .._enforce import _enforce  # deferred: _enforce imports this package

    return _enforce(
        func,
        domains={"units"},
        units_on_missing=on_missing,
        units_on_inexact=on_inexact,
        units_on_output=on_output,
    )
