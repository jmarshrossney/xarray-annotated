"""Deprecated: `declare_schema`, superseded by `xarray_annotated.enforce`."""

import warnings
from collections.abc import Callable
from typing import Any

from ._config import OnMismatch


def declare_schema(
    func: Callable[..., Any] | None = None,
    *,
    on_mismatch: OnMismatch | None = None,
) -> Callable[..., Any]:
    """Apply a function's signature-declared structure at runtime.

    !!! warning "Deprecated"

        Use `xarray_annotated.enforce`, which applies every facet in the correct
        order.  `on_mismatch` becomes `schema_on_mismatch`.

    Behaves as `enforce` restricted to the schema facet.

    Args:
        func: The function to decorate (when used bare); `None` when parametrised.
        on_mismatch: Default severity for this function, overriding the policy.

    Returns:
        The wrapped function, or a decorator when `func` is `None`.
    """
    warnings.warn(
        "declare_schema is deprecated; use xarray_annotated.enforce (on_mismatch "
        "becomes schema_on_mismatch)",
        DeprecationWarning,
        stacklevel=2,
    )
    from .._enforce import _enforce  # deferred: _enforce imports this package

    return _enforce(func, domains={"schema"}, schema_on_mismatch=on_mismatch)
