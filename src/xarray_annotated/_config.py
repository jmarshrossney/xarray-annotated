"""The package-wide master switch, the only global setting.

Behaviour (`on_missing`, `on_mismatch`, ...) is declared on markers or passed to
`enforce` / the `check_*` primitives, never configured globally, so that how a
function validates can be read off its signature.  What remains global is
whether xarray-annotated does anything at all.

The switch resolves on every call as: an `enabled(...)` block, then the
`XARRAY_ANNOTATED_ENABLED` environment variable, then on.  Code beats the
environment in both directions, so `enabled(True)` restores checking in a process
started with the variable set to false.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = ["ENABLED_ENV_VAR", "enabled", "resolve_enabled"]

ENABLED_ENV_VAR = "XARRAY_ANNOTATED_ENABLED"

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSEY: frozenset[str] = frozenset({"0", "false", "no", "off"})

_override: ContextVar[bool | None] = ContextVar(
    "xarray_annotated_enabled", default=None
)


def _parse_enabled_env(value: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSEY:
        return False
    raise ValueError(
        f"Invalid {ENABLED_ENV_VAR} value {value!r}. "
        f"Use one of {sorted(_TRUTHY | _FALSEY)}."
    )


def resolve_enabled() -> bool:
    """Return whether xarray-annotated is active for this call.

    Returns:
        The innermost `enabled(...)` value if inside one, else the parsed
        `XARRAY_ANNOTATED_ENABLED`, else `True`.

    Raises:
        ValueError: If the environment variable holds an unrecognised value.
    """
    override = _override.get()
    if override is not None:
        return override
    env = os.environ.get(ENABLED_ENV_VAR)
    if env:
        return _parse_enabled_env(env)
    return True


@contextmanager
def enabled(value: bool) -> Generator[None]:
    """Switch xarray-annotated on or off for a block.

    Off means the package does nothing: no checks, no input conversion and no
    output stamping, so a body declaring `"Pa"` can receive `"hPa"`.
    Declarations are still validated when a function is decorated.

    The block beats `XARRAY_ANNOTATED_ENABLED` in both directions, and is
    scoped to the current thread or async task.

    Args:
        value: `False` to switch off, `True` to switch on.

    Examples:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> @xra.enforce
        ... def to_pa(p: Annotated[xr.DataArray, xra.Unit("Pa")]):
        ...     return p
        >>> hpa = xr.DataArray([1.0], attrs={"units": "hPa"})
        >>> to_pa(hpa).values
        array([100.])
        >>> with xra.enabled(False):
        ...     to_pa(hpa).values
        array([1.])
    """
    token = _override.set(bool(value))
    try:
        yield
    finally:
        _override.reset(token)
