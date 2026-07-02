"""Process-wide validation policy for the structural (schema) domain.

Simpler than the units policy because structural validation never converts — it
only asserts.  Two axes:

- `enabled` — the **package-wide** master switch (shared with every domain via
  `xarray_annotated._config`); `False` makes all validation a no-op.
- `on_mismatch` — what to do when a DataArray's structure does not match a
  declaration (`"error"` / `"warn"` / `"ignore"`).  This is the *default*; an
  individual marker may override it (e.g. error on wrong dims but only warn on
  wrong dtype).

Each axis resolves independently: environment variable, then a process override
set via `set_policy` / the `policy` context manager, then the module default. The
shared `_resolve` helper encodes that precedence.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from .._config import (
    DEFAULT_ENABLED,
    ENABLED_ENV_VAR,
    _resolve,
    _Unset,
    _UNSET,
    get_enabled_override,
    resolve_enabled,
    set_enabled_override,
)

# `ENABLED_ENV_VAR` is re-exported (the master switch is shared but callers still
# reach it as `schema._config.ENABLED_ENV_VAR`).
__all__ = [
    "ENABLED_ENV_VAR",
    "OnMismatch",
    "Policy",
    "get_policy",
    "policy",
    "set_policy",
]

OnMismatch = Literal["error", "warn", "ignore"]

_VALID_ON_MISMATCH: frozenset[str] = frozenset({"error", "warn", "ignore"})

#: A structural mismatch is usually a genuine bug (wrong array wired in), so the
#: default is to raise.  Soften it per-marker (`Dims(..., on_mismatch="warn")`) or
#: package-wide via the policy.
DEFAULT_ON_MISMATCH: OnMismatch = "error"

ON_MISMATCH_ENV_VAR = "XARRAY_ANNOTATED_SCHEMA_ON_MISMATCH"

_process_on_mismatch: OnMismatch | None = None


@dataclass(frozen=True, slots=True)
class Policy:
    """The resolved schema-validation policy — the two axes as a single value.

    Returned by `get_policy`.  Build overrides via `set_policy` or the `policy`
    context manager rather than constructing this directly.

    Attributes:
        enabled: Master switch; `False` makes all validation a no-op.
        on_mismatch: Default behaviour on a structural mismatch
            (`"error"`, `"warn"`, or `"ignore"`); a marker may override it.
    """

    enabled: bool = DEFAULT_ENABLED
    on_mismatch: OnMismatch = DEFAULT_ON_MISMATCH


def _validate_on_mismatch(value: str) -> OnMismatch:
    if value not in _VALID_ON_MISMATCH:
        raise ValueError(
            f"Invalid on_mismatch {value!r}. "
            f"Choose one of {sorted(_VALID_ON_MISMATCH)}."
        )
    return value  # type: ignore[return-value]


def get_policy() -> Policy:
    """Resolve the active schema policy (env → process → default, per axis).

    Returns:
        The resolved `Policy`.
    """
    return Policy(
        enabled=resolve_enabled(),
        on_mismatch=_resolve(
            ON_MISMATCH_ENV_VAR,
            _process_on_mismatch,
            DEFAULT_ON_MISMATCH,
            lambda v: _validate_on_mismatch(v.lower()),
        ),
    )


def set_policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_mismatch: OnMismatch | None | _Unset = _UNSET,
) -> None:
    """Set process-wide schema-policy overrides for one or more axes.

    Only the axes you pass are changed.  Pass a value to set it, `None` to
    *clear* that axis's override (so its env var / default applies again), or
    omit it to leave it untouched.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_mismatch: Override the on-mismatch axis, or `None` to clear.
    """
    global _process_on_mismatch
    if not isinstance(enabled, _Unset):
        set_enabled_override(enabled)
    if not isinstance(on_mismatch, _Unset):
        _process_on_mismatch = (
            None if on_mismatch is None else _validate_on_mismatch(on_mismatch)
        )


@contextmanager
def policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_mismatch: OnMismatch | None | _Unset = _UNSET,
):
    """Temporarily override schema-policy axes, restoring them on exit.

    Sets every axis you pass in one go, and restores the previous process
    overrides afterwards even if an exception is raised.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_mismatch: Override the on-mismatch axis, or `None` to clear.
    """
    global _process_on_mismatch
    saved = (get_enabled_override(), _process_on_mismatch)
    set_policy(enabled=enabled, on_mismatch=on_mismatch)
    try:
        yield
    finally:
        enabled_saved, _process_on_mismatch = saved
        set_enabled_override(enabled_saved)
