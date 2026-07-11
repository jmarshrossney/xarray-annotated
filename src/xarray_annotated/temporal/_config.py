"""Process-wide validation policy for the temporal (frequency) domain.

Like the schema policy, temporal validation never converts — it only asserts — but
it has a *second* behavioural axis, because a frequency declaration can fail in two
genuinely different ways:

- `on_mismatch` — the array's time axis has a frequency, and it is not the declared
  one (the declaration is **violated**).
- `on_uninferable` — no frequency could be determined at all: fewer than three
  timestamps, or irregular spacing (the declaration could not be **tested**).  This
  defaults to `"warn"` rather than `"error"`: a short axis is legitimate (a
  two-timestep test fixture), but silently skipping a contract check deserves noise.

Plus `enabled` — the **package-wide** master switch (shared with every domain via
`xarray_annotated._config`); `False` makes all validation a no-op.

Each axis resolves independently: environment variable, then a process override set
via `set_policy` / the `policy` context manager, then the module default.  The shared
`_resolve` helper encodes that precedence.
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
# reach it as `temporal._config.ENABLED_ENV_VAR`).
__all__ = [
    "ENABLED_ENV_VAR",
    "OnMismatch",
    "OnUninferable",
    "Policy",
    "get_policy",
    "policy",
    "set_policy",
]

OnMismatch = Literal["error", "warn", "ignore"]
OnUninferable = Literal["error", "warn", "ignore"]

_VALID_SEVERITIES: frozenset[str] = frozenset({"error", "warn", "ignore"})

#: A frequency mismatch is a genuine bug (a weekly series wired into a daily
#: consumer, or a resample landing on the wrong weekday), so the default is to raise.
DEFAULT_ON_MISMATCH: OnMismatch = "error"

#: An uninferable axis means the declaration was never *tested*, not that it was
#: violated — worth noise, but not fatal.
DEFAULT_ON_UNINFERABLE: OnUninferable = "warn"

ON_MISMATCH_ENV_VAR = "XARRAY_ANNOTATED_TEMPORAL_ON_MISMATCH"
ON_UNINFERABLE_ENV_VAR = "XARRAY_ANNOTATED_TEMPORAL_ON_UNINFERABLE"

_process_on_mismatch: OnMismatch | None = None
_process_on_uninferable: OnUninferable | None = None


@dataclass(frozen=True, slots=True)
class Policy:
    """The resolved temporal-validation policy — the three axes as a single value.

    Returned by `get_policy`.  Build overrides via `set_policy` or the `policy`
    context manager rather than constructing this directly.

    Attributes:
        enabled: Master switch; `False` makes all validation a no-op.
        on_mismatch: Default behaviour when the inferred frequency contradicts the
            declaration (`"error"`, `"warn"`, or `"ignore"`); a marker may override it.
        on_uninferable: Behaviour when no frequency can be inferred from the time
            axis — too few points, or irregular spacing.
    """

    enabled: bool = DEFAULT_ENABLED
    on_mismatch: OnMismatch = DEFAULT_ON_MISMATCH
    on_uninferable: OnUninferable = DEFAULT_ON_UNINFERABLE


def _validate_severity(value: str, axis: str) -> str:
    if value not in _VALID_SEVERITIES:
        raise ValueError(
            f"Invalid {axis} {value!r}. Choose one of {sorted(_VALID_SEVERITIES)}."
        )
    return value


def _validate_on_mismatch(value: str) -> OnMismatch:
    return _validate_severity(value, "on_mismatch")  # type: ignore[return-value]


def _validate_on_uninferable(value: str) -> OnUninferable:
    return _validate_severity(value, "on_uninferable")  # type: ignore[return-value]


def get_policy() -> Policy:
    """Resolve the active temporal policy (env → process → default, per axis).

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
        on_uninferable=_resolve(
            ON_UNINFERABLE_ENV_VAR,
            _process_on_uninferable,
            DEFAULT_ON_UNINFERABLE,
            lambda v: _validate_on_uninferable(v.lower()),
        ),
    )


def set_policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_mismatch: OnMismatch | None | _Unset = _UNSET,
    on_uninferable: OnUninferable | None | _Unset = _UNSET,
) -> None:
    """Set process-wide temporal-policy overrides for one or more axes.

    Only the axes you pass are changed.  Pass a value to set it, `None` to *clear*
    that axis's override (so its env var / default applies again), or omit it to
    leave it untouched.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_mismatch: Override the on-mismatch axis, or `None` to clear.
        on_uninferable: Override the on-uninferable axis, or `None` to clear.
    """
    global _process_on_mismatch, _process_on_uninferable
    if not isinstance(enabled, _Unset):
        set_enabled_override(enabled)
    if not isinstance(on_mismatch, _Unset):
        _process_on_mismatch = (
            None if on_mismatch is None else _validate_on_mismatch(on_mismatch)
        )
    if not isinstance(on_uninferable, _Unset):
        _process_on_uninferable = (
            None if on_uninferable is None else _validate_on_uninferable(on_uninferable)
        )


@contextmanager
def policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_mismatch: OnMismatch | None | _Unset = _UNSET,
    on_uninferable: OnUninferable | None | _Unset = _UNSET,
):
    """Temporarily override temporal-policy axes, restoring them on exit.

    Sets every axis you pass in one go, and restores the previous process overrides
    afterwards even if an exception is raised.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_mismatch: Override the on-mismatch axis, or `None` to clear.
        on_uninferable: Override the on-uninferable axis, or `None` to clear.
    """
    global _process_on_mismatch, _process_on_uninferable
    saved = (get_enabled_override(), _process_on_mismatch, _process_on_uninferable)
    set_policy(enabled=enabled, on_mismatch=on_mismatch, on_uninferable=on_uninferable)
    try:
        yield
    finally:
        enabled_saved, _process_on_mismatch, _process_on_uninferable = saved
        set_enabled_override(enabled_saved)
