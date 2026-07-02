"""Process-wide validation policy: three orthogonal axes.

`check_units` handles three distinct events, and the policy exposes one independent
control per event:

- `enabled` — master switch; when `False` no validation happens at all (a true
  no-op, at every layer).  This is the **package-wide** switch shared with every
  other domain; it lives in `xarray_annotated._config` and is merely surfaced on
  this domain's `Policy`.
- `on_missing` — what to do when a `DataArray` has no parseable unit to check
  against (an absent or unparseable `units` attribute): `"error"` / `"warn"` /
  `"ignore"`.
- `on_inexact` — what to do when the declared and actual units are dimensionally
  compatible but *value-changing* (e.g. `"hPa"` where `"Pa"` is declared):
  `"convert"` / `"warn"` / `"error"`.

A fourth case — a *dimensional* mismatch (e.g. a mass where a pressure is declared)
— is deliberately not configurable: it always raises, because the values cannot be
converted and continuing would corrupt results silently.

Each axis resolves independently: environment variable, then a process override set
via `set_policy` / the `policy` context manager, then the module default. The
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

# `ENABLED_ENV_VAR` / `DEFAULT_ENABLED` are re-exported (the master switch is
# shared but callers still reach it as `units._config.ENABLED_ENV_VAR`).
__all__ = ["ENABLED_ENV_VAR", "Policy", "get_policy", "policy", "set_policy"]

OnMissing = Literal["error", "warn", "ignore"]
OnInexact = Literal["convert", "warn", "error"]

_VALID_ON_MISSING: frozenset[str] = frozenset({"error", "warn", "ignore"})
_VALID_ON_INEXACT: frozenset[str] = frozenset({"convert", "warn", "error"})

#: When a `DataArray` carries no parseable unit, `warn` flags it without
#: failing — non-breaking for inputs that lack unit metadata, and dev-friendly.
DEFAULT_ON_MISSING: OnMissing = "warn"

#: A dimensionally compatible but value-changing unit is silently converted by
#: default; `warn` also converts but says so, `error` refuses.
DEFAULT_ON_INEXACT: OnInexact = "convert"

ON_MISSING_ENV_VAR = "XARRAY_ANNOTATED_UNITS_ON_MISSING"
ON_INEXACT_ENV_VAR = "XARRAY_ANNOTATED_UNITS_ON_INEXACT"

_process_on_missing: OnMissing | None = None
_process_on_inexact: OnInexact | None = None


@dataclass(frozen=True, slots=True)
class Policy:
    """The resolved validation policy — the three axes as a single value.

    Returned by `get_policy`.  Build overrides via `set_policy` or the `policy`
    context manager rather than constructing this directly.

    Attributes:
        enabled: Master switch; `False` makes all validation a no-op.
        on_missing: Behaviour when a `DataArray` has no parseable unit
            (`"error"`, `"warn"`, or `"ignore"`).
        on_inexact: Behaviour for a value-changing conversion
            (`"convert"`, `"warn"`, or `"error"`).
    """

    enabled: bool = DEFAULT_ENABLED
    on_missing: OnMissing = DEFAULT_ON_MISSING
    on_inexact: OnInexact = DEFAULT_ON_INEXACT


def _validate_on_missing(value: str) -> OnMissing:
    if value not in _VALID_ON_MISSING:
        raise ValueError(
            f"Invalid on_missing {value!r}. Choose one of {sorted(_VALID_ON_MISSING)}."
        )
    return value  # type: ignore[return-value]


def _validate_on_inexact(value: str) -> OnInexact:
    if value not in _VALID_ON_INEXACT:
        raise ValueError(
            f"Invalid on_inexact {value!r}. Choose one of {sorted(_VALID_ON_INEXACT)}."
        )
    return value  # type: ignore[return-value]


def get_policy() -> Policy:
    """Resolve the active validation policy (env → process → default, per axis).

    Returns:
        The resolved `Policy`.
    """
    return Policy(
        enabled=resolve_enabled(),
        on_missing=_resolve(
            ON_MISSING_ENV_VAR,
            _process_on_missing,
            DEFAULT_ON_MISSING,
            lambda v: _validate_on_missing(v.lower()),
        ),
        on_inexact=_resolve(
            ON_INEXACT_ENV_VAR,
            _process_on_inexact,
            DEFAULT_ON_INEXACT,
            lambda v: _validate_on_inexact(v.lower()),
        ),
    )


def set_policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_missing: OnMissing | None | _Unset = _UNSET,
    on_inexact: OnInexact | None | _Unset = _UNSET,
) -> None:
    """Set process-wide policy overrides for one or more axes.

    Only the axes you pass are changed.  Pass a value to set it, `None` to
    *clear* that axis's override (so its env var / default applies again), or
    omit it to leave it untouched.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_missing: Override the on-missing axis, or `None` to clear.
        on_inexact: Override the on-inexact axis, or `None` to clear.
    """
    global _process_on_missing, _process_on_inexact
    if not isinstance(enabled, _Unset):
        set_enabled_override(enabled)
    if not isinstance(on_missing, _Unset):
        _process_on_missing = (
            None if on_missing is None else _validate_on_missing(on_missing)
        )
    if not isinstance(on_inexact, _Unset):
        _process_on_inexact = (
            None if on_inexact is None else _validate_on_inexact(on_inexact)
        )


@contextmanager
def policy(
    *,
    enabled: bool | None | _Unset = _UNSET,
    on_missing: OnMissing | None | _Unset = _UNSET,
    on_inexact: OnInexact | None | _Unset = _UNSET,
):
    """Temporarily override policy axes, restoring all of them on exit.

    Sets every axis you pass in one go, and restores the previous process
    overrides afterwards even if an exception is raised.

    Args:
        enabled: Override the (package-wide) master switch, or `None` to clear.
        on_missing: Override the on-missing axis, or `None` to clear.
        on_inexact: Override the on-inexact axis, or `None` to clear.

    Examples:
        >>> from xarray_annotated.units import policy
        >>> import xarray as xr
        >>> da = xr.DataArray([1.0], attrs={"units": "Pa"})
        >>> with policy(on_missing="error"):
        ...     pass  # policy restored after block
    """
    global _process_on_missing, _process_on_inexact
    saved = (get_enabled_override(), _process_on_missing, _process_on_inexact)
    set_policy(enabled=enabled, on_missing=on_missing, on_inexact=on_inexact)
    try:
        yield
    finally:
        enabled_saved, _process_on_missing, _process_on_inexact = saved
        set_enabled_override(enabled_saved)
