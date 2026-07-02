"""Process-wide validation policy: three orthogonal axes.

`check_units` handles three distinct events, and the policy exposes one independent
control per event:

- `enabled` — master switch; when `False` no validation happens at all (a true
  no-op, at every layer).
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
via `set_policy` / the `policy` context manager, then the module default. `_resolve`
encodes that shared precedence.
"""

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TypeVar

OnMissing = Literal["error", "warn", "ignore"]
OnInexact = Literal["convert", "warn", "error"]

_VALID_ON_MISSING: frozenset[str] = frozenset({"error", "warn", "ignore"})
_VALID_ON_INEXACT: frozenset[str] = frozenset({"convert", "warn", "error"})

#: Master switch default. Validation is on unless explicitly disabled.
DEFAULT_ENABLED: bool = True

#: When a `DataArray` carries no parseable unit, `warn` flags it without
#: failing — non-breaking for inputs that lack unit metadata, and dev-friendly.
DEFAULT_ON_MISSING: OnMissing = "warn"

#: A dimensionally compatible but value-changing unit is silently converted by
#: default; `warn` also converts but says so, `error` refuses.
DEFAULT_ON_INEXACT: OnInexact = "convert"

ENABLED_ENV_VAR = "XARRAY_SIGNATURE_UNITS_ENABLED"
ON_MISSING_ENV_VAR = "XARRAY_SIGNATURE_UNITS_ON_MISSING"
ON_INEXACT_ENV_VAR = "XARRAY_SIGNATURE_UNITS_ON_INEXACT"

#: String values (lower-cased) accepted for the boolean `enabled` env var.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSEY: frozenset[str] = frozenset({"0", "false", "no", "off"})

_process_enabled: bool | None = None
_process_on_missing: OnMissing | None = None
_process_on_inexact: OnInexact | None = None

_T = TypeVar("_T")


class _Unset:
    """Sentinel distinguishing 'argument omitted' from 'explicitly set to None'."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


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


def _resolve(
    env_var: str,
    process_value: _T | None,
    default: _T,
    parse: Callable[[str], _T],
) -> _T:
    """Resolve one axis: env var (parsed) → process override → default."""
    env = os.environ.get(env_var)
    if env:
        return parse(env)
    if process_value is not None:
        return process_value
    return default


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


def get_policy() -> Policy:
    """Resolve the active validation policy (env → process → default, per axis).

    Returns:
        The resolved `Policy`.
    """
    return Policy(
        enabled=_resolve(
            ENABLED_ENV_VAR, _process_enabled, DEFAULT_ENABLED, _parse_enabled_env
        ),
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
        enabled: Override the master switch, or `None` to clear.
        on_missing: Override the on-missing axis, or `None` to clear.
        on_inexact: Override the on-inexact axis, or `None` to clear.
    """
    global _process_enabled, _process_on_missing, _process_on_inexact
    if not isinstance(enabled, _Unset):
        _process_enabled = None if enabled is None else bool(enabled)
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
        enabled: Override the master switch, or `None` to clear.
        on_missing: Override the on-missing axis, or `None` to clear.
        on_inexact: Override the on-inexact axis, or `None` to clear.

    Examples:
        >>> from xarray_signature_units import policy
        >>> import xarray as xr
        >>> da = xr.DataArray([1.0], attrs={"units": "Pa"})
        >>> with policy(on_missing="error"):
        ...     pass  # policy restored after block
    """
    global _process_enabled, _process_on_missing, _process_on_inexact
    saved = (_process_enabled, _process_on_missing, _process_on_inexact)
    set_policy(enabled=enabled, on_missing=on_missing, on_inexact=on_inexact)
    try:
        yield
    finally:
        _process_enabled, _process_on_missing, _process_on_inexact = saved
