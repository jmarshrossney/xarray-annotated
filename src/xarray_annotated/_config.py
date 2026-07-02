"""Shared configuration kernel for xarray-annotated.

Machinery common to every annotation-validation domain (currently `units` and
`schema`), so the domains never drift on how a policy axis resolves:

- `_resolve` — the shared precedence for one axis: environment variable (parsed),
  then a process override, then a module default.
- the **`enabled`** master switch — package-wide (it gates *every* domain, not just
  one), so it lives here rather than in any single domain. Each domain's `Policy`
  carries an `enabled` field but resolves it through `resolve_enabled` / the
  override accessors below, so toggling it in one domain toggles it everywhere.
- `_Unset` / `_UNSET` — the sentinel that lets a `set_policy` keyword distinguish
  "argument omitted" from "explicitly set to `None`" (which *clears* an override).

Domain-specific axes (units' `on_missing` / `on_inexact`, schema's `on_mismatch`)
stay in the respective domain's `_config.py`; only the genuinely shared pieces
live here.
"""

import os
from collections.abc import Callable
from typing import TypeVar

#: Master switch default. Validation is on unless explicitly disabled.
DEFAULT_ENABLED: bool = True

#: Master switch is package-wide (gates every annotation-validation domain), so it
#: is deliberately un-namespaced; each domain's behavioural knobs are namespaced.
ENABLED_ENV_VAR = "XARRAY_ANNOTATED_ENABLED"

#: String values (lower-cased) accepted for the boolean `enabled` env var.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSEY: frozenset[str] = frozenset({"0", "false", "no", "off"})

#: Process-wide override for the master switch, shared across all domains.
_process_enabled: bool | None = None

_T = TypeVar("_T")


class _Unset:
    """Sentinel distinguishing 'argument omitted' from 'explicitly set to None'."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


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
    """Resolve the package-wide master switch (env → process override → default)."""
    return _resolve(
        ENABLED_ENV_VAR, _process_enabled, DEFAULT_ENABLED, _parse_enabled_env
    )


def get_enabled_override() -> bool | None:
    """Return the current process override for the master switch (`None` if unset).

    Used by each domain's `policy` context manager to save/restore the shared
    switch around a temporary override.
    """
    return _process_enabled


def set_enabled_override(value: bool | None) -> None:
    """Set (or, with `None`, clear) the process override for the master switch.

    Shared by every domain's `set_policy`, so disabling validation in one domain
    disables it package-wide.
    """
    global _process_enabled
    _process_enabled = None if value is None else bool(value)
