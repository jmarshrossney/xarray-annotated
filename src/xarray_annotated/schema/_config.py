"""Behavioural axis of the schema domain: its values, validator and default.

Structural validation never converts, so there is one axis:

- `on_mismatch` — a DataArray's structure does not match a declaration
  (`"error"` / `"warn"` / `"ignore"`).

It resolves per call as: the marker's override (so a wrong dtype can warn while
wrong dims still raise), then the `check_schema` argument, then the default here.
`enforce` has no keyword for it.  Nothing is configured globally.
"""

from typing import Literal

__all__ = ["DEFAULT_ON_MISMATCH", "OnMismatch"]

OnMismatch = Literal["error", "warn", "ignore"]

_VALID_ON_MISMATCH: frozenset[str] = frozenset({"error", "warn", "ignore"})

#: A structural mismatch is usually a genuine bug (wrong array wired in), so the
#: default is to raise.  Soften it per marker: `Dims(..., on_mismatch="warn")`.
DEFAULT_ON_MISMATCH: OnMismatch = "error"


def _validate_on_mismatch(value: str) -> OnMismatch:
    if value not in _VALID_ON_MISMATCH:
        raise ValueError(
            f"Invalid on_mismatch {value!r}. "
            f"Choose one of {sorted(_VALID_ON_MISMATCH)}."
        )
    return value  # type: ignore[return-value]
