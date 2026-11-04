"""Behavioural axis of the temporal domain: its values, validator and default.

Temporal validation never converts.  A frequency that contradicts a declaration always
raises, so there is one axis:

- `on_uninferable` — no frequency could be determined at all: fewer than three
  timestamps, or irregular spacing (the declaration could not be **tested**).

It resolves as: the `enforce` keyword or `check_freq` argument, then the default.  It
has no marker override, since a short axis is a property of the data rather than the
declaration.  Nothing is configured globally.
"""

from typing import Literal

__all__ = ["DEFAULT_ON_UNINFERABLE", "OnUninferable"]

OnUninferable = Literal["error", "warn", "ignore"]

_VALID_ON_UNINFERABLE: frozenset[str] = frozenset({"error", "warn", "ignore"})

#: An uninferable axis means the declaration was never *tested*, not that it was
#: violated — worth noise, but not fatal.  A short test fixture is legitimate;
#: silence it with a warnings filter.
DEFAULT_ON_UNINFERABLE: OnUninferable = "warn"


def _validate_on_uninferable(value: str) -> OnUninferable:
    if value not in _VALID_ON_UNINFERABLE:
        raise ValueError(
            f"Invalid on_uninferable {value!r}. "
            f"Choose one of {sorted(_VALID_ON_UNINFERABLE)}."
        )
    return value  # type: ignore[return-value]
