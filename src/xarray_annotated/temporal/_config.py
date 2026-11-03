"""Behavioural axes of the temporal domain: their values, validators and defaults.

Temporal validation never converts, but a frequency declaration can fail in two
genuinely different ways:

- `on_mismatch` — the array's time axis has a frequency, and it is not the
  declared one (the declaration is **violated**).
- `on_uninferable` — no frequency could be determined at all: fewer than three
  timestamps, or irregular spacing (the declaration could not be **tested**).

`on_mismatch` resolves per call as: the `Freq` marker's override, then the
`enforce` keyword or `check_freq` argument, then the default here.
`on_uninferable` has no marker override, since a short axis is a property of the
data rather than the declaration.  Nothing is configured globally.
"""

from typing import Literal

__all__ = [
    "DEFAULT_ON_MISMATCH",
    "DEFAULT_ON_UNINFERABLE",
    "OnMismatch",
    "OnUninferable",
]

OnMismatch = Literal["error", "warn", "ignore"]
OnUninferable = Literal["error", "warn", "ignore"]

_VALID_SEVERITIES: frozenset[str] = frozenset({"error", "warn", "ignore"})

#: A frequency mismatch is a genuine bug (a weekly series wired into a daily
#: consumer, or a resample landing on the wrong weekday), so the default is to raise.
DEFAULT_ON_MISMATCH: OnMismatch = "error"

#: An uninferable axis means the declaration was never *tested*, not that it was
#: violated — worth noise, but not fatal.  A short test fixture is legitimate;
#: silence it with a warnings filter.
DEFAULT_ON_UNINFERABLE: OnUninferable = "warn"


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
