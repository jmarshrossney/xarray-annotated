"""Behavioural axes of the units domain: their values, validators and defaults.

Inputs have no axis, only an opt-in, and outputs have one axis:

- `convert` — the declared and actual units are dimensionally compatible but
  *value-changing* (e.g. `"hPa"` where `"Pa"` is declared).  Not an axis but an
  opt-in on the `Unit` marker (or `check_units` argument): without it this
  raises, because whether a conversion is correct depends on the quantity.
- `on_output` — whether a declared return value's own `units` attribute is
  verified before being overwritten: `"stamp"` / `"strict"` / `"require"`.  See
  `apply_output_units` for why the default cannot be otherwise.

An input with no parseable unit (absent, over-long or unparseable `units`)
always raises, because the declaration cannot be checked; attach the unit where
the data enters.  A *dimensional* mismatch (a mass where a pressure is declared)
always raises too, because the values cannot be converted.

`on_output` resolves per call as: the `Unit` marker's override, then the
`enforce` keyword or `apply_output_units` argument, then the default here.
`convert` has no `enforce` keyword.  Nothing is configured
globally.
"""

from typing import Literal

__all__ = [
    "DEFAULT_ON_OUTPUT",
    "OnOutput",
]

OnOutput = Literal["stamp", "strict", "require"]

_VALID_ON_OUTPUT: frozenset[str] = frozenset({"stamp", "strict", "require"})

#: Outputs are stamped, not verified, by default: a body that does its own unit
#: arithmetic leaves `attrs` stale, and re-converting would corrupt the values.
DEFAULT_ON_OUTPUT: OnOutput = "stamp"


def _validate_on_output(value: str) -> OnOutput:
    if value not in _VALID_ON_OUTPUT:
        raise ValueError(
            f"Invalid on_output {value!r}. Choose one of {sorted(_VALID_ON_OUTPUT)}."
        )
    return value  # type: ignore[return-value]


def _validate_convert(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"convert must be a bool, not {type(value).__name__}.")
    return value
