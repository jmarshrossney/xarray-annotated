"""Behavioural axes of the units domain: their values, validators and defaults.

Conversion has no axis, only an opt-in, and outputs have one axis:

- `convert` — the declared and actual units are dimensionally compatible but
  *value-changing* (e.g. `"hPa"` where `"Pa"` is declared), on an input or a
  pint-quantified output.  Not an axis but an opt-in on the `Unit` marker: without
  it this raises, because whether a conversion is correct depends on the quantity.
- `on_output` — whether a declared return value's own `units` attribute is
  verified before being overwritten: `"stamp"` / `"strict"` / `"require"`.  See
  `apply_output_units` for why the default cannot be otherwise.

An input with no parseable unit (absent, over-long or unparseable `units`)
always raises, because the declaration cannot be checked; attach the unit where
the data enters.  A *dimensional* mismatch (a mass where a pressure is declared)
always raises too, because the values cannot be converted.

`on_output` resolves per call as: the `Unit` marker's override, then the
`enforce` keyword or `apply_output_units` argument, then the default here.
`convert` is set only on the marker.  Nothing is configured globally.
"""

from typing import Literal

import numpy as np

__all__ = [
    "DEFAULT_ON_OUTPUT",
    "OnOutput",
]

OnOutput = Literal["stamp", "strict", "require"]

_VALID_ON_OUTPUT: frozenset[str] = frozenset({"stamp", "strict", "require"})

#: Outputs are verified by default.  A body doing its own unit arithmetic leaves
#: `attrs` stale and must say so, which is a one-line fix, whereas stamping over the
#: label would silently mislabel a body that forgot the conversion.
DEFAULT_ON_OUTPUT: OnOutput = "strict"


def _validate_on_output(value: str) -> OnOutput:
    if value not in _VALID_ON_OUTPUT:
        raise ValueError(
            f"Invalid on_output {value!r}. Choose one of {sorted(_VALID_ON_OUTPUT)}."
        )
    return value  # type: ignore[return-value]


def _validate_convert(value: object) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"convert must be a bool, not {type(value).__qualname__}.")
    return bool(value)
