"""The one behavioural setting of the units domain: its validator.

`convert` applies when an input's unit is dimensionally compatible with the declared
one but *value-changing* (e.g. `"hPa"` where `"Pa"` is declared).  Without it this
raises, because whether a conversion is correct depends on the quantity.  It is an
opt-in on the `Unit` marker, and nothing is configured globally.

Everything else always raises: an input with no parseable unit, because the
declaration cannot be checked; a *dimensional* mismatch, because the values cannot be
converted; and an output that does not match its declaration, because the function's
author controls what it returns.
"""

import numpy as np


def _validate_convert(value: object) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"convert must be a bool, not {type(value).__qualname__}.")
    return bool(value)
