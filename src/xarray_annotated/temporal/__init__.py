"""Temporal-frequency domain for xarray-annotated.

Declares and validates the *frequency* of a DataArray's time axis — a property
derived from the values of its datetime coordinate rather than carried in its
metadata, which is why it is a domain of its own rather than a fourth structural
marker.  Like ``xarray_annotated.schema`` (and unlike ``units``) it only *asserts*:
it never converts, resamples, or stamps anything.

A declaration compares on two things: **spacing** (always) and **phase** (the
``End``/``Begin`` convention, and the anchor where both sides spell one).  So
``Freq("7D")`` accepts a weekly axis on any weekday, while ``Freq("W-SUN")`` catches
the classic resample footgun of a series landing on Wednesdays::

    from typing import Annotated
    import xarray as xr
    from xarray_annotated.temporal import declare_freq, Freq

    @declare_freq
    def weekly_mean(x: Annotated[xr.DataArray, Freq("D")]) -> Annotated[xr.DataArray, Freq("W-SUN")]:
        return x.resample(time="W-SUN").mean()

``freq_compatible`` is the same comparison with no array in hand, for a build-time
check of a producer/consumer edge.
"""

from ._annotations import Freq, annotated_freq, freq_from_signature
from ._check import (
    FreqError,
    FreqWarning,
    assert_valid_freq,
    check_freq,
    freq_compatible,
)
from ._config import (
    OnMismatch,
    OnUninferable,
    Policy,
    get_policy,
    policy,
    set_policy,
)
from ._decorator import declare_freq

__all__ = [
    "Freq",
    "FreqError",
    "FreqWarning",
    "OnMismatch",
    "OnUninferable",
    "Policy",
    "annotated_freq",
    "assert_valid_freq",
    "check_freq",
    "declare_freq",
    "freq_compatible",
    "freq_from_signature",
    "get_policy",
    "policy",
    "set_policy",
]
