"""Temporal-frequency domain for xarray-annotated.

Declares and validates the *frequency* of a DataArray's time axis — a property
derived from the values of its datetime coordinate rather than carried in its
metadata, which is why it is a domain of its own rather than a fourth structural
marker.  Like ``xarray_annotated.schema`` (and unlike ``units``) it only *asserts*:
it never converts, resamples, or stamps anything.

A declaration compares on two things: **spacing** (always) and **phase** (the
``End``/``Begin`` convention, and the anchor where both sides spell one).  So
``Freq("7D")`` accepts a weekly axis on any weekday, while ``Freq("W-SUN")`` catches
the classic resample footgun of a series landing on Wednesdays:

    from typing import Annotated
    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Freq

    @xra.enforce
    def weekly_mean(
        x: Annotated[xr.DataArray, Freq("D")],
    ) -> Annotated[xr.DataArray, Freq("W-SUN")]:
        return x.resample(time="W-SUN").mean()

``freq_compatible`` is the same comparison with no array in hand, for a build-time
check of a producer/consumer edge.
"""

from ._annotations import Freq
from ._check import (
    FreqError,
    assert_valid_freq,
    check_freq,
    freq_compatible,
)

__all__ = [
    "Freq",
    "FreqError",
    "assert_valid_freq",
    "check_freq",
    "freq_compatible",
]
