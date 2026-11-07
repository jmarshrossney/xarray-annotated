# Temporal declarations

[`Freq`][xarray_annotated.Freq] declares the spacing of an array's time axis.
A mismatch raises [`FreqError`][xarray_annotated.temporal.FreqError], and a frequency pandas cannot parse raises `ValueError` when the function is decorated, including the legacy aliases `"M"` and `"H"`, now spelled `"ME"` and `"h"`.

```python
import numpy as np
import pandas as pd


@xra.check_annotations
def daily_total(x: Annotated[xr.DataArray, Freq("D")]) -> None: ...


days = pd.date_range("2024-01-01", periods=5, freq="D")
series = xr.DataArray(np.ones(5), dims="time", coords={"time": days})
daily_total(series)
```

A missing timestamp makes the axis irregular, which raises:

<!-- docs-test: raises FreqError -->
```python
daily_total(series.drop_isel(time=2))
```

## The time axis

The time axis is the array's one datetime coordinate with a pandas index, which a dimension coordinate has by default, so the check reads nothing from disk.
`cftime` calendars work too.

- **Two time coordinates:** pass `dim` to choose, as in `Freq("h", dim="valid_time")`.
- **None:** the check raises `FreqError`, as after `.mean("time")`, and names the fix where it can tell.
- **Unindexed coordinates:** a scalar, 2-D or unindexed coordinate is not a time axis, so `time(obs)` needs `.set_xindex("time")`.

## Spacing

Spacing is compared by meaning rather than by string:

| Declared | Actual | Result | Why |
|---|---|---|---|
| `Freq("7D")` | `W-WED` | pass | Seven days either way |
| `Freq("D")` | `24h` | pass | Same fixed spacing |
| `Freq("QE")` | `3ME` | pass | A quarter is three months |
| `Freq("ME")` | `30D` | fail | Calendar months are not fixed-length days |

Every axis is tested, whatever its length: two timestamps are compared directly, and zero or one always passes.
Business-day and custom frequencies have no step to compare, so a two-timestamp axis passes them.

## Phase

Phase is compared in two parts:

- **End or begin:** `"ME"` and `"MS"` always differ.
- **Anchor:** compared only if you spell it, so `Freq("W")` means weekly on any weekday and `Freq("W-SUN")` weekly on Sundays only. Unlike pandas, an unspelled anchor means "any" rather than a default.

A quarterly anchor names four dates rather than one month, so `"QE-MAR"` and `"QE-DEC"` are the same.

## Calendar steps on any day

Monthly means stamped at month midpoints have no pandas frequency, as the midpoint moves with the length of the month, so `"MS"` and `"ME"` raise on them.
Declare the calendar step alone instead:

```python
Freq("month")  # one value per calendar month, on any day
Freq("3month")  # one value every three months, such as seasonal means
Freq("year")  # one value per year, in the same month each year
```

These compare consecutive months and ignore the day and time, so a missing, duplicated or out-of-order month still raises, on `cftime` calendars too.
The legacy alias `"M"` is still an error.

## Axes with gaps

`Freq("D")` means each value represents one day and no timestamp is missing.
A function that works point by point needs only the first of those, so `gaps_ok=True` declares the sampling period alone:

```python
Freq("D", gaps_ok=True)  # daily values, such as January 2025 joined to January 2026
Freq("ME", gaps_ok=True)  # month ends, with some months absent
Freq("month", gaps_ok=True)  # December to February of several years
```

Timestamps must increase, every step must be a whole multiple of the declared one, and the smallest step must equal it.
The end or begin convention and any anchor are still compared.

| Declared | Actual | Result | Why |
|---|---|---|---|
| `Freq("D", gaps_ok=True)` | Two runs of daily values | pass | Whole days, and some are one day apart |
| `Freq("D", gaps_ok=True)` | Half-hourly | fail | Half a day is not a whole multiple |
| `Freq("D", gaps_ok=True)` | Weekly | fail | No two values are one day apart |
| `Freq("D", gaps_ok=True)` | Daily values with no two adjacent | fail | The axis cannot be told apart from coarser data |
| `Freq("month", gaps_ok=True)` | January of each year | fail | No two values are one month apart, so declare `Freq("year")` |

`gaps_ok` cannot be combined with business-day or custom frequencies, which have no step to compare, so that raises `ValueError` where the function is defined.

Comparing declarations with [`freq_compatible`][xarray_annotated.temporal.freq_compatible] ignores `gaps_ok`, as it asks only whether two declarations can describe the same axis.
A consumer without `gaps_ok` still raises on an axis with holes when the data arrives.
