# Declaration patterns

[Getting started](../getting_started.md) covers one array in and one array out.
This page covers the other shapes a signature takes.
For exactly what each marker and `enforce` accept, see the [API reference](../api/package.md).

## Return several arrays

Return a `TypedDict` or dataclass, and declare its fields:

```python
from typing import NotRequired, TypedDict

import numpy as np


class Wind(TypedDict):
    speed: Annotated[xr.DataArray, Dims("time"), Unit("m/s")]
    direction: Annotated[xr.DataArray, Dims("time"), Unit("degree")]
    gust: NotRequired[Annotated[xr.DataArray, Dims("time"), Unit("m/s")]]


@xra.enforce
def wind(
    u: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
    v: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
) -> Wind:
    speed = np.hypot(u, v).assign_attrs(units="m/s")
    direction = np.degrees(np.arctan2(u, v)).assign_attrs(units="degree")
    return {"speed": speed, "direction": direction}
```

- **Missing fields:** leaving out a required field raises `TypeError`, whereas a `NotRequired` field may be absent.
- **Undeclared fields:** a field with no markers is not checked.
- **The container itself:** the caller receives a shallow copy of the dict or dataclass, which shares its arrays, and `__post_init__` does not run again.

A `tuple` or `NamedTuple` return has no fields `enforce` can read, so markers there raise `ValueError` when the function is decorated.
The same goes for a parameter holding a class with declared fields, so declare a parameter's arrays as parameters of their own.

## Accept or return an optional array

Add what else the value may be to the hint:

```python
@xra.enforce
def weighted_mean(
    x: Annotated[xr.DataArray, Dims("time"), Unit("K")],
    weights: Annotated[xr.DataArray, Dims("time")] | None = None,
) -> Annotated[xr.DataArray, Unit("K")]:
    if weights is None:
        return x.mean("time", keep_attrs=True)
    return x.weighted(weights).mean("time", keep_attrs=True)
```

The checks apply when the value is a `DataArray`, and `None` passes.
Without `| None`, passing `None` raises `TypeError`.

Any other member of the union allows anything else, unchecked, so `Annotated[xr.DataArray, Unit("K")] | float` accepts a `float` or even a string.

A `TypedDict` or dataclass return may also be optional, written `Wind | None`.

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

## Reuse a declaration

Writing markers in the signature keeps each declaration in view, so it is a good default.
When signatures get long, or the same markers repeat across many functions, name a declaration with a `type` statement or an assignment:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]
```

Wrap a named declaration to add markers to it:

```python
type TypedPressure = Annotated[Pressure, Dtype("float64")]
```

An alias is checked exactly like the hint it stands for, but keep these in mind:

- **The markers move out of the signature.** A reader, an AI agent or generated documentation sees `Pressure` and has to look up what it declares, so name an alias after what it declares and keep it close to the code that uses it.
- **Only repetition pays for an alias.** It takes no marker arguments, so each distinct combination needs its own name, and a one-off function is usually clearer written out.
- **Alias what is shared.** A marker cannot be replaced, so a second `Unit` raises `ValueError`. Name the parts that repeat, such as dims and frequency, and add the rest where they are used.
- **Import aliases at run time.** `enforce` reads them at decoration, so an alias imported only under `TYPE_CHECKING` raises `NameError`.
