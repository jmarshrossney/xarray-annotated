# Declaring properties

Declare what a `DataArray` should look like with markers in its `Annotated` hint, and `@xra.enforce` checks them on every call.

```python
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Coords, Dims, Dtype, Freq, Unit


@xra.enforce
def weekly_mean(
    x: Annotated[
        xr.DataArray,
        Dims("time"),
        Coords("time"),
        Dtype("float64"),
        Unit("degC"),
        Freq("D"),
    ],
) -> Annotated[xr.DataArray, Dims("time"), Unit("degC"), Freq("W-SUN")]:
    return x.resample(time="W-SUN").mean()
```

| Marker | Passes when | Options |
|---|---|---|
| `Dims(*names)` | The array has exactly these dims, in any order | `ordered=True` pins the order |
| `Coords(*names)` | These coordinates are present (extras are fine) | |
| `Dtype(dtype)` | The array has this dtype, in either byte order; `Dtype("float")` means any float, as do `int`, `uint`, `complex`, `str`, `bytes`, `datetime64` and `timedelta64` for their own kinds | |
| `Unit(unit)` | The array is in this unit, however it is spelled | |
| `Freq(freq)` | [The time axis](#frequency) has this spacing, and this phase where spelled | `dim=` |

## Rules

- **Where markers go:** on a parameter, on a return, or on the fields of a `TypedDict` or dataclass return.
- **Return containers:** a `TypedDict` or dataclass return may also be written `R | None`, `R[int]` or `Annotated[R, "description"]`, or through a `type` alias. Fields carrying markers anywhere else, such as in a parameter, a `NamedTuple`, `list[R]`, `R1 | R2`, or a class nested in another's field, raise at decoration.
- **Returned containers are not changed in place:** the caller receives a shallow copy of the dict or dataclass, which shares its arrays. `__post_init__` does not run again.
- **Only a `DataArray` passes:** anything else raises `TypeError`, including `None`, an ndarray and a `Dataset`.
- **Unless the hint allows it:** `| None` allows `None`, `| float` allows anything else unchecked, and `Annotated[xr.DataArray | None, ...]` and `Annotated[xr.DataArray, ...] | None` are equivalent. A `TypedDict` field may be `NotRequired`.
- **One of each:** a second marker of the same kind on one hint raises `ValueError`, so write `Coords("time", "lat")`.
- **Nowhere else:** a marker `enforce` cannot apply raises `ValueError`, such as inside a `tuple[...]` return or on `Annotated[xr.Dataset, ...]`.
- **Fail fast:** a malformed declaration (a typo'd unit, an unparseable dtype or frequency) raises `ValueError` when the function is decorated.
- **Units first:** units are checked before other markers, on both sides.
- **Nothing changes:** every marker only checks, so the function receives and returns the arrays as they were.

A few hints are silently not declarations; see [Troubleshooting](troubleshooting.md#a-declaration-is-silently-ignored).

## Units

Inputs and outputs must both already be in the declared unit, and anything else raises.
See [Units](units.md) for what happens in each case and how to convert an input, and [Computing with pint](pint.md) for how a body gets the unit it returns.

## Frequency

`Freq` checks the array's one datetime coordinate with a pandas index, which a dimension coordinate has by default, so the check reads nothing from disk.
If there is no such coordinate, or there are two, it raises `FreqError`, and names the fix where it can tell; use `dim=` to choose between two.
A scalar, 2-D or unindexed coordinate is not a time axis, so `time(obs)` needs `.set_xindex("time")`, and alongside a real one it is left unchecked.

A regular axis has its frequency inferred with `xarray.infer_freq`, so `cftime` calendars work too.
**Spacing** is always compared, by meaning rather than by string:

| Declared | Actual | Result | Why |
|---|---|---|---|
| `Freq("7D")` | `W-WED` | pass | Seven days either way |
| `Freq("D")` | `24h` | pass | Same fixed spacing |
| `Freq("QE")` | `3ME` | pass | A quarter is three months |
| `Freq("ME")` | `30D` | fail | Calendar months are not fixed-length days |

**Phase** is compared in two parts.
The end or begin convention (`"ME"` vs `"MS"`) is always compared, and the anchor only if you spell it:

```python
Freq("W")  # weekly, any weekday
Freq("W-SUN")  # weekly, Sundays only
```

Unlike pandas, an unspelled anchor means "any" rather than a default.
A quarterly anchor names four dates rather than one month, so `"QE-MAR"` and `"QE-DEC"` are the same.

Every axis is tested, whatever its length.
Irregular spacing raises, such as a missing or duplicated timestamp, unless [the declaration allows gaps](#axes-with-gaps), and two timestamps are compared directly, though business-day and custom frequencies have nothing to compare, so pass.
Zero or one timestamp always passes.

### Calendar steps on any day

Monthly means stamped at month midpoints have no pandas frequency, as the midpoint moves with the length of the month, so `"MS"` and `"ME"` raise on them.
Declare the calendar step alone instead:

```python
Freq("month")  # one value per calendar month, on any day
Freq("3month")  # one value every three months, such as seasonal means
Freq("year")  # one value per year, in the same month each year
```

These compare consecutive months and ignore the day and time, so a missing, duplicated or out-of-order month still raises, on `cftime` calendars too.
The legacy alias `"M"` is still an error.

### Axes with gaps

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

## Reusing a declaration

Writing markers in the signature keeps each declaration in view, so it is a good default.
When signatures get long, or the same markers repeat across many functions, you can name a declaration with a `type` statement or an assignment:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]
```

Wrap a named declaration to add markers to it:

```python
type TypedPressure = Annotated[Pressure, Dtype("float64")]
```

It declares `Unit`, `Dims` and `Dtype`.

An alias is checked exactly like the hint it stands for, but keep these in mind:

- **The markers move out of the signature.** A reader, an AI agent or generated documentation sees `Pressure` and has to look up what it declares, so name an alias after what it declares and keep it close to the code that uses it.
- **Only repetition pays for an alias.** It takes no marker arguments, so each distinct combination needs its own name, and a one-off function is usually clearer written out.
- **Alias what is shared.** A marker cannot be replaced, so a second `Unit` raises `ValueError`. Name the parts that repeat, such as dims and frequency, and add the rest where they are used, rather than adjusting a complete alias.
- **Import aliases at run time.** `enforce` reads them at decoration, so an alias imported only under `TYPE_CHECKING` raises `NameError`.
