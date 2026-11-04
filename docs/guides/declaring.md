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
| `Dtype(dtype)` | The numpy kind matches: `float32` passes `Dtype("float64")`, `uint8` fails `Dtype("int32")` | `exact=True` requires the precise dtype |
| `Unit(unit)` | The array is in this unit, however it is spelled | |
| `Freq(freq)` | [The time axis](#which-coordinate-is-the-time-axis) has this spacing, and this phase [where spelled](#what-the-same-frequency-means) | `dim=`, `anchored=` |

## Rules

- **Where markers go:** on a parameter, on a return, or on the fields of a `TypedDict` or dataclass return.
- **Return containers:** a `TypedDict` or dataclass return may also be written `R | None`, `R[int]` or `Annotated[R, "description"]`, or through a `type` alias. Fields carrying markers anywhere else, such as in a `NamedTuple`, `list[R]`, `R1 | R2`, or a class nested in another's field, raise at decoration.
- **Returned containers are not changed in place:** the caller receives a shallow copy of the dict or dataclass when a field changed, which shares its arrays. A failed check leaves the original as it was, and `__post_init__` does not run again.
- **Optional values:** `Annotated[xr.DataArray | None, ...]` and `Annotated[xr.DataArray, ...] | None` are equivalent, and a `TypedDict` field may be `NotRequired`.
- **Other values:** anything that is not a `DataArray` passes through untouched.
- **One of each:** a second marker of the same kind on one hint raises `ValueError`, so write `Coords("time", "lat")`.
- **Nowhere else:** a marker `enforce` cannot apply raises `ValueError`, such as inside a `tuple[...]` return or on `Annotated[xr.Dataset, ...]`.
- **Fail fast:** a malformed declaration (a typo'd unit, an unparseable dtype or frequency) raises `ValueError` when the function is decorated.
- **Units first:** units are checked before other markers, on both sides.
- **Values never change:** `Unit` only respells a label as declared, and the other markers just check.

A few hints are silently not declarations; see [Troubleshooting](troubleshooting.md#a-declaration-is-silently-ignored).

## Units

**Inputs are checked.**
Any other unit raises: one in the same dimension (`"hPa"` where `"Pa"` is declared), so that [you decide whether to convert it](conversion.md), and a dimensionally wrong, missing or unparseable one.

**Outputs must already match.**
A return with a different unit, or none, raises, and nothing is converted or labelled for you; see [Output units](policy.md#units-outputs).
A body that changes the unit states the new one with `.assign_attrs(units=...)`.
Values are never inspected, so that statement is a claim rather than a check; see [Troubleshooting](troubleshooting.md#a-forgotten-conversion-is-not-caught).

CF-convention strings such as `"umol m-2 s-1"` need [the CF registry](registry.md).

## Which coordinate is the time axis

The time axis is the array's one datetime coordinate with a pandas index, which xarray builds by default for a dimension coordinate.
That index is in memory, so the check reads nothing from disk; building it reads the whole coordinate once, at open or at `set_xindex`.

| The array has | Checked | To change that |
|---|---|---|
| A `time` dimension | `time` | |
| A 1-D `time` with no index: `time(obs)`, `create_default_indexes=False`, `.drop_indexes("time")` | `FreqError` | `.set_xindex("time")`, which keeps the dimensions |
| A scalar `time`, as from `.isel(time=0)` | `FreqError` | `.isel(time=[0])` or `.expand_dims("time")`; a one-point axis [passes](#what-the-same-frequency-means) |
| A custom index on `time` | `FreqError` | `.drop_indexes("time").set_xindex("time")` |
| Numbers rather than datetimes, as from `decode_times=False` | `FreqError` | Decode the times |
| Two indexed datetime coordinates | `FreqError` | Name one with `Freq("7D", dim="time")` |
| `time` and another datetime coordinate with no index | `time` only | `.set_xindex("valid")` and `Freq("7D", dim="valid")` |

The last row means a scalar, 2-D or auxiliary coordinate is silently left unchecked, and a forecast with `init`, `step` and a 2-D `valid_time` has `init` checked.
`cftime` follows the same rules, but an error cannot name an unindexed one, because its dtype does not show it without reading it.

```python
times = xr.date_range("2024-01-01", periods=5, freq="D")
track = xr.DataArray([0.0] * 5, dims="obs", coords={"time": ("obs", times)})


@xra.enforce
def daily(x: Annotated[xr.DataArray, Dims("obs"), Freq("D")]) -> xr.DataArray:
    return x


daily(track.set_xindex("time"))  # daily(track) raises FreqError
```

## What "the same frequency" means

The frequency is inferred from [the time axis](#which-coordinate-is-the-time-axis) with `xarray.infer_freq`, so `cftime` calendars work too.

**Spacing** is always compared, by meaning rather than by string:

| Declared | Actual | Result | Why |
|---|---|---|---|
| `Freq("7D")` | `W-WED` | pass | Seven days either way |
| `Freq("D")` | `24h` | pass | Same fixed spacing |
| `Freq("QE")` | `3ME` | pass | A quarter is three months |
| `Freq("ME")` | `30D` | fail | Calendar months are not fixed-length days |

**Phase** is compared in two parts:

- The end or begin convention (`"ME"` vs `"MS"`) is always compared.
- The anchor (`-WED`, `-MAR`) is compared only if you spell it:

```python
Freq("W")  # weekly, any weekday
Freq("W-SUN")  # weekly, Sundays only
```

Unlike pandas, an unspelled anchor means "any" rather than a default.
`anchored=` overrides this either way: `Freq("W", anchored=True)` means Sundays, pandas' default, and `Freq("W-SUN", anchored=False)` means any weekday.

A quarterly anchor names four dates rather than one month, so `"QE-MAR"` and `"QE-DEC"` are the same, while `"QE-MAR"` and `"QE-NOV"` conflict.

Every axis is tested, whatever its length:

| The axis has | Result |
|---|---|
| Three or more timestamps, regularly spaced | Compared as above |
| Three or more timestamps, irregularly spaced, such as a missing or duplicated one | `FreqError` |
| Two timestamps | Their spacing and phase are compared directly; business-day and custom frequencies pass, having nothing to compare |
| Zero or one timestamp | Passes, as there is no spacing to contradict |

`Freq` therefore only fits an axis a pandas frequency describes.
Monthly means stamped at month midpoints are not described by one, so leave `Freq` off those.

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
