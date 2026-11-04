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
        Unit("degC", convert=True),
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
| `Unit(unit)` | The array is in this unit, however it is spelled | `convert=True`, `on_output=` |
| `Freq(freq)` | The time axis has this spacing, and this phase [where spelled](#what-the-same-frequency-means) | `dim=`, `anchored=` |

`Dims`, `Coords`, `Dtype` and `Freq` also take `on_mismatch=`; see [Configuring validation](policy.md).

## Rules

- **Where markers go:** on a parameter, on a return, or on the fields of a `TypedDict` or dataclass return.
- **Optional values:** `Annotated[xr.DataArray | None, ...]` and `Annotated[xr.DataArray, ...] | None` are equivalent, and a `TypedDict` field may be `NotRequired`.
- **Other values:** anything that is not a `DataArray` passes through untouched.
- **One of each:** a second marker of the same kind on one hint raises `ValueError`, so write `Coords("time", "lat")`.
- **Nowhere else:** a marker `enforce` cannot apply raises `ValueError`, such as inside a `tuple[...]` return or on `Annotated[xr.Dataset, ...]`.
- **Fail fast:** a malformed declaration (a typo'd unit, an unparseable dtype or frequency) raises `ValueError` when the function is decorated.
- **Units first:** inputs are converted before their other markers are checked, and output units are applied before theirs, so checks see what the body and the caller see.
- **Only `Unit` changes an array:** the other markers just check it.

A few hints are silently not declarations; see [Troubleshooting](troubleshooting.md#a-declaration-is-silently-ignored).

## Units

**Inputs are checked.**
A different unit in the same dimension (`"hPa"` where `"Pa"` is declared) raises unless the declaration says `convert=True`, and a dimensionally wrong, missing or unparseable unit always raises.

**Outputs are stamped.**
A declared return has `attrs["units"]` overwritten with the declared unit, and its values are never inspected.
Everything downstream then compares against that declaration, so a return's `Unit` is a claim rather than a check; see [Troubleshooting](troubleshooting.md#a-forgotten-conversion-is-not-caught).
[`on_output`](policy.md#units-on_output) tightens this.

CF-convention strings such as `"umol m-2 s-1"` need [the CF registry](registry.md).

## What "the same frequency" means

The frequency is inferred from the time coordinate with `xarray.infer_freq`, so `cftime` calendars work too.
The time axis is the array's only datetime-like coordinate; if it has two, name one with `Freq("7D", dim="time")`.

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

An axis with fewer than three timestamps or irregular spacing has no inferable frequency, so by default it warns instead of raising.

## Reusing a declaration

Name a declaration with a `type` statement or an assignment:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]
```

Wrap a named declaration to add markers to it:

```python
type TypedPressure = Annotated[Pressure, Dtype("float64")]
```

It declares `Unit`, `Dims` and `Dtype`.
A marker cannot be replaced this way, so a second `Unit` raises `ValueError`.
