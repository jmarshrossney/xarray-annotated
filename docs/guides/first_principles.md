# First principles

## Writing a declaration

A declaration is one or more markers in an [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) hint on a parameter or return value.
Decorate the function with [`enforce`][xarray_annotated.enforce] and the markers are checked on every call:

```python
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Unit


@xra.enforce
def mean_over_time(
    x: Annotated[xr.DataArray, Dims("time", "site"), Unit("K")],
) -> Annotated[xr.DataArray, Dims("site"), Unit("K")]:
    return x.mean("time")
```

| Marker | Passes when | More |
|---|---|---|
| [`Dims("time", "x")`][xarray_annotated.Dims] | The array has exactly these dims, in any order | [Schema](schema.md#dims) |
| [`Coords("time")`][xarray_annotated.Coords] | These coordinates are present, and extras are fine | [Schema](schema.md#coords) |
| [`Dtype("float64")`][xarray_annotated.Dtype] | The array has this dtype, and `Dtype("float")` means any float | [Schema](schema.md#dtype) |
| [`Unit("hPa")`][xarray_annotated.Unit] | The array is in this unit, however it is spelled | [Units](units.md) |
| [`Freq("h")`][xarray_annotated.Freq] | The time axis has this spacing | [Temporal](temporal.md) |

Declare as much or as little as you like: a marker you leave off is not checked, and neither is a parameter with no markers, so you can adopt this one function at a time.

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

A dataclass works the same way:

```python
from dataclasses import dataclass


@dataclass
class WindSummary:
    speed: Annotated[xr.DataArray, Dims("time"), Unit("m/s")]
    direction: Annotated[xr.DataArray, Dims("time"), Unit("degree")]


@xra.enforce
def wind_summary(
    u: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
    v: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
) -> WindSummary:
    speed = np.hypot(u, v).assign_attrs(units="m/s")
    direction = np.degrees(np.arctan2(u, v)).assign_attrs(units="degree")
    return WindSummary(speed=speed, direction=direction)
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

## How checks behave

| Behaviour | Detail |
|---|---|
| Declarations are checked at decoration | A malformed declaration, such as a misspelled unit, raises `ValueError` when the module is imported |
| Arrays are checked on each call | Inputs before the body, and outputs after it |
| Each array stops at its first failure | Units are checked first, then structure, then frequency |
| Every failed check raises | `UnitError`, `SchemaError` or `FreqError`, each a [`ContractError`][xarray_annotated.ContractError] and not a `ValueError`. There are no warnings and no lenient mode |
| Only a `DataArray` passes | Anything else raises `TypeError`, including `None`, unless the hint allows it |
| Only metadata is read | Names, dtypes, unit labels and indexes, never values, so a lazily opened array stays unread. A check therefore [cannot tell whether a label is true](units.md#what-a-label-cannot-catch) |
| Checks change nothing | The body receives the arrays the caller passed, and the caller receives what the body returned, with no conversion or relabelling |
| Only markers declare anything | [Anything else in the hint is ignored](troubleshooting.md#a-declaration-is-silently-ignored), so `Annotated[xr.DataArray, "Pa"]` checks nothing |

## Switching the package off { #enabled }

While prototyping, switch the package off:

```python
with xra.enabled(False):
    ...
```

Off is a total no-op, so a function declaring `Unit("Pa")` can receive `hPa`.
Malformed declarations still raise when a function is decorated.

- **For a process:** set `XARRAY_ANNOTATED_ENABLED` to `0`, `false`, `no` or `off`.
- **Precedence:** a block beats the variable in both directions, and the innermost block wins.
- **Scope:** a block applies to the current thread or async task only, and to dask's threaded and synchronous schedulers when it surrounds `.compute()`. It does not reach a new thread or another process.
