# First principles

## Writing a declaration

A declaration is one or more markers in an [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) hint on a parameter or return value.
Decorate the function with [`xra.check_annotations`][xarray_annotated.check_annotations] and the markers are checked on every call:

```python
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Unit


@xra.check_annotations
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

## How checks behave

- **Declarations are checked at decoration:** a malformed declaration, such as a misspelled unit, raises `ValueError` when the module is imported.
- **Arrays are checked on each call:** inputs before the body, and outputs after it.
- **Each array stops at its first failure:** units are checked first, then structure, then frequency.
- **Every failed check raises:** a [`ContractError`][xarray_annotated.ContractError] and not a `ValueError`, as `UnitError`, `SchemaError` and `FreqError` are.
  There are no warnings and no lenient mode.
- **Only a `DataArray` passes:** anything else raises `ContractError`, including `None`, [unless the hint allows it](#optional-arguments-and-unions).
- **Only metadata is read:** names, dtypes, unit labels and indexes, never values, so a lazily opened array stays unread.
  A check therefore [cannot tell whether a label is true](units.md#what-a-label-cannot-catch).
- **Checks change nothing:** the body receives the arrays the caller passed, and the caller receives what the body returned, with no conversion or relabelling.
- **Only markers declare anything:** [anything else in the hint is ignored](troubleshooting.md#a-declaration-is-silently-ignored), so `Annotated[xr.DataArray, "Pa"]` checks nothing.

## Multiple returns

Return a `TypedDict` or dataclass, and declare its fields:

=== "TypedDict"

    ```python
    from typing import NotRequired, TypedDict

    import numpy as np


    class Wind(TypedDict):
        speed: Annotated[xr.DataArray, Dims("time"), Unit("m/s")]
        direction: Annotated[xr.DataArray, Dims("time"), Unit("degree")]
        gust: NotRequired[Annotated[xr.DataArray, Dims("time"), Unit("m/s")]]


    @xra.check_annotations
    def wind(
        u: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
        v: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
    ) -> Wind:
        speed = np.hypot(u, v).assign_attrs(units="m/s")
        direction = np.degrees(np.arctan2(u, v)).assign_attrs(units="degree")
        return {"speed": speed, "direction": direction}
    ```

=== "Dataclass"

    ```python
    from dataclasses import dataclass

    import numpy as np


    @dataclass
    class Wind:
        speed: Annotated[xr.DataArray, Dims("time"), Unit("m/s")]
        direction: Annotated[xr.DataArray, Dims("time"), Unit("degree")]
        gust: Annotated[xr.DataArray, Dims("time"), Unit("m/s")] | None = None


    @xra.check_annotations
    def wind(
        u: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
        v: Annotated[xr.DataArray, Dims("time"), Unit("m/s")],
    ) -> Wind:
        speed = np.hypot(u, v).assign_attrs(units="m/s")
        direction = np.degrees(np.arctan2(u, v)).assign_attrs(units="degree")
        return Wind(speed=speed, direction=direction)
    ```

- **Missing fields:** leaving out a required field raises `ContractError`, whereas a `NotRequired` field may be absent, and a dataclass field declared `| None` may be `None`.
- **Undeclared fields:** a field with no markers is not checked.
- **The container itself:** the caller receives a shallow copy of the dict or dataclass, which shares its arrays, and `__post_init__` does not run again.

A `tuple` or `NamedTuple` return has no fields `xra.check_annotations` can read, so markers there raise `ValueError` when the function is decorated.
The same goes for a parameter holding a class with declared fields, so declare a parameter's arrays as parameters of their own.

## Optional arguments and Unions

Add what else a value may be to the hint as a union, on a parameter, a return value or a field of a returned class.
Markers are checked only when the value is a `DataArray`.
With `A` standing for `Annotated[xr.DataArray, Unit("K")]`:

| Hint | `DataArray` | `None` | Anything else |
|---|---|---|---|
| `A` | checked | `ContractError` | `ContractError` |
| `A | None` | checked | passes | `ContractError` |
| `A | float` | checked | `ContractError` | passes |
| `A | float | None` | checked | passes | passes |

- **A default is not enough:** defaults are checked too, so `x: A = None` raises `ContractError` when `x` is left out; write `x: A | None = None`.
- **Other types are not checked:** `A | float` accepts a string, just as `n: int` does, so leave those to a type checker.
- **The union goes outside the `Annotated`:** `Annotated[xr.DataArray | None, Unit("K")]` and `Annotated[xr.DataArray | float, Unit("K")]` raise `ValueError` when the function is decorated.
- **Only one member can be declared:** `A | list[A]` and a union of two declared arrays raise `ValueError` when the function is decorated.
- **A returned class can only be optional:** `Wind | None` is fine, and any other union holding it raises `ValueError` when the function is decorated.

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
