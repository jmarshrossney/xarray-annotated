# Getting started

## Installation

=== "uv (recommended)"

    ```sh
    uv add xarray-annotated
    ```

=== "pip"

    ```sh
    pip install xarray-annotated
    ```

The package requires Python `>=3.12`.
CF-convention unit strings such as `"umol m-2 s-1"` need the `cf` extra, `xarray-annotated[cf]`, and [a line of setup](guides/registry.md).

## Start with a function you already have

This function computes the daily mean density of dry air.
Its expectations are in the docstring, where nothing checks them:

```python
R = 287.05  # specific gas constant of dry air, J / kg / K


def daily_air_density(p: xr.DataArray, t: xr.DataArray) -> xr.DataArray:
    """Daily mean density of dry air.

    Args:
        p: Hourly pressure in hPa, with dims (time, site).
        t: Hourly temperature in K, with dims (time, site).

    Returns:
        Daily mean density in kg/m^3, with dims (time, site).
    """
    rho = 100.0 * p / (R * t)
    return rho.resample(time="D").mean()
```

## Move the expectations into the signature

Each expectation becomes a marker in an `Annotated` hint, and `enforce` checks them on every call:

```python
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Freq, Unit


@xra.enforce
def daily_air_density(
    p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],
    t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:
    """Daily mean density of dry air."""
    rho = 100.0 * p / (R * t)
    return rho.resample(time="D").mean().assign_attrs(units="kg/m^3")  # (1)!
```

1.  The one change to the body. The return value is checked too, and xarray cannot work out its unit, so the function has to state it.

Your arrays need a `units` attribute, and the return value must end up in the declared unit.
Nothing else about calling the function changes, as a check never alters the arrays passing through it.

## See what it catches

Take `p` at 1013.25 `hPa` and `t` at 288.15 `K`, both hourly with dims `(time, site)`.
Both versions of the function return 1.225.
Now pass something slightly different:

| Pass | Before | After |
|---|---|---|
| `p` as the same pressure in `Pa` | 122.5, no error | `ValueError` |
| `t` as the same temperature in `degC` | 23.53, no error | `ValueError` |
| `t` with dims `(time, station)` | Gains a `station` dimension, no error | `SchemaError` |
| `t` every three hours | 1.225 from every third hour, no error | `FreqError` |

Each error says what is wrong and, where it can, what to do about it:

```
ValueError: [daily_air_density] input 'p': unit 'Pa' differs from declared 'hPa'.
  Converting would change the values; if that is correct for this quantity,
  pass xra.units.convert(da, 'hPa').
```

Nothing is converted for you, because whether a conversion is correct depends on the quantity; see [Converting an input](guides/units.md#converting-an-input).

## What you can declare

Declare as much or as little as you like: a marker you leave off is not checked, and neither is a parameter with no markers, so you can adopt this one function at a time.

| Marker | Passes when |
|---|---|
| [`Dims("time", "x")`][xarray_annotated.Dims] | The array has exactly these dims, in any order, or in this order with `ordered=True` |
| [`Coords("time")`][xarray_annotated.Coords] | These coordinates are present, and extras are fine |
| [`Dtype("float64")`][xarray_annotated.Dtype] | The array has this dtype, and `Dtype("float")` means any float |
| [`Unit("hPa")`][xarray_annotated.Unit] | The array is in this unit, however it is spelled |
| [`Freq("h")`][xarray_annotated.Freq] | The time axis has this spacing |

- **Only a `DataArray` passes:** anything else raises `TypeError`, including `None`, unless the hint allows it, as with `| None`.
- **A malformed declaration fails at import:** a misspelled unit raises `ValueError` when the function is decorated, not when it is called.
- **A failed check always raises:** to check nothing while prototyping, [switch the package off](guides/troubleshooting.md#enabled).

## Let pint work out the unit

`.assign_attrs` states a unit, and the output check can only read that statement, so a wrong body passes.
Computing with [pint](guides/pint.md) inside the function ties the unit to the arithmetic, and this is what we recommend wherever a body does unit arithmetic:

```python
R = xra.units.get_registry().Quantity(287.05, "J / kg / K")  # (1)!


@xra.enforce
def daily_air_density(
    p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],
    t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:
    """Daily mean density of dry air."""
    rho = xra.units.quantify(p) / (R * xra.units.quantify(t))  # (2)!
    daily = rho.resample(time="D").mean()
    return xra.units.dequantify(xra.units.convert(daily, "kg/m^3"))  # (3)!
```

1.  A constant needs a unit too, as a bare number is dimensionless.
2.  Pint works out the unit, so the factor of 100 between `hPa` and `Pa` is not written by hand.
3.  If the units of the arithmetic do not work out, this conversion raises. Callers still pass and receive plain arrays.

## Where to next

- [Declaration patterns](guides/declaring.md): several outputs, optional arrays, irregular time axes and shared declarations.
- [Units](guides/units.md) and [Computing with pint](guides/pint.md): what to do when units differ, and getting them right in the first place.
- [Worked example](example.md): an end-to-end pipeline you can run.
- [API reference](api/package.md): exactly what each marker accepts.
