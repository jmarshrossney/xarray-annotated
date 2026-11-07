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
CF-convention unit strings such as `"umol m-2 s-1"` need the `cf` extra, `xarray-annotated[cf]`, and [a line of setup](guides/units.md#cf-convention-units).

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

Each expectation becomes a marker in an `Annotated` hint, and `xra.check_annotations` checks them on every call:

```python
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Freq, Unit


@xra.check_annotations
def daily_air_density(
    p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],
    t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:
    """Daily mean density of dry air."""
    rho = 100.0 * p / (R * t)
    return rho.resample(time="D").mean().assign_attrs(units="kg/m^3")  # (1)!
```

1.  The one change to the body. The return value is checked too, and xarray cannot work out its unit, so the function has to state it. See [Assigning units on output](guides/units.md#outputs).

Your arrays need a `units` attribute, and the return value must end up in the declared unit.
Nothing else about calling the function changes, as a check never alters the arrays passing through it.

## See what it catches

Take `p` at 1013.25 `hPa` and `t` at 288.15 `K`, both hourly with dims `(time, site)`.
Both versions of the function return 1.225.
Now pass something slightly different:

| Pass | Before | After |
|---|---|---|
| `p` as the same pressure in `Pa` | 122.5, no error | `UnitError` |
| `t` as the same temperature in `degC` | 23.53, no error | `UnitError` |
| `t` with dims `(time, station)` | Gains a `station` dimension, no error | `SchemaError` |
| `t` every three hours | 1.225 from every third hour, no error | `FreqError` |

Each error says what is wrong and, where it can, what to do about it:

```
UnitError: [daily_air_density] input 'p': unit 'Pa' differs from declared 'hPa'.
  Converting would change the values; if that is correct for this quantity,
  pass xra.units.convert(da, 'hPa').
```

Nothing is converted for you, because whether a conversion is correct depends on the quantity; see [Converting an input](guides/units.md#converting-an-input).

## Where to next

Read the user guide in order:

1. [First principles](guides/first_principles.md): writing a declaration, how checks behave, multiple returns, optional arguments and unions, and how to switch the package off.
2. [Schema](guides/schema.md), [Units](guides/units.md) and [Temporal](guides/temporal.md) declarations: each kind of marker in more detail, as you need it.

The [worked example](example.md) shows all of this in an end-to-end pipeline you can run.
