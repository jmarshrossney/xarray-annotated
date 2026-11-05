# Getting Started

## Installation

`xarray-annotated` can be installed from PyPI using `pip`, or tools such as `uv` that wrap around it.

=== "uv (recommended)"

    ```sh
    uv add xarray-annotated
    ```

=== "pip"

    ```sh
    pip install xarray-annotated
    ```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) need the optional `cf` extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io).
It also needs activating at startup --- see [Choosing a unit registry](guides/registry.md).

=== "uv (recommended)"

    ```sh
    uv add "xarray-annotated[cf]"
    ```

=== "pip"

    ```sh
    pip install "xarray-annotated[cf]"
    ```

The package requires Python `>=3.12`.
Python 3.12, 3.13 and 3.14 are tested in CI.


## Before and after

Here is a function that computes the daily mean density of dry air from pressure and temperature, written three ways: with its expectations in the docstring, where nothing checks them; with them in the signature, on arrays that carry their unit in `attrs`; and with them in the signature, on arrays quantified with [pint](guides/pint.md).
Each tab is a complete script that you can copy and run.

=== "xarray"

    ```python
    import numpy as np
    import pandas as pd
    import xarray as xr

    R = 287.05  # specific gas constant of dry air, J / kg / K


    def hourly(value, units, dims=("time", "site"), freq="h"):
        time = pd.date_range("2024-07-01", "2024-07-02T23:00", freq=freq)
        return xr.DataArray(
            np.full((time.size, 2), value),
            dims=dims,
            coords={"time": time},
            attrs={"units": units},
        )


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


    p = hourly(1013.25, "hPa")
    t = hourly(288.15, "K")
    print(daily_air_density(p, t).round(3))
    ```
    ```
    <xarray.DataArray (time: 2, site: 2)> Size: 32B
    array([[1.225, 1.225],
           [1.225, 1.225]])
    Coordinates:
      * time     (time) datetime64[us] 16B 2024-07-01 2024-07-02
    Dimensions without coordinates: site
    ```

=== "xarray-annotated"

    ```python
    from typing import Annotated

    import numpy as np
    import pandas as pd
    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Dims, Freq, Unit

    R = 287.05  # specific gas constant of dry air, J / kg / K


    def hourly(value, units, dims=("time", "site"), freq="h"):
        time = pd.date_range("2024-07-01", "2024-07-02T23:00", freq=freq)
        return xr.DataArray(
            np.full((time.size, 2), value),
            dims=dims,
            coords={"time": time},
            attrs={"units": units},
        )


    @xra.enforce  # (1)!
    def daily_air_density(
        p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],  # (2)!
        t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
    ) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:  # (3)!
        """Daily mean density of dry air."""
        rho = 100.0 * p / (R * t)
        return rho.resample(time="D").mean().assign_attrs(units="kg/m^3")  # (4)!


    p = hourly(1013.25, "hPa")
    t = hourly(288.15, "K")
    print(daily_air_density(p, t).round(3))
    ```

    1.  `enforce` reads the declarations from the signature and checks them on every call.
    2.  These dimensions must be present, in any order, and no others. The unit must be `hPa`, and the `time` coordinate must be hourly.
    3.  The return value is checked too, and has to match.
    4.  xarray would otherwise leave the `hPa` label on the result, so the body states its unit.

    ```
    <xarray.DataArray (time: 2, site: 2)> Size: 32B
    array([[1.225, 1.225],
           [1.225, 1.225]])
    Coordinates:
      * time     (time) datetime64[us] 16B 2024-07-01 2024-07-02
    Dimensions without coordinates: site
    Attributes:
        units:    kg/m^3
    ```

=== "xarray-annotated + pint"

    ```python
    from typing import Annotated

    import numpy as np
    import pandas as pd
    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Dims, Freq, Unit

    R = xra.units.get_registry().Quantity(287.05, "J / kg / K")  # (1)!


    def hourly(value, units, dims=("time", "site"), freq="h"):
        time = pd.date_range("2024-07-01", "2024-07-02T23:00", freq=freq)
        return xr.DataArray(
            np.full((time.size, 2), value),
            dims=dims,
            coords={"time": time},
            attrs={"units": units},
        )


    @xra.enforce
    def daily_air_density(
        p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],
        t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
    ) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:
        """Daily mean density of dry air."""
        rho = p / (R * t)  # (2)!
        return xra.units.convert(rho.resample(time="D").mean(), "kg/m^3")  # (3)!


    p = xra.units.quantify(hourly(1013.25, "hPa"))  # (4)!
    t = xra.units.quantify(hourly(288.15, "K"))
    print(daily_air_density(p, t).round(3))
    ```

    1.  A constant needs a unit too, as a bare number is dimensionless.
    2.  Pint works out the unit, so the factor of 100 between `hPa` and `Pa` is not written by hand.
    3.  The result is in `hPa kg / J`, which is a density but not yet the declared unit.
    4.  The arrays are quantified before they reach the function. `enforce` only checks, so the body receives them as they are.

    ```
    <xarray.DataArray (time: 2, site: 2)> Size: 32B
    Quantity(array([[1.225, 1.225],
           [1.225, 1.225]]), "kilogram / meter ** 3")
    Coordinates:
      * time     (time) datetime64[us] 16B 2024-07-01 2024-07-02
    Dimensions without coordinates: site
    ```

Whether units live in `attrs` or in pint-quantified arrays is a decision for the whole workflow, rather than for one function.
We [recommend pint where it is feasible](guides/pint.md), because a unit that was computed is worth more than one that was assigned by hand.

## Now break it

Change one line of the script, and see what each version does:

| Change | xarray | xarray-annotated, with or without pint |
|---|---|---|
| `hourly(101325.0, "Pa")` for `p` | 122.5, no error | `ValueError` |
| `hourly(15.0, "degC")` for `t` | 23.53, no error | `ValueError` |
| `hourly(288.15, "m")` for `t` | 1.225, no error | `DimensionalityError` |
| `hourly(288.15, "K", dims=("time", "station"))` for `t` | Gains a `station` dimension, no error | `SchemaError` |
| `hourly(288.15, "K", freq="3h")` for `t` | 1.225 from every third hour, no error | `FreqError` |

Each error says what is wrong and, where it can, what to do about it:

```
ValueError: [daily_air_density] input 'p': unit 'Pa' differs from declared 'hPa'.
  Converting would change the values; if that is correct for this quantity,
  pass xra.units.convert(da, 'hPa').
```

Pressure is safe to [convert](guides/units.md#converting-an-input), but nothing is converted for you, because whether a conversion is correct depends on the quantity.

A violation always raises; to check nothing while prototyping, [switch the package off](guides/troubleshooting.md#enabled).

Next, see [Declaring properties](guides/declaring.md) for every marker, including [how to shorten long signatures](guides/declaring.md#reusing-a-declaration), [Computing with pint](guides/pint.md), or the [worked example](example.md) for an end-to-end pipeline.
