# xarray-annotated

`xarray-annotated` turns function signatures into explicit, machine-readable contracts for `DataArray` structure, units and time frequency, so you can check that the functions in a pipeline agree about the arrays passing between them.
The contracts are enforced at run time and can also be compared with each other, and they read only metadata and indexes, so a mismatch between stages or in an input can be caught before any data is loaded.

## Why?

**Why check at all?**
A mistake that raises an error gets fixed.
One that doesn't can produce plausible numbers, which can travel all the way into published results.
xarray doesn't check the dimensions, dtype, units or time axis of the arrays a function receives, and an expectation written in a docstring protects nothing.
If a mistake can be caught by an automated check, it should be.

**Why annotations?**

- **One source of truth:** an expectation is written once, in the signature, so the checks and the generated documentation cannot drift apart.
- **Machine-readable:** annotations are code, so pipeline tools, documentation generators and AI agents can all use them.
- **Usable before running:** a check inside a function body only runs when the function does, whereas a declaration can be compared with other declarations and with opened files before anything expensive starts.

## What it catches

Each example below is a short pipeline with one slip in it.
Without annotations it runs and gives a wrong answer, and with them it raises.
The tabs differ only in the annotations, and in the fix where there is one.

```python
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Coords, Dims, Dtype, Freq, Unit
```

### A unit mixed up between stages

Temperatures are converted to `degC` for a report, and the report's array is passed to a stage that needs kelvin.

=== "Without annotations"

    ```python
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant, W m-2 K-4


    def to_celsius(t):
        return (t - 273.15).assign_attrs(units="degC")


    def outgoing_longwave(t):
        """Emitted longwave radiation in W/m^2, from a temperature in K."""
        return (SIGMA * t**4).assign_attrs(units="W/m^2")


    t = xr.DataArray([273.15, 288.15, 303.15], dims="site", attrs={"units": "K"})
    t_report = to_celsius(t)
    print(outgoing_longwave(t_report).values)  # should be about [316 391 479]
    ```

    ```text
    [0.         0.00287044 0.045927  ]
    ```

=== "With annotations"

    <!-- docs-test: raises UnitError -->
    ```python
    SIGMA = 5.67e-8  # Stefan-Boltzmann constant, W m-2 K-4


    @xra.check_annotations
    def to_celsius(
        t: Annotated[xr.DataArray, Unit("K")],
    ) -> Annotated[xr.DataArray, Unit("degC")]:
        return (t - 273.15).assign_attrs(units="degC")


    @xra.check_annotations
    def outgoing_longwave(
        t: Annotated[xr.DataArray, Unit("K")],
    ) -> Annotated[xr.DataArray, Unit("W/m^2")]:
        """Emitted longwave radiation in W/m^2, from a temperature in K."""
        return (SIGMA * t**4).assign_attrs(units="W/m^2")


    t = xr.DataArray([273.15, 288.15, 303.15], dims="site", attrs={"units": "K"})
    t_report = to_celsius(t)
    print(outgoing_longwave(t_report).values)  # should be about [316 391 479]
    ```

    ```text
    UnitError: [outgoing_longwave] input 't': unit 'degC' differs from declared 'K'.
    Converting would change the values; if that is correct for this quantity, pass
    xra.units.convert(da, 'K').
    ```

### A label that no longer matches its values

xarray keeps `attrs` through arithmetic, so converting by hand leaves the old unit on the new values, and that label is what gets written to disk.

=== "Without annotations"

    ```python
    def to_pascals(p):
        return p * 100


    def altitude(p):
        """Height in m, from a pressure in Pa."""
        return (44330.0 * (1.0 - (p / 101325.0) ** 0.1903)).assign_attrs(units="m")


    p = xr.DataArray([1013.25, 900.0], dims="site", attrs={"units": "hPa"})
    p_pa = to_pascals(p)
    z = altitude(p_pa)
    print(p_pa.values, p_pa.attrs["units"])
    ```

    ```text
    [101325.  90000.] hPa
    ```

=== "With annotations"

    <!-- docs-test: raises UnitError -->
    ```python
    @xra.check_annotations
    def to_pascals(
        p: Annotated[xr.DataArray, Unit("hPa")],
    ) -> Annotated[xr.DataArray, Unit("Pa")]:
        return p * 100


    @xra.check_annotations
    def altitude(
        p: Annotated[xr.DataArray, Unit("Pa")],
    ) -> Annotated[xr.DataArray, Unit("m")]:
        """Height in m, from a pressure in Pa."""
        return (44330.0 * (1.0 - (p / 101325.0) ** 0.1903)).assign_attrs(units="m")


    p = xr.DataArray([1013.25, 900.0], dims="site", attrs={"units": "hPa"})
    p_pa = to_pascals(p)
    z = altitude(p_pa)
    print(p_pa.values, p_pa.attrs["units"])
    ```

    ```text
    UnitError: [to_pascals] output 'return': unit 'hPa' differs from declared 'Pa'. A label
    does not follow arithmetic, so first work out which unit the values are in. If 'hPa',
    and converting is correct for this quantity, use xra.units.convert(da, 'Pa'). If 'Pa',
    correct the label with .assign_attrs(units='Pa'). If neither, fix the body, as a new
    label does not change the values.
    ```

=== "Fixed"

    ```python
    @xra.check_annotations
    def to_pascals(
        p: Annotated[xr.DataArray, Unit("hPa")],
    ) -> Annotated[xr.DataArray, Unit("Pa")]:
        return xra.units.convert(p, "Pa")  # converts the values and the label


    @xra.check_annotations
    def altitude(
        p: Annotated[xr.DataArray, Unit("Pa")],
    ) -> Annotated[xr.DataArray, Unit("m")]:
        """Height in m, from a pressure in Pa."""
        return (44330.0 * (1.0 - (p / 101325.0) ** 0.1903)).assign_attrs(units="m")


    p = xr.DataArray([1013.25, 900.0], dims="site", attrs={"units": "hPa"})
    p_pa = to_pascals(p)
    z = altitude(p_pa)
    print(p_pa.values, p_pa.attrs["units"])
    ```

    ```text
    [101325.  90000.] Pa
    ```

### Dimensions that don't line up

The site elevations come from a file that calls the dimension `station`, so xarray broadcasts rather than matching sites.

=== "Without annotations"

    ```python
    def sea_level_temperature(t, z):
        """Temperature reduced to sea level, from site elevation in m."""
        return t + 0.0065 * z


    def daily_mean(x):
        return x.resample(time="D").mean()


    time = pd.date_range("2024-07-01", periods=48, freq="h")
    t = xr.DataArray(np.full((48, 3), 285.0), dims=("time", "site"), coords={"time": time})
    z = xr.DataArray([10.0, 250.0, 900.0], dims="station")
    print(daily_mean(sea_level_temperature(t, z)).sizes)
    ```

    ```text
    Frozen({'time': 2, 'site': 3, 'station': 3})
    ```

=== "With annotations"

    <!-- docs-test: raises SchemaError -->
    ```python
    @xra.check_annotations
    def sea_level_temperature(
        t: Annotated[xr.DataArray, Dims("time", "site")],
        z: Annotated[xr.DataArray, Dims("site")],
    ) -> Annotated[xr.DataArray, Dims("time", "site")]:
        """Temperature reduced to sea level, from site elevation in m."""
        return t + 0.0065 * z


    @xra.check_annotations
    def daily_mean(
        x: Annotated[xr.DataArray, Dims("time", "site")],
    ) -> Annotated[xr.DataArray, Dims("time", "site")]:
        return x.resample(time="D").mean()


    time = pd.date_range("2024-07-01", periods=48, freq="h")
    t = xr.DataArray(np.full((48, 3), 285.0), dims=("time", "site"), coords={"time": time})
    z = xr.DataArray([10.0, 250.0, 900.0], dims="station")
    print(daily_mean(sea_level_temperature(t, z)).sizes)
    ```

    ```text
    SchemaError: [sea_level_temperature] 'z' dims mismatch: expected ('site',) in any order,
    got ('station',)
    ```

### Axes in the wrong order for numpy

xarray doesn't care about dimension order, but numpy does, and this field was read in with longitude first.
It equals its latitude, so its northward gradient should be 1 everywhere.

=== "Without annotations"

    ```python
    def load_field():
        lat = np.arange(50.0, 54.0)
        return xr.DataArray(np.tile(lat, (3, 1)), dims=("lon", "lat"))


    def northward_gradient(x):
        """Change per grid step northward, for an array with dims (lat, lon)."""
        return x.copy(data=np.gradient(x.values, axis=0))


    field = load_field()
    print(northward_gradient(field).values)
    ```

    ```text
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    ```

=== "With annotations"

    <!-- docs-test: raises SchemaError -->
    ```python
    @xra.check_annotations
    def load_field() -> Annotated[xr.DataArray, Dims("lat", "lon")]:
        lat = np.arange(50.0, 54.0)
        return xr.DataArray(np.tile(lat, (3, 1)), dims=("lon", "lat"))


    @xra.check_annotations
    def northward_gradient(
        x: Annotated[xr.DataArray, Dims("lat", "lon", ordered=True)],
    ) -> Annotated[xr.DataArray, Dims("lat", "lon", ordered=True)]:
        """Change per grid step northward, for an array with dims (lat, lon)."""
        return x.copy(data=np.gradient(x.values, axis=0))


    field = load_field()
    print(northward_gradient(field).values)
    ```

    ```text
    SchemaError: [northward_gradient] 'x' dims order mismatch: expected ('lat', 'lon'), got
    ('lon', 'lat')
    ```

=== "Fixed"

    ```python
    @xra.check_annotations
    def load_field() -> Annotated[xr.DataArray, Dims("lat", "lon")]:
        lat = np.arange(50.0, 54.0)
        return xr.DataArray(np.tile(lat, (3, 1)), dims=("lon", "lat"))


    @xra.check_annotations
    def northward_gradient(
        x: Annotated[xr.DataArray, Dims("lat", "lon", ordered=True)],
    ) -> Annotated[xr.DataArray, Dims("lat", "lon", ordered=True)]:
        """Change per grid step northward, for an array with dims (lat, lon)."""
        return x.copy(data=np.gradient(x.values, axis=0))


    field = load_field().transpose("lat", "lon")
    print(northward_gradient(field).values)
    ```

    ```text
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    ```

### An integer dtype that overflows

Cloud cover is stored in eighths of the sky as `uint8`, and numpy keeps that dtype through `* 100`, which wraps around past 255.

=== "Without annotations"

    ```python
    def load_oktas():
        return xr.DataArray(np.array([0, 4, 8], dtype="uint8"), dims="time")


    def cloud_percent(oktas):
        return oktas * 100 / 8


    print(cloud_percent(load_oktas()).values)  # should be [0 50 100]
    ```

    ```text
    [ 0. 18.  4.]
    ```

=== "With annotations"

    <!-- docs-test: raises SchemaError -->
    ```python
    @xra.check_annotations
    def load_oktas() -> Annotated[xr.DataArray, Dtype("uint8")]:
        return xr.DataArray(np.array([0, 4, 8], dtype="uint8"), dims="time")


    @xra.check_annotations
    def cloud_percent(oktas: Annotated[xr.DataArray, Dtype("float")]) -> xr.DataArray:
        return oktas * 100 / 8


    print(cloud_percent(load_oktas()).values)  # should be [0 50 100]
    ```

    ```text
    SchemaError: [cloud_percent] 'oktas' dtype mismatch: expected any float, got uint8
    ```

### Monthly data into a daily stage

Monthly means are made for a plot, and that array is passed to a stage that sums daily values.

=== "Without annotations"

    ```python
    def monthly_mean(x):
        return x.resample(time="MS").mean()


    def annual_total(x):
        """Total over each year, from daily values."""
        return x.resample(time="YS").sum()


    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rain = xr.DataArray(np.full(days.size, 2.0), dims="time", coords={"time": days})
    monthly = monthly_mean(rain)
    print(annual_total(monthly).values)  # should be [732]
    ```

    ```text
    [24.]
    ```

=== "With annotations"

    <!-- docs-test: raises FreqError -->
    ```python
    @xra.check_annotations
    def monthly_mean(
        x: Annotated[xr.DataArray, Freq("D")],
    ) -> Annotated[xr.DataArray, Freq("MS")]:
        return x.resample(time="MS").mean()


    @xra.check_annotations
    def annual_total(
        x: Annotated[xr.DataArray, Freq("D")],
    ) -> Annotated[xr.DataArray, Freq("YS")]:
        """Total over each year, from daily values."""
        return x.resample(time="YS").sum()


    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rain = xr.DataArray(np.full(days.size, 2.0), dims="time", coords={"time": days})
    monthly = monthly_mean(rain)
    print(annual_total(monthly).values)  # should be [732]
    ```

    ```text
    FreqError: [annual_total] 'x' frequency mismatch on 'time': expected 'D', got 'MS'
    ```

### A file with missing hours

A rain gauge was offline for four hours, so its file has no rows for them.
Declaring what the loader returns checks the file as it is opened, reading only its metadata and time index.

??? note "Making the file"

    ```python
    import tempfile
    from pathlib import Path

    hours = pd.date_range("2024-07-01", periods=48, freq="h").delete([5, 6, 7, 8])
    path = Path(tempfile.mkdtemp()) / "rain.nc"
    xr.Dataset(
        {"rain": ("time", np.full(hours.size, 1.0), {"units": "mm"})},
        coords={"time": hours},
    ).to_netcdf(path)
    ```

=== "Without annotations"

    ```python
    def load_rain(path):
        return xr.open_dataset(path)["rain"]


    print(load_rain(path).resample(time="D").sum().values)  # should be [24 24]
    ```

    ```text
    [20. 24.]
    ```

=== "With annotations"

    <!-- docs-test: raises FreqError -->
    ```python
    @xra.check_annotations
    def load_rain(path) -> Annotated[xr.DataArray, Dims("time"), Unit("mm"), Freq("h")]:
        return xr.open_dataset(path)["rain"]


    print(load_rain(path).resample(time="D").sum().values)  # should be [24 24]
    ```

    ```text
    FreqError: [load_rain] 'return' frequency mismatch on 'time': expected 'h', got
    irregular spacing
    ```

### Arrays matched by position instead of by date

A model that works on numpy arrays returns values without their dates, and the observations start a day later, so xarray lines the two up by position.
The model is perfect, so the bias should be zero.

=== "Without annotations"

    ```python
    def run_model(forcing):
        return xr.DataArray(forcing.values * 1.0, dims="time")


    def bias(model, obs):
        return model - obs


    days = pd.date_range("2024-01-01", periods=5, freq="D")
    forcing = xr.DataArray(np.arange(5.0), dims="time", coords={"time": days})
    obs = forcing.assign_coords(time=days + pd.Timedelta("1D")) + 1.0
    print(bias(run_model(forcing), obs).values)
    ```

    ```text
    [-1. -1. -1. -1. -1.]
    ```

=== "With annotations"

    <!-- docs-test: raises SchemaError -->
    ```python
    @xra.check_annotations
    def run_model(
        forcing: Annotated[xr.DataArray, Dims("time"), Coords("time")],
    ) -> Annotated[xr.DataArray, Dims("time"), Coords("time")]:
        return xr.DataArray(forcing.values * 1.0, dims="time")


    @xra.check_annotations
    def bias(
        model: Annotated[xr.DataArray, Coords("time")],
        obs: Annotated[xr.DataArray, Coords("time")],
    ) -> xr.DataArray:
        return model - obs


    days = pd.date_range("2024-01-01", periods=5, freq="D")
    forcing = xr.DataArray(np.arange(5.0), dims="time", coords={"time": days})
    obs = forcing.assign_coords(time=days + pd.Timedelta("1D")) + 1.0
    print(bias(run_model(forcing), obs).values)
    ```

    ```text
    SchemaError: [run_model] 'return' missing coords ['time']; present: []
    ```

=== "Fixed"

    ```python
    @xra.check_annotations
    def run_model(
        forcing: Annotated[xr.DataArray, Dims("time"), Coords("time")],
    ) -> Annotated[xr.DataArray, Dims("time"), Coords("time")]:
        return forcing.copy(data=forcing.values * 1.0)  # keeps the dates


    @xra.check_annotations
    def bias(
        model: Annotated[xr.DataArray, Coords("time")],
        obs: Annotated[xr.DataArray, Coords("time")],
    ) -> xr.DataArray:
        return model - obs


    days = pd.date_range("2024-01-01", periods=5, freq="D")
    forcing = xr.DataArray(np.arange(5.0), dims="time", coords={"time": days})
    obs = forcing.assign_coords(time=days + pd.Timedelta("1D")) + 1.0
    print(bias(run_model(forcing), obs).values)
    ```

    ```text
    [0. 0. 0. 0.]
    ```

### What a check reads

A check compares a declaration with another declaration, or with an array's metadata, such as its `units` attribute.
It never reads values, so a wrong calculation with the right label passes, and needs testing like any other code.

Many checks can run pre-flight, before any data is loaded, if a computational graph can be built, as in [catena](https://github.com/NERC-CEH/catena) or [dask](https://www.dask.org/).
See [Pre-flight checks](advanced/preflight.md) for more details.

To add declarations to your own functions, start with [Getting started](getting_started.md).

## Where to next?

- [Getting started](getting_started.md): install the package and add declarations to a function.
- [User guide](guides/first_principles.md): writing declarations and how checks behave, then [schema](guides/schema.md), [units](guides/units.md) and [temporal](guides/temporal.md) declarations in detail.
- Advanced usage: [working with declarations](advanced/working_with_declarations.md) and [pre-flight checks](advanced/preflight.md).
- Alongside other tools: [pint](tools/pint.md), [jaxtyping](tools/jaxtyping.md) and [dask](tools/dask.md).
- [Worked example](example.md): an end-to-end pipeline you can run.
- Help: [troubleshooting](guides/troubleshooting.md) when something fails, and the [FAQ](faq.md) on why the package raises and what it leaves out.
- [API reference](api/package.md): exactly what each marker and function accepts.
- [CONTRIBUTING.md](https://github.com/jmarshrossney/xarray-annotated/blob/main/CONTRIBUTING.md): the package's scope and how to set up a development environment.

## Used by

- [NERC-CEH/catena](https://github.com/NERC-CEH/catena) - An opinionated integration of Apache Hamilton and xarray for building configurable environmental data pipelines.
- [NERC-CEH/xharvest](https://github.com/NERC-CEH/xharvest) - Environmental data harvester that standardises data to metadata-rich netCDF or zarr.

<!--
## Motivations

`xarray.DataArray` objects carry properties that matter for correctness but are invisible to the type system: dimensions, coordinates and dtype describe the structure of every array, and the `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`) carries a physical unit.
xarray doesn't enforce any of these, so it's common to find bugs that swap dims, feed an integer array where a float is expected, or mix up units.

Structural properties are easy to check by hand (`if array.dims != ("time", "lat", "lon"): raise ...`), but this can lead to a lot of boilerplate that's not necessarily visible in the public documentation.
For units, mercifully, a technical solution already exists in the form of pint and pint-xarray, which provide unit arithmetic and automatic conversion.
Still, there remains a gap between where expectations are declared --- a docstring, probably --- and where they're enforced, if they are at all.

Ideally these properties would be written **once**, in a format that both run-time checkers and documentation generation tools can read directly.
Storing metadata in the function signature with `typing.Annotated` is a well-trodden pattern (e.g. [Pydantic](https://docs.pydantic.dev/), [FastAPI](https://fastapi.tiangolo.com/), [Typer](https://typer.tiangolo.com/)) that does exactly this.

The same declarations help when working with AI agents: they tell an agent what each function expects, and the checks catch it when the agent gets it wrong.
Automated correctness checks are a prerequisite for semi-automated development of, and experimentation with, scientific code: if you aren't sure the outputs are correct then you have to audit every step by hand, at which point you may as well have slowed down and written it yourself.
My hope is that tools like this let scientists actually make use of AI in the areas where correctness *can* be checked mechanically.
-->
