# xarray-annotated

`xarray-annotated` turns function signatures into explicit, machine-readable contracts for `DataArray` structure, units and time frequency.
The contracts are enforced at run time, but they read only metadata and indexes, so a mismatch in an input can be caught before its data is loaded.

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

## How?

Add markers to the [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) hint of each `DataArray` parameter or return value, and decorate the function with `enforce`:

=== "Plain xarray"

    ```python
    import xarray as xr

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

=== "With annotations"

    ```python
    from typing import Annotated

    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Dims, Freq, Unit

    R = 287.05  # specific gas constant of dry air, J / kg / K


    @xra.enforce
    def daily_air_density(
        p: Annotated[xr.DataArray, Dims("time", "site"), Unit("hPa"), Freq("h")],
        t: Annotated[xr.DataArray, Dims("time", "site"), Unit("K"), Freq("h")],
    ) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kg/m^3"), Freq("D")]:
        """Daily mean density of dry air."""
        rho = 100.0 * p / (R * t)
        return rho.resample(time="D").mean().assign_attrs(units="kg/m^3")
    ```

Anything that does not match its declaration raises.

Many checks can run pre-flight, before any data is loaded, if a computational graph can be built, as in [catena](https://github.com/NERC-CEH/catena) or [dask](https://www.dask.org/).
See [Pre-flight checks](advanced/preflight.md) for more details.

## Where to next?

- [Getting started](getting_started.md): install the package and add declarations to a function.
- [User guide](guides/first_principles.md): writing declarations and how checks behave, then [schema](guides/schema.md), [units](guides/units.md) and [temporal](guides/temporal.md) declarations in detail.
- Advanced usage: [accessing declarations](advanced/accessing_declarations.md) and [pre-flight checks](advanced/preflight.md).
- Integrations: [pint](integrations/pint.md) and [jaxtyping](integrations/jaxtyping.md).
- [Worked example](example.md): an end-to-end pipeline you can run.
- [FAQ](faq.md): why the package raises, and what it leaves out.
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
