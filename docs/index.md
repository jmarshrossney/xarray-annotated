# xarray-annotated

`xarray-annotated` checks `xarray.DataArray` arguments and return values against properties declared in the function signature with [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated):

- **structure**: dimensions, coordinates and dtype;
- **physical units**;
- **time frequency** and phase.

Making signatures the single source of truth has several advantages: the same declaration

- can be used by automated checks and validation, and hence enforced at run time;
- appears in generated documentation without needing to be hand-written;
- provides rich context for AI coding agents.


## Installation

Either using `uv` (recommended) or `pip`:

```sh
uv add xarray-annotated
# or
pip install xarray-annotated
```

## How it works

Add markers to the `Annotated` hint of each `DataArray` parameter or return value, and decorate the function with `enforce`:

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

See [Getting started](getting_started.md) for a worked walkthrough, [Declaring properties](guides/declaring.md) for each marker in detail, [Units](guides/units.md) for what to do when the units differ, and [Computing with pint](guides/pint.md) for getting the units right in the first place.

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

## Contributing

Issues and pull requests are welcome.
See [CONTRIBUTING.md](https://github.com/jmarshrossney/xarray-annotated/blob/main/CONTRIBUTING.md) for the package's scope and how to set up a development environment.

## See also

**Uses:**

- [xarray](https://xarray.dev/)
- [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io), for unit parsing and conversion
- [cf-xarray](https://cf-xarray.readthedocs.io/en/latest/) (optional, via the `cf` extra), for UDUNITS spellings such as `"g m-2 d-1"`; see [Choosing a unit registry](guides/registry.md)

**Used by:**

- [NERC-CEH/conduit](https://github.com/NERC-CEH/conduit) - An opinionated integration of Apache Hamilton and xarray for building configurable environmental data pipelines.
- [NERC-CEH/xharvest](https://github.com/NERC-CEH/xharvest) - Environmental data harvester that standardises data to metadata-rich netCDF or zarr.
