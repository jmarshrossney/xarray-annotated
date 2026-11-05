# xarray-annotated

[![PyPI version](https://img.shields.io/pypi/v/xarray-annotated)](https://pypi.org/project/xarray-annotated/)
[![Python versions](https://img.shields.io/pypi/pyversions/xarray-annotated)](https://pypi.org/project/xarray-annotated/)
[![CI](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml/badge.svg)](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml)
[![Docs](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/docs.yml/badge.svg)](https://jmarshrossney.github.io/xarray-annotated)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmarshrossney/xarray-annotated/blob/main/LICENSE)

`xarray-annotated` turns function signatures into explicit, machine-readable contracts for `DataArray` structure, units and time frequency.
The contracts are enforced at run time, but they read only metadata and indexes, so a mismatch in an input can be caught before its data is loaded.

For full user documentation please visit **[https://jmarshrossney.github.io/xarray-annotated/](https://jmarshrossney.github.io/xarray-annotated/)**.

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


## Installation

Either using `uv` (recommended) or `pip`:

```sh
uv add xarray-annotated
# or
pip install xarray-annotated
```

## Usage

Add markers to the [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) hint of each `DataArray` parameter or return value, and decorate the function with `enforce`:

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
See [Look before you leap](https://jmarshrossney.github.io/xarray-annotated/guides/look_before_you_leap.html) for more details.

## See Also

**Uses:**

- [xarray](https://xarray.dev/)
- [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io), for unit parsing and conversion
- [cf-xarray](https://cf-xarray.readthedocs.io/en/latest/) (optional, via the `cf` extra), for UDUNITS spellings such as `"g m-2 d-1"`; see [Choosing a unit registry](https://jmarshrossney.github.io/xarray-annotated/guides/registry.html)

**Used by:**

- [NERC-CEH/catena](https://github.com/NERC-CEH/catena) - An opinionated integration of Apache Hamilton and xarray for building configurable environmental data pipelines.
- [NERC-CEH/xharvest](https://github.com/NERC-CEH/xharvest) - Environmental data harvester that standardises data to metadata-rich netCDF or zarr.

**Similar tools:**

- [xarray-schema](https://github.com/carbonplan/xarray-schema) and [xarray-validate](https://github.com/leroyvn/xarray-validate/), for validating xarray objects against a schema.
- [pint's `wraps` and `check`](https://pint.readthedocs.io/en/stable/advanced/wrapping.html), decorators that take units or dimensions as arguments rather than from annotations.
- [TypedUnit](https://github.com/MartinPdeS/TypedUnit), for checking the dimensions of pint quantities from annotations.
- [pydantic-pint](https://pypi.org/project/pydantic-pint/), for validating and converting pint quantities in Pydantic models.
- [astropy's `quantity_input`](https://docs.astropy.org/en/stable/api/astropy.units.quantity_input.html), the same idea for astropy quantities.

## Contributing

Issues and pull requests are welcome.
See [CONTRIBUTING.md](https://github.com/jmarshrossney/xarray-annotated/blob/main/CONTRIBUTING.md) for the package's scope and how to set up a development environment.
