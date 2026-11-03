# xarray-annotated

[![PyPI version](https://img.shields.io/pypi/v/xarray-annotated)](https://pypi.org/project/xarray-annotated/)
[![Python versions](https://img.shields.io/pypi/pyversions/xarray-annotated)](https://pypi.org/project/xarray-annotated/)
[![CI](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml/badge.svg)](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml)
[![Docs](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/docs.yml/badge.svg)](https://jmarshrossney.github.io/xarray-annotated)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmarshrossney/xarray-annotated/blob/main/LICENSE)

`xarray-annotated` checks `xarray.DataArray` arguments and return values against properties declared in the function signature with [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated):

- **structure**: dimensions, coordinates and dtype;
- **physical units**, converting inputs to the declared unit where needed;
- **time frequency** and phase.

Making signatures the single source of truth has several advantages: the same declaration

- can be used by automated checks and validation, and hence enforced at run time;
- appears in generated documentation without needing to be hand-written;
- provides rich context for AI coding agents.

For validating and automatically converting physical units we lean on the excellent [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io).
([cf-xarray](https://cf-xarray.readthedocs.io) is an optional dependency for CF/UDUNITS unit strings.)

For full user documentation please visit **[https://jmarshrossney.github.io/xarray-annotated/](https://jmarshrossney.github.io/xarray-annotated/)**.

## Installation

Either using `uv` (recommended) or `pip`:

```sh
uv add xarray-annotated
# or
pip install xarray-annotated
```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) need the optional `cf` extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io).

```sh
uv add "xarray-annotated[cf]"
# or
pip install "xarray-annotated[cf]"
```

## A short example

Here is a function that converts barometric pressure to altitude.
It assumes a unit (pascals) and says so in the docstring but doesn't check or enforce it.

```python
import xarray as xr


def altitude(p: xr.DataArray) -> xr.DataArray:
    """Barometric altitude. Assumes p is in Pa."""
    return 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)


p = xr.DataArray([1013.0, 1000.0], dims=["time"], attrs={"units": "hPa"})
altitude(p)
```
```
<xarray.DataArray (time: 2)> Size: 16B
array([25876.55998663, 25921.86206292])
Dimensions without coordinates: time
Attributes:
    units:    hPa
```

The input units are wrong (hPa vs Pa) but the onus is on the user to check, and in reality we all know that such bugs can go unnoticed for a terrifyingly long time.
The returned altitude is numerically wrong and carries a completely bogus hPa units in `attrs`.

Here we use `xarray-annotated` to declare expected properties in the signature and enforce them at run-time with minimal boilerplate (one decorator).

```python
from typing import Annotated
import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Unit


@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Dims("time"), Unit("Pa")],
) -> Annotated[xr.DataArray, Dims("time"), Unit("m")]:
    """Barometric altitude."""
    return 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)


p = xr.DataArray([1013.0, 1000.0], dims=["time"], attrs={"units": "hPa"})
altitude(p)
```
```
<xarray.DataArray (time: 2)> Size: 16B
array([  2.08162886, 110.90398059])
Dimensions without coordinates: time
Attributes:
    units:    m
```

Now the outputs are physical and carry the correct units attribute.
The `hPa` input was converted to `Pa` before the body ran, and the result was stamped with the unit the signature promised.
The docstring no longer needs to say "assumes Pa", because the signature does --- and unlike the docstring, it is actually enforced.

If we had handed the function an array whose units were _dimensionally_ wrong, or with an extra dimension, this would have failed gracefully:

```python
altitude(xr.DataArray([1.0, 2.0], dims=["time"], attrs={"units": "kg"}))
# DimensionalityError: Cannot convert from 'kilogram' ([mass])
#   to 'pascal' ([mass] / [length] / [time] ** 2)

altitude(xr.DataArray([[1013.0, 1000.0]], dims=["run", "time"], attrs={"units": "hPa"}))
# SchemaError: [altitude] 'p' dims mismatch: expected ('time',) in any order, got ('run', 'time')
```

## Contributing

Issues and pull requests are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the package's scope and how to set up a development environment.
