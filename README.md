# xarray-annotated

[![CI](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml/badge.svg)](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/ci.yml)
[![Docs](https://github.com/jmarshrossney/xarray-annotated/actions/workflows/docs.yml/badge.svg)](https://jmarshrossney.github.io/xarray-annotated)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmarshrossney/xarray-annotated/blob/main/LICENSE)

`xarray-annotated` enables run-time validation of `xarray.DataArray` properties declared in function signatures via [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated).

The main idea is that annotations serve as a single source of truth for

- structural schema (dims, coords, dtype),
- physical units,
- frequency (and phase) of a time axis,

from which

- static checkers and run-time validation can derive expected properties,
- documentation can be generated automatically,
- automated tools (coding agents) receive steering.

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

Here we use `xarray-annotated` to declare expected properties in the signature and enforce them at run-time with minimal boilerplate (just decorators).

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dims
from xarray_annotated.units import declare_units, Unit

@declare_units
@declare_schema
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

```sh
uv sync   # install the package + dev dependencies into .venv
just      # lint, typecheck, test, build docs
```

See the `justfile` for individual targets (`just lint`, `just test`, `just test-cov`, ...).

A `.pre-commit-config.yaml` is included to run the same linting (ruff) and type-checking (pyright) steps on every commit. Install the hooks with:

```sh
uv run pre-commit install
```

Contributions are welcome --- open an issue or pull request on GitHub.
