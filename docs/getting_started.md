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


## A short example

### The problem

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
The returned altitude is numerically wrong --- sea-level pressure has come back as 26 km --- and carries a completely bogus hPa units in `attrs`.

### The fix

Here we use `xarray-annotated` to declare expected properties in the signature and enforce them at run-time with minimal boilerplate (one decorator).

```python
from typing import Annotated
import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Unit


@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Dims("time"), Unit("Pa", convert=True)],
) -> Annotated[xr.DataArray, Dims("time"), Unit("m")]:
    """Barometric altitude."""
    return 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)


altitude(p)
```
```
<xarray.DataArray (time: 2)> Size: 16B
array([  2.08162886, 110.90398059])
Dimensions without coordinates: time
Attributes:
    units:    m
```

The `hPa` input was converted to `Pa` before the body ran, and the result was stamped with the unit the signature promised.
Conversion is opt-in: without `convert=True` the `hPa` input would have raised, because a compatible unit is not always the right one.

### When a check fails

This would have failed gracefully had we passed an array whose units were _dimensionally_ wrong,

<!-- docs-test: raises DimensionalityError -->
```python
altitude(xr.DataArray([1.0, 2.0], dims=["time"], attrs={"units": "kg"}))
```
```
DimensionalityError: Cannot convert from 'kilogram' ([mass])
  to 'pascal' ([mass] / [length] / [time] ** 2)
```

or with an extra dimension,

<!-- docs-test: raises SchemaError -->
```python
altitude(xr.DataArray([[1013.0, 1000.0]], dims=["run", "time"], attrs={"units": "hPa"}))
```
```
SchemaError: [altitude] 'p' dims mismatch: expected ('time',) in any order, got ('run', 'time')
```

Whether a schema violation raises, warns or is ignored is [configurable](guides/policy.md).

Next, see [Declaring properties](guides/declaring.md) for every marker, or the [worked example](example.md) for an end-to-end pipeline.
