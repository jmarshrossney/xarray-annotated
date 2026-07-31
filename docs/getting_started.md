# Getting Started

## Installation

`xarray-annotated` can be installed directly from GitHub using `pip`, or tools such as `uv` that wrap around it.

=== "uv (recommended)"

    ```sh
    uv add git+https://github.com/jmarshrossney/xarray-annotated.git
    ```

=== "pip"

    ```sh
    pip install git+https://github.com/jmarshrossney/xarray-annotated.git
    ```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) need the optional `cf` extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io).
It also needs activating at startup --- see [Choosing a unit registry](guides/registry.md).

=== "uv (recommended)"

    ```sh
    uv add "xarray-annotated[cf] @ git+https://github.com/jmarshrossney/xarray-annotated.git"
    ```

=== "pip"

    ```sh
    pip install "xarray-annotated[cf] @ git+https://github.com/jmarshrossney/xarray-annotated.git"
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

### The catch

This would have failed gracefully had we passed an array whose units were _dimensionally_ wrong,

```python
altitude(xr.DataArray([1.0, 2.0], dims=["time"], attrs={"units": "kg"}))
```
```
DimensionalityError: Cannot convert from 'kilogram' ([mass])
  to 'pascal' ([mass] / [length] / [time] ** 2)
```

or with an extra dimension,

```python
altitude(xr.DataArray([[1013.0, 1000.0]], dims=["run", "time"], attrs={"units": "hPa"}))
```
```
SchemaError: [altitude] 'p' dims mismatch: expected ('time',) in any order, got ('run', 'time')
```

Whether a schema violation raises, warns or is ignored is set by a [policy](guides/policy.md).

## Where to go next

- How-to guides:
    - **[Declaring properties](guides/declaring.md)** --- the full set of markers (dims, coords, dtype, units, frequency), what each one considers a match, and how to combine them.
    - **[Configuring validation](guides/policy.md)** --- raise, warn or ignore; scoping overrides to a block or a function; turning validation off in production.
    - **[Choosing a unit registry](guides/registry.md)** --- needed only if you use CF-convention unit strings such as `"umol m-2 s-1"`.
    - **[Reading and writing declarations](guides/tooling.md)** --- inspecting declarations programmatically, and generating documentation from them.
    - **[Troubleshooting](guides/troubleshooting.md)** --- common errors and what they mean.
- **[A worked pipeline](example.md)** --- the same ideas applied end-to-end, as a runnable notebook.
- **[API reference](api/package.md)** --- every marker, decorator and exception, generated from the source.
