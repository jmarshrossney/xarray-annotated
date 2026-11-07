# Using with dask

xarray-annotated does not depend on [dask](https://www.dask.org/), and nothing on this page is required.

Checks read only metadata and indexes, so they leave a dask-backed array uncomputed, and a declared function can be called on one like any other.
Opening a file with `xr.open_dataset("in.nc", chunks={})` gives you such an array, as does `.chunk()` on one in memory:

```python
import dask
import numpy as np
import pandas as pd

time = pd.date_range("2024-07-01", periods=48, freq="h")
p = xr.DataArray(
    np.full(48, 1013.25), dims="time", coords={"time": time}, attrs={"units": "hPa"}
).chunk(time=24)


@xra.check_annotations
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa"), Freq("h")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return xra.units.convert(p, "Pa")


p_pa = to_pascals(p)  # checked, and not yet computed
```

## When checks run

Checks run when the function is called, which is not always when the work happens:

| How the function is used | Checks run | To switch them off, surround |
|---|---|---|
| Called directly on a lazy array | At the call, before anything computes | The call |
| `dask.delayed(f)(x)` | During `.compute()`, in a worker | The `.compute()` |
| `xr.map_blocks(f, x)` | [Once on an empty stand-in, then on each block](#inside-map-blocks) | The `.compute()` as well as the call |

Prefer the first, as it fails while building the graph rather than part way through computing it.
`dask.delayed` defers the check along with the call, so this raises nothing until it is computed:

```python
@xra.check_annotations
def needs_pascals(x: Annotated[xr.DataArray, Unit("Pa")]) -> xr.DataArray:
    return x


task = dask.delayed(needs_pascals)(p)
```

<!-- docs-test: raises UnitError -->
```python
task.compute()
```

[`xra.enabled`](../guides/first_principles.md#enabled) reaches dask's threaded and synchronous schedulers when it surrounds `.compute()`, but not another process, such as a distributed worker.
Set `XARRAY_ANNOTATED_ENABLED` to switch checks off there.

## What an output check sees

On a lazy output, a check reads what dask expects the result to be, which comes from whatever built the graph.
Usually that is right, but `xr.apply_ufunc` and similar calls take your word for it, and the check can only read what you told them:

```python
@xra.check_annotations
def to_single_precision(
    x: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa"), Dtype("float64")]:
    return xr.apply_ufunc(
        lambda values: values.astype("float32"),
        x,
        dask="parallelized",
        output_dtypes=[np.float64],  # untrue, and the check passes
        keep_attrs=True,
    )


out = to_single_precision(p_pa)
out.dtype, out.compute().dtype
# (dtype('float64'), dtype('float32'))
```

Keep these arguments accurate, as nothing rechecks them once the values exist.

## Which operations stay lazy { #which-operations-stay-lazy }

Arithmetic, numpy functions, `resample`, `rolling` and `where` without `drop` all build a graph without computing.
These do not, as of xarray 2026.9:

| In a body | On a dask-backed array |
|---|---|
| `.values`, `.compute()`, `float(x)`, `np.asarray(x)` | Reads the data |
| `if (x > 0).any():` | Reads the data to decide the branch |
| `x.dropna(...)` | Reads the data, as the result's shape depends on it |
| `x.where(x > 0, drop=True)`, `x.isel(time=x > 0)` | Reads the data, then raises `KeyError`. Call `.compute()` on the mask first, and the result stays lazy, for the cost of reading the mask |
| `xr.apply_ufunc(f, x)` | Raises `ValueError` until told how to handle dask, as with `dask="parallelized"` |

To find out whether a function really is lazy, make any computation raise while calling it:

```python
def refuse(*args, **kwargs):
    raise RuntimeError("a body computed")


with dask.config.set(scheduler=refuse):
    to_pascals(p)
```

## Inside `xr.map_blocks` { #inside-map-blocks }

A declared function can be passed to `xr.map_blocks`, which calls it on a zero-length stand-in while building the graph, and then on each block during `.compute()`:

```python
doubled = xr.map_blocks(to_pascals, p)  # called once, with no time steps
doubled.compute()  # called twice, with 24 time steps each
```

Each call sees only its own block, so a check cannot look across blocks.
In particular, `Freq` cannot see a gap between two of them, and it passes on a block with one time step.
Declare `Freq` on a function that receives the whole axis instead.

## Quantified arrays

[`xra.units.quantify`][xarray_annotated.units.quantify] keeps a dask-backed array lazy, while on any other array it reads the values into memory.
See [Using with pint](pint.md), which also covers the operations that [lose the unit on a dask-backed array](pint.md#what-pint-cannot-catch).

## Pre-flight checks

If a pipeline's functions all stay lazy, calling them on lazily opened files checks every input and output before any data is read.
See [Pre-flight checks](../advanced/preflight.md#mostly-lazy).
