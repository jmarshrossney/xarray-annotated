# Home

`xarray-annotated` enables run-time validation of `xarray.DataArray` properties
declared in function signatures via
[`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated).
A DataArray has several properties that matter for correctness — its dimensions,
coordinates, dtype, physical units, and the frequency of its time axis — and
`xarray-annotated` lets you declare all of them in one place and validate them
automatically.

!!! note

    Imports are namespaced into three domains, imported separately:
    `xarray_annotated.schema` (structural properties — dims, coords, dtype;
    validate-only, never mutates), `xarray_annotated.units` (physical units,
    checked *and converted* via pint/CF), and `xarray_annotated.temporal` (the
    frequency and phase of a time axis; validate-only). They share the same
    `typing.Annotated` mechanism and a common validation policy, so the decorators
    compose and toggle together. See the [Usage](usage.md) guide for the full
    walkthrough.

For example:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import declare_units, Unit
from xarray_annotated.schema import declare_schema, Dims, Dtype

@declare_units
@declare_schema
def normalise_pressure(
    p: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p

p = xr.DataArray([[1013.0, 1000.0]], dims=["time", "x"], attrs={"units": "hPa"})
normalise_pressure(p)
# <xarray.DataArray (time: 1, x: 2)>
# array([[101300., 100000.]])
# Attributes:
#     units:    Pa
```

## Motivations

`xarray.DataArray` objects carry properties that matter for correctness but are invisible to
the type system: dimensions, coordinates, and dtype are structural assertions every array makes,
and the `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`) carries a physical unit. xarray
itself doesn't enforce any of these contracts at call sites — it's easy to swap dims, feed an
integer array where a float is expected, or mistakenly mix `hPa`/`Pa`.

Existing tools address pieces of this: [pint](https://pint.readthedocs.io/en/stable/) and
[pint-xarray](https://pint-xarray.readthedocs.io) provide unit arithmetic and conversion but
don't *declare* or *enforce* expectations in signatures; for structural properties there is no
equivalent. `xarray-annotated` fills the gap — it moves all of these expectations into the
function signature, the single source of truth, and enforces them transparently at run time
under a swtichable policy:

1. **Declare properties in one place** — `Annotated[xr.DataArray, Dims("time", "x"),
   Dtype("float64"), Unit("Pa")]` — so the contract is written once, in the signature.
2. **Validate automatically** via `@declare_schema` (structural: dims, coords, dtype) and
   `@declare_units` (physical units, via pint/CF), or their public primitives `check_schema`
   / `check_units` when a decorator doesn't fit.

Attaching metadata to type hints with `typing.Annotated` is an increasingly common pattern in
the Python ecosystem, used by libraries such as [Pydantic](https://docs.pydantic.dev/),
[FastAPI](https://fastapi.tiangolo.com/) and [Typer](https://typer.tiangolo.com/).
Declared properties sit naturally alongside other typed metadata rather than in a separate
schema.

## Installation

`xarray-annotated` can be installed directly from GitHub using `pip`, or tools
such as `uv` that wrap around it.

=== "uv (recommended)"

    ```sh
    uv add git+https://github.com/jmarshrossney/xarray-annotated.git
    ```

=== "pip"

    ```sh
    pip install git+https://github.com/jmarshrossney/xarray-annotated.git
    ```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) need the optional `cf`
extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io):

=== "uv (recommended)"

    ```sh
    uv add "xarray-annotated[cf] @ git+https://github.com/jmarshrossney/xarray-annotated.git"
    ```

=== "pip"

    ```sh
    pip install "xarray-annotated[cf] @ git+https://github.com/jmarshrossney/xarray-annotated.git"
    ```

Currently Python versions equal to or above 3.12 are supported.

## Overview of usage

The workflow is the same for every property.

1. **Declare** properties in the signature with typed markers inside `Annotated` —
   `Unit("Pa")`, `Dims("time", "x")`, `Dtype("float64")`, `Coords("time")`. A single
   hint can carry several markers at once.
2. **Apply** the declarations — decorate with `@declare_schema` (validates dims, coords,
   dtype; passes through unchanged) and/or `@declare_units` (validates + converts units;
   stamps outputs). Stack both to check everything, or call `check_schema(...)` /
   `check_units(...)` directly where a decorator doesn't fit.

See the [Usage](usage.md) guide for the full walkthrough — the validation policy, choosing a pint
vs. CF registry, reading declarations off a signature for your own tooling, and the cross-domain
`annotate` / `declarations_from_signature` API — and the
[API reference](api/package.md) for the complete surface.

## Philosophy

`xarray-annotated` is a deliberately thin validation layer: it adds property *declaration* (via
`Annotated` markers) and *enforcement* (via decorators and policy) on top of the libraries that
already handle the heavy lifting — pint/pint-xarray for unit arithmetic, xarray for data
structures. It is not a units engine, a type checker, or a general accessor; those spaces are
already well served. Schema is validate-only by design (it asserts structural properties without
ever converting or mutating), while units validates *and* converts. Both domains share the same
annotation mechanism and the same global policy switch, so they compose cleanly and toggle
together.

This is by design: the aim is to work seamlessly alongside those tools without ever getting in
the way. I developed it to serve a specific purpose in my own work, and don't plan to make it
significantly more complex or feature-rich — but please feel free to raise an
[issue](https://github.com/jmarshrossney/xarray-annotated/issues) or open a
[pull request](https://github.com/jmarshrossney/xarray-annotated/pulls) to suggest a change
or feature.
