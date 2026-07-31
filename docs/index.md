# xarray-annotated

## In a nutshell

`xarray-annotated` enables run-time validation of `xarray.DataArray` properties declared in function signatures via [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated).

The main idea is that annotations serve as a single source of truth for

- structural schema (dims, coords, dtype),
- physical units,
- frequency (and phase) of a time axis,

from which

- static checkers and run-time validation can derive expected properties,
- documentation can be generated automatically,
- automated tools (coding agents) receive steering.

## How it works

Declare properties as `typing.Annotated` metadata[^1] using typed markers --- `Dims("time", "x")`, `Coords("time")`, `Dtype("float64")`, `Unit("Pa")`, `Freq("W-SUN")` --- then apply a decorator that enforces them on every call:

```python
@declare_units
@declare_schema
def normalise(
    p: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]: ...
```

The behaviour of `declare_*` decorators (e.g. whether a violation raises, warns, or is ignored) is set through a [policy](guides/policy.md) that can be scoped to a block, function, or global.

Units can also be *converted automatically* thanks to the excellent [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io).

See [Getting started](getting_started.md) for a worked walkthrough.

## Motivations

`xarray.DataArray` objects carry properties that matter for correctness but are invisible to the type system: dimensions, coordinates, and dtype are structural assertions that every array makes, and the `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`) carries a physical unit.
xarray itself doesn't enforce any of these contracts, so it's common to find bugs that swap dims, feed an integer array where a float is expected, or mix up units.

Structural properties are easy to check (`if array.dims() != ("time", "lat", "lon"): raise ...`), but this can lead to a lot of boilerplate that's not necessarily visible in, or even compatible with, the public documentation.
For units, mercifully, a technical solution already exists in the form of [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io), which provide unit arithmetic and automatic conversion.
Still, there remains a gap between where expectations are declared --- a docstring, probably --- and where they're enforced, if they are at all.

**This package moves all of these expectations into the function signature, which becomes the single source of truth, and enforces them transparently at run time.**

Storing metadata with `typing.Annotated` is becoming standard --- it is already how [Pydantic](https://docs.pydantic.dev/), [FastAPI](https://fastapi.tiangolo.com/) and [Typer](https://typer.tiangolo.com/) work.

The main advantages of this approach are:

- API docs are generated from the function signatures, so they are guaranteed to stay in sync with the schema and units actually enforced.
- The declarations are cheap to write and cost nothing to maintain because there is only one copy.
- Annotations and automated checks provide context and guard rails for AI / agentic work.

Automated correctness checks are a prerequisite for semi-automated development of, and experimentation with, scientific code: if you aren't sure the outputs are correct then you have to audit every step by hand, at which point you may as well have slowed down and written it yourself.
My hope is that tools like this let scientists actually make use of AI in the areas where correctness *can* be checked mechanically.

## Philosophy

`xarray-annotated` is a thin layer on top of `xarray` (for data structure) and `pint`/`pint-xarray` (for units).
The aim is to do one simple job well without getting in the way of other packages and workflows.
I developed it to serve a specific purpose in my own work, and don't plan to make it significantly more complex or feature-rich.

With that out of the way, please feel free to raise an [issue](https://github.com/jmarshrossney/xarray-annotated/issues) or open a [pull request](https://github.com/jmarshrossney/xarray-annotated/pulls) to suggest a change or feature.

## See also

**Uses:**

- [xarray](https://xarray.dev/)
- [pint](https://pint.readthedocs.io/en/stable/)
- [pint-xarray](https://pint-xarray.readthedocs.io)
- [cf-xarray](https://cf-xarray.readthedocs.io/en/latest/)

**Used by:**

- [nerc-ceh/conduit](https://github.com/nerc-ceh/conduit) --- DAGs
- nerc-ceh/breadboard --- WIP
