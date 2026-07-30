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

For validating and automatically converting physical units we lean on the excellent [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io).
([cf-xarray](https://cf-xarray.readthedocs.io) is an optional dependency for CF/UDUNITS unit strings.)

 
## How it works


Declare properties as `typing.Annotated` metadata using typed markers — `Dims("time", "x")`, `Coords("time")`, `Dtype("float64")`, `Unit("Pa")`, `Freq("W-SUN")` — then apply a decorator that enforces them on every call:

```python
@declare_units
@declare_schema
def normalise(
    p: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]: ...
```

Structural and temporal markers only ever *check*; arrays pass through unchanged.
Units are also *converted*, so a caller handing you `hPa` gets a conversion rather than a rejection, and the output is stamped with the unit you promised.
Whether a violation raises, warns, or is ignored is set by a [policy](guides/policy.md) that can be scoped to a block, a function, or a single marker, and switched off entirely in production.

Because the declarations *are* the signature, they cannot drift from what is enforced — anything that renders signatures renders them too, including this site's [API reference](api/package.md), and [`declarations_from_signature`](guides/tooling.md#cross-domain-reader-declarations_from_signature) exposes them to tooling without calling anything.

See [Getting started](getting_started.md) for a worked walkthrough.


## Motivations

It's not always possible to automate correctness checks, but sometimes it is, so why wouldn't you?

`xarray.DataArray` objects carry properties that matter for correctness but are invisible to the type system: dimensions, coordinates, and dtype are structural assertions that every array makes, and the `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`) carries a physical unit.
xarray itself doesn't enforce any of these contracts so it's common to find bugs swap dims, feed an integer array where a float is expected, or mixing up units.

Regarding units, mercifully, a technical solution already exists in the form of [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io), which provide unit arithmetic and automatic conversion.
Still, there remains a gap between where expectations are declared (probably in docstrings) and where they are enforced in the code (if they are at all), which requires a level of diligence from developers and users that, let's face it, is often not met.

The story is similar for structural properties: dimensions, coordinates, data type.
They're technically very easy to check (`if array.dims() != ("time", "lat", "lon"): raise Exception`) but this can lead to a lot of boilerplate that's not necessarily visible in, or even compatible with, the public documentation.

**This package moves all of these expectations into the function signature, which becomes the single source of truth, and enforces them transparently at run time.**

The main advantages of this approach are:

- Easy, low-maintenance for developers.
- Generate API docs from function signatures, so the API docs are guaranteed to stay in sync with the schema/units actually enforced.
- It is becoming standard to store metadata with `typing.Annotated` --- already used by libraries such as [Pydantic](https://docs.pydantic.dev/), [FastAPI](https://fastapi.tiangolo.com/) and [Typer](https://typer.tiangolo.com/).
- Provides important context and validation tools for AI / agentic work.[^foot_ai]

[^foot_ai]: 
  Automatic correctness checks are a prerequisite for semi-automated development of, and experimentation with, scientific codes.
  If you're not sure that the outputs are correct, then either you need to go through and understand/audit each step (in which case it would have been better to slow down and write it yourself), or frankly it's worthless.
  My hope is that tools like this actually allow scientists to make use of AI in areas where correctness checks can be automated.

I originally wrote `xarray-annotated` for [nerc-ceh/conduit](https://github.com/nerc-ceh/conduit), which uses [Hamilton](https://hamilton.dagworks.io/) to build Directed Acyclic Graphs where every node (python function) consumes and returns `DataArray`s.
The most useful thing about signature annotations is that it allows us to extend Hamilton's node compatibility checks to include schema, units, frequencies etc, all by static analysis, before any data is loaded or function called.
A DAG of a hundred nodes can be checked end to end in the time it takes to import it, which is the difference between finding an error in CI and finding it three hours into a run.

## Philosophy

`xarray-annotated` is a thin layer on top of `xarray` (for data structure) and `pint`/`pint-xarray` (for units).
The aim is to do one simple job well without getting in the way of other packages and workflows.
I developed it to serve a specific purpose in my own work, and don't plan to make it significantly more complex or feature-rich.

With that out of the way, please feel free to raise an [issue](https://github.com/jmarshrossney/xarray-annotated/issues) or open a [pull request](https://github.com/jmarshrossney/xarray-annotated/pulls) to suggest a change or feature.

## See also

- [nerc-ceh/conduit](https://github.com/nerc-ceh/conduit) --- DAGs
- nerc-ceh/breadboard --- WIP

I originally wrote `xarray-annotated` for [nerc-ceh/conduit](https://github.com/nerc-ceh/conduit), which uses [Hamilton](https://hamilton.dagworks.io/) to build Directed Acyclic Graphs where every node (python function) consumes and returns `DataArray`s.
The most useful thing about signature annotations is that it allows us to extend Hamilton's node compatibility checks to include schema, units, frequencies etc, all by static analysis, before any data is loaded or function called.
A DAG of a hundred nodes can be checked end to end in the time it takes to import it, which is the difference between finding an error in CI and finding it three hours into a run.
