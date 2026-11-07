# Pre-flight checks

A check reads only metadata and indexes, never an array's values.
That makes it cheap enough to run before an expensive workflow starts, so a transposed axis or a `hPa`-for-`Pa` slip fails in a second rather than an hour into a run.

A pre-flight applies two kinds of check across a pipeline's whole graph, which works for any pipeline, and one that can be evaluated lazily can go a step further:

| Check | Compares | Reads | Catches |
|---|---|---|---|
| [Static](#static) | Declarations with each other | No arrays | Stages that disagree about what passes between them |
| [Metadata](#metadata) | Declarations with opened inputs | Metadata and indexes | A file with a wrong unit, a renamed dimension or the wrong frequency |
| [Lazy run](#lazy-run) | Declarations with the arrays each body returns | Metadata and indexes, if every body stays lazy | A body whose output has the wrong label or structure |

This guide uses one small pipeline and one input file, in which `t` is in `degC` where the pipeline needs `K`:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


@xra.enforce
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return xra.units.convert(p, "Pa")


@xra.enforce
def air_density(
    p: Annotated[xr.DataArray, Unit("Pa")],
    t: Annotated[xr.DataArray, Unit("K")],
) -> Annotated[xr.DataArray, Unit("kg/m^3"), Freq("h")]:
    return (p / (287.05 * t)).assign_attrs(units="kg/m^3")


@xra.enforce
def daily_mean(
    x: Annotated[xr.DataArray, Unit("kg/m^3"), Freq("h")],
) -> Annotated[xr.DataArray, Unit("kg/m^3"), Freq("D")]:
    return x.resample(time="D").mean().assign_attrs(x.attrs)


time = pd.date_range("2024-07-01", periods=48, freq="h")
path = Path(tempfile.mkdtemp()) / "in.nc"
xr.Dataset(
    {
        "p": ("time", np.full(48, 1013.25), {"units": "hPa"}),
        "t": ("time", np.full(48, 15.0), {"units": "degC"}),
    },
    coords={"time": time},
).to_netcdf(path)
```

## Static checks { #static }

Each declaration can look fine on its own and still disagree with the next stage.
Comparing the two needs no data at all:

```python
_, produced = xra.declarations_from_signature(to_pascals)
consumed, _ = xra.declarations_from_signature(air_density)

xra.declaration_conflicts(produced, consumed["p"])
# []
```

An empty list means these two stages agree.
See [Comparing declarations](working_with_declarations.md#comparing) for what a conflict looks like and which rules apply.

## Metadata checks { #metadata }

Next, open the inputs without reading them, and check them against what their consumers declare:

```python
ds = xr.open_dataset(path)

xra.declaration_violations(ds["t"], consumed["t"], "t", "air_density")
# ["[air_density] input 't': unit 'degC' differs from declared 'K'. Converting would ..."]
```

[`declaration_violations`][xarray_annotated.declaration_violations] returns one message per failure, which suits a report of everything that is wrong.
[`check_declared`][xarray_annotated.check_declared] applies the same checks, but raises at the first failure, as `enforce` would when calling the function:

<!-- docs-test: raises UnitError -->
```python
xra.check_declared(ds["t"], consumed["t"], "t", "air_density")
```

Neither reads the array's values, with or without dask:

| Marker | Reads |
|---|---|
| `Unit` | `attrs["units"]`, or the unit a quantified array holds |
| `Dims`, `Coords`, `Dtype` | The array's names and dtype |
| `Freq` | The time axis's index, which xarray holds in memory |

A failed check raises a [`ContractError`][xarray_annotated.ContractError]: `UnitError`, `SchemaError` or `FreqError`, or `ContractError` itself for a value that is not a `DataArray`.
A malformed declaration raises `ValueError` instead.

## Checking the whole graph { #whole-graph }

Without a pre-flight, each mismatch appears only when the run reaches it, so a mistake in the last stage costs the whole run, and fixing it reveals the next one.
Applying both kinds of check to every edge and input of the pipeline's graph, before loading anything, reports them all at the start.

This adds coverage rather than a new kind of check, and because no function runs, it works on any pipeline, however its bodies compute.

### Every edge { #every-edge }

Write the graph out with each node naming its function and where each of its parameters comes from, and [`wiring_conflicts`][xarray_annotated.wiring_conflicts] compares the declarations along every edge:

```python
graph = {
    "p_pa": (to_pascals, {"p": "p"}),
    "density": (air_density, {"p": "p_pa", "t": "t"}),
    "daily": (daily_mean, {"x": "density"}),
}

xra.wiring_conflicts(graph, inputs=["p", "t"])
# []
```

Wiring the last stage to the wrong node shows what a conflict looks like:

```python
miswired = {**graph, "daily": (daily_mean, {"x": "p_pa"})}

xra.wiring_conflicts(miswired, inputs=["p", "t"])
# ["p_pa -> daily(x): unit: Unit('Pa') and Unit('kg/m^3') are dimensionally incompatible"]
```

It applies the [rules for comparing declarations](working_with_declarations.md#comparing) to the whole graph:

- **Several outputs:** a source `("stats", "mean")` is the `mean` field of the `TypedDict` or dataclass that node `stats` returns, and each field is its own producer.
- **Several consumers:** the declarations for one value are compared in all pairs, so two stages reading one input are compared with each other.
- **Nodes that declare nothing:** they break the chain, as carrying a declaration across one is a decision about what its body does.
- **Values that cannot be arrays:** a node returning `float`, a record or `None`, or a record field hinted `int`, is reported where it feeds a declared parameter that `enforce` would refuse it for. A missing or `Any` hint says nothing, and checking what reaches an undeclared parameter is a type checker's job, so neither is reported.
- **Parameters left out of the wiring:** a declared parameter with no default must be wired, even to a name in `inputs` for a fixed array such as a mask, so no declared edge goes unchecked. Any other parameter may be left out, as for one taking a constant.
- **Names that do not match the functions:** a source that is neither a node nor an input, a parameter its function does not take, or a field its record does not have, raises `ValueError` rather than leaving an edge unchecked, as does an unwired declared parameter.

Like `declaration_conflicts`, it reports a fact about declarations, so it ignores [the switch](#respecting-the-switch).

### Every input { #every-input }

The inputs are checked against each of their consumers with `declaration_violations`, as above:

```python
sources = {"p": ds["p"], "t": ds["t"]}
problems = []
for name, (func, wiring) in graph.items():
    consumed, _ = xra.declarations_from_signature(func)
    for param, upstream in wiring.items():
        if upstream in sources and param in consumed:
            problems += xra.declaration_violations(
                sources[upstream], consumed[param], param, func.__qualname__
            )

problems
# ["[air_density] input 't': unit 'degC' differs from declared 'K'. Converting would ..."]
```

Between them, these report every problem the declarations can show, with no data read and no function run.

### With a graph tool

The dictionary is a second description of the pipeline.
`wiring_conflicts` raises when its names stop matching the functions, but cannot tell when the real calls are wired differently from it.

A tool that owns the graph removes that second description, and is the better way to run a pre-flight on anything sizeable:

- **[Apache Hamilton](https://hamilton.apache.org/concepts/node/):** a function's name is its node and its parameter names are its dependencies, so the graph is read from the signatures that hold the declarations, and cannot drift from them.
- **[catena](https://github.com/NERC-CEH/catena):** builds on Hamilton, and `catena run --dry-run` opens the input headers, builds the graph and checks every edge and input without executing a node, using the functions on this page. It also carries units across the nodes it knows preserve them.

## Going further: calling the bodies { #lazy-run }

Walking the graph compares what functions declare, so it cannot see what a body actually returns.
Only calling the function shows that, and whether you can afford to before the run depends on whether the pipeline can be evaluated lazily:

| Pipeline | Before the run | What bodies return is checked |
|---|---|---|
| [Mostly lazy](#mostly-lazy) | Walk the graph, then call every function | Before loading, up to any body that needs values |
| [Cannot be lazy](#not-lazy) | Walk the graph | During the run |

[Using with dask](../tools/dask.md#which-operations-stay-lazy) lists the operations that cannot be deferred, and shows how to test a function.

### A pipeline that is mostly lazy { #mostly-lazy }

If the inputs are dask-backed, which `chunks={}` asks for when opening a file, each function builds a dask graph rather than computing, and `enforce` checks its inputs and output as it would in a real run:

```python
ds = xr.open_dataset(path, chunks={})
results = {"p": ds["p"], "t": xra.units.convert(ds["t"], "K")}

for name, (func, wiring) in graph.items():
    results[name] = func(**{param: results[up] for param, up in wiring.items()})

results["daily"].sizes
# Frozen({'time': 2})
```

Every function has now been called and checked, no array has been read, and `results["daily"].compute()` would do the work.
No graph tool is needed for this, as calling the pipeline as ordinary code builds [dask](https://www.dask.org/)'s graph, and a Hamilton graph given lazily opened inputs behaves the same way.

This adds two things to walking the graph:

- **Actual outputs:** a body's real label and structure are checked, not just its declaration. Below, the declarations agree with each other and with the file, and only the call fails.
- **Nodes that declare nothing:** the next function checks what such a node actually produced, where the walk stops at it.

<!-- docs-test: raises UnitError -->
```python
@xra.enforce
def stale_label(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p * 100


stale_label(ds["p"])
```

It has limits of its own:

- **Labels, not values:** `(p * 100).assign_attrs(units="Pa")` passes, right or wrong, as the check still reads only the label.
- **A body that needs values:** it reads what it needs when called, which costs that step and everything upstream of it, and functions downstream are checked only afterwards. A small step late in the pipeline costs little, and one early on may cost most of the run.
- **`dask.delayed`:** wrapping a call in it defers the call, and so defers [its checks](../tools/dask.md#when-checks-run) until compute. Pass lazy arrays to functions called directly instead.

### A pipeline that cannot be lazy { #not-lazy }

If bodies need the values early on, such as a solver written in numpy or a loop over a time axis, then calling the pipeline is the run.
Walking the graph is unaffected, since it runs nothing, so it is the pre-flight for these pipelines, and what each body returns is left to `enforce` during the run.

Two things make that go further:

- **Declare both sides of every edge:** the walk only sees what is declared, so a node with no declaration leaves its edges unchecked until the data arrives.
- **Run a short slice first:** calling the pipeline on `ds.isel(time=slice(0, 48))` checks every body for the cost of a little data, if the slice is long enough for the bodies and any declared `Freq`.

## What looking cannot tell you

Reading metadata is what makes these checks cheap, and it is also their limit:

- **Values go unread:** a check cannot tell that a label is false, so 1000 labelled `Pa` that is really `hPa` passes. See [What a label cannot catch](../guides/units.md#what-a-label-cannot-catch).
- **Meaning goes unchecked:** two quantities in one unit look the same, such as specific humidity and mixing ratio in `kg/kg`.
- **Only a real run is final:** keep `enforce` on for it, as a pre-flight describes only the files and code it saw.

## Respecting the switch

`check_declared` and `declaration_violations` do nothing when [the package is switched off](../guides/first_principles.md#enabled).
`declaration_conflicts` and `wiring_conflicts` report a fact about declarations, so they ignore the switch, and a tool that should stay quiet can ask [`resolve_enabled`][xarray_annotated.resolve_enabled] first.
