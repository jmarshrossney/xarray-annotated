# Look before you leap

A check reads only metadata and indexes, never an array's values.
That makes it cheap enough to run before an expensive workflow starts, so a transposed axis or a `hPa`-for-`Pa` slip fails in a second rather than an hour into a run.

Contracts can be evaluated at three points:

| Stage | Compares | Reads | Catches |
|---|---|---|---|
| [1. Assembling a pipeline](#declarations) | Declarations with each other | No arrays | Stages that disagree about what passes between them |
| [2. Opening the inputs](#inputs) | Declarations with opened files | Metadata and indexes | A file with a wrong unit, a renamed dimension or the wrong frequency |
| [3. Running](#running) | Declarations with every array | Whatever the body computes | Whatever a body actually produced |

[catena](https://github.com/NERC-CEH/catena) does the first two for [Apache Hamilton](https://hamilton.apache.org/) pipelines: `catena run --dry-run` checks every edge of the graph and every input's metadata without executing a node.
The functions below are what a tool like that is built on.

## Comparing declarations { #declarations }

Each declaration can look fine on its own and still disagree with the next stage.
[`declaration_conflicts`][xarray_annotated.declaration_conflicts] compares two of them with no array in hand:

```python
@xra.enforce
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return xra.units.convert(p, "Pa")


@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 1013.25) ** 0.1903)
    return alt.assign_attrs(units="m")


_, produced = xra.declarations_from_signature(to_pascals)
consumed, _ = xra.declarations_from_signature(altitude)

xra.declaration_conflicts(produced, consumed["p"])
# ["unit: Unit('Pa') and Unit('hPa') are different units"]
```

- **Partly annotated pipelines:** a facet declared on only one side cannot conflict, so you can adopt this one function at a time.
- **Units:** they conflict unless equal, matching `enforce`. `units_compatible` alone would pass this example, which `enforce` would then refuse.
- **Several consumers:** compare all pairs of the declarations for one array, not each against the first. A loose `Dims` or `Dtype` is compatible with stricter ones that conflict with each other.

Walking the graph is left to the tool, including how a declaration carries across a node that declares nothing.

## Checking opened inputs { #inputs }

[`check_declared`][xarray_annotated.check_declared] makes the check `enforce` makes on an input, without running the function:

<!-- docs-test: skip -->
```python
inputs, _ = xra.declarations_from_signature(to_pascals)

p = xr.open_dataset("in.nc")["p"]
xra.check_declared(p, inputs["p"], "p")
```

It leaves a lazily opened array unread, with or without dask:

| Marker | Reads |
|---|---|
| `Unit` | `attrs["units"]`, or the unit a quantified array holds |
| `Dims`, `Coords`, `Dtype` | The array's names and dtype |
| `Freq` | The time axis's index, which xarray holds in memory |

It stops at the first failure for an array, and raises `TypeError`, or a [`ContractError`][xarray_annotated.ContractError]: `UnitError`, `SchemaError` or `FreqError`.
A malformed declaration raises `ValueError` instead, and [`assert_valid_declaration`][xarray_annotated.assert_valid_declaration] asks only that, with no array in hand.
To report every failing input at once, catch those for each input in turn.

## Checking during the run { #running }

`enforce` remains necessary, because only running a body shows what it returns.
The earlier stages establish that the declarations agree with each other and with the inputs, not that a body honours its own.

Without a graph tool, opening files with `chunks={}` gets you much of the same benefit.
Each stage then builds a dask graph rather than computing, so calling the whole pipeline runs every check, outputs included, and nothing is computed until you ask.
This stops working at the first body that calls `.compute()` or reads `.values`.

## What looking cannot tell you

Reading metadata is what makes these checks cheap, and it is also their limit:

- **Values go unread:** a check cannot tell that a label is false, so 1000 labelled `Pa` that is really `hPa` passes. [Computing with pint](pint.md) gives you a unit worth trusting on outputs.
- **Meaning goes unchecked:** two quantities in one unit look the same, such as specific humidity and mixing ratio in `kg/kg`.

## Respecting the switch

`check_declared` does nothing when [the package is switched off](troubleshooting.md#enabled).
`declaration_conflicts` reports a fact about declarations, so it ignores the switch, and a tool that should stay quiet can ask [`resolve_enabled`][xarray_annotated.resolve_enabled] first.
