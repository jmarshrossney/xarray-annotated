# Reading and writing declarations (advanced)

!!! note "You probably don't need this page"

    `@declare_schema`, `@declare_units` and `@declare_freq` are the recommended entry
    point, and most users need nothing else. Everything below is for *building tooling* on
    top of the declarations — skip it unless you are.

The primitives the decorators are built from are public too, for code that needs to
validate an array by hand, or to *inspect* declarations without calling anything —
build-time graph checks, documentation generation, code generation, custom consumers.

This is the API that makes a declaration useful to something other than the function it
sits on.


## Validating by hand

Use these where a decorator doesn't fit: inside a loop, at a boundary you don't own, or
on an array that arrived from somewhere other than a parameter.

### `check_schema`

Validates a single array against a marker or list of markers and returns it
**unchanged** (or raises `SchemaError`):

```python
from xarray_annotated.schema import check_schema, Dims

check_schema(da, Dims("time", "x"), name="da", on_mismatch=None, qualname=None)
```

`on_mismatch` defaults to the [active policy](policy.md#schema-on_mismatch) when `None`;
`name` labels the array in messages; it is a total no-op when the policy is disabled.

### `check_units`

Validates *and converts* a single array:

```python
from xarray_annotated.units import check_units

check_units(da, declared, name, on_missing=None, on_inexact=None, qualname=None)
```

Given an input `da`, `check_units`:

1. reads `da.attrs["units"]`;
2. if present and parseable, converts `da` to `declared` and re-stamps
   `attrs["units"] = declared` on the result;
3. if missing or unparseable, follows the [`on_missing`](policy.md#units-on_missing) axis;
4. if present but **dimensionally incompatible** with `declared` (e.g. `"kg"` where
   `"Pa"` is declared), raises `pint.DimensionalityError` naming the offending
   variable — always, regardless of policy.

`on_missing` and `on_inexact` may be passed per call; each defaults to the active policy
when `None`.

### `apply_output_units`

The output counterpart, and deliberately not symmetric — it *stamps* rather than converts:

```python
from xarray_annotated.units import apply_output_units

apply_output_units(da, declared, name, on_output=None, qualname=None)
```

Sets `attrs["units"] = declared` on `da` and returns it. Under
[`on_output="strict"`](policy.md#units-on_output) a label that is present, parseable, and
different from `declared` raises instead. An absent, over-long, or unparseable label is
always stamped — absence is not evidence of a mismatch.

Reach for this when you are applying a declaration you read off a signature yourself.
**Always use the return value.** For an `attrs`-labelled array it stamps in place and hands
back the same object, but a [pint-quantified](troubleshooting.md#quantified-arrays) one is
converted rather than stamped, so what comes back may be a new array.

### `check_freq`

Validates a single array's time axis and returns it **unchanged** (or raises
`FreqError`), taking the same shape of arguments:

```python
from xarray_annotated.temporal import check_freq, Freq

check_freq(da, Freq("7D"), name="da", on_mismatch=None, on_uninferable=None, qualname=None)
```

### Checking a declaration itself

`assert_valid_unit(unit, context)` / `assert_valid_schema(marker, context)` /
`assert_valid_freq(marker, context)` provide the same fail-fast declaration checks the
decorators run at import — useful if you are building markers dynamically and want the
typo caught where it was made.


## Reading declarations off a signature

These extract a function's declared properties without calling it — the single source
that both the decorators and any static checker consume, so a declaration is never
written twice.

### `units_from_signature`

```python
from typing import Annotated, TypedDict
import xarray as xr
from xarray_annotated.units import units_from_signature

class Output(TypedDict):
    gpp: Annotated[xr.DataArray, "g m-2 d-1"]
    lue: Annotated[xr.DataArray, "g MJ-1"]

def node(
    temp: Annotated[xr.DataArray, "degC"],
    plain: xr.DataArray,
) -> Output: ...

inputs, outputs = units_from_signature(node)
# inputs  == {"temp": "degC"}
# outputs == {"gpp": "g m-2 d-1", "lue": "g MJ-1"}
```

Only parameters — or fields of a `TypedDict`/`dataclass` return type — with a
unit-annotated `DataArray` contribute; a plain `xr.DataArray` hint with no unit is
ignored. A bare `Annotated[DataArray, unit]` return annotation yields a single unit
string rather than a dict.

### `schema_from_signature`

Mirrors it, returning the *list* of markers on each parameter/field (since a hint may
declare several):

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import schema_from_signature, Dims, Dtype

def node(
    x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")],
    plain: xr.DataArray,
) -> Annotated[xr.DataArray, Dims("time", "x")]: ...

inputs, output = schema_from_signature(node)
# inputs == {"x": [Dims("time", "x"), Dtype("float64")]}
# output == [Dims("time", "x")]
```

`TypedDict`/`dataclass` returns are read per-field, exactly as for units.
`freq_from_signature` does the same for the `Freq` marker (one marker, or `None`, per
parameter).

### Reading a single hint

Each domain also exposes the per-annotation reader underneath its signature reader, for
when you already hold one hint and don't want to reach into a private module:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import Unit, annotated_unit
from xarray_annotated.schema import Dims, annotated_schema
from xarray_annotated.temporal import Freq, annotated_freq

hint = Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]

annotated_unit(hint)    # "Pa"        (normalises the bare-string shorthand too)
annotated_schema(hint)  # [Dims("time")]
annotated_freq(hint)    # Freq("D")
```

Each returns `None` when the hint declares nothing for that domain, isn't `Annotated`, or
annotates a non-`DataArray` base type. These are the building blocks the `*_from_signature`
readers are driven from, alongside `walk_signature` and `unwrap_annotated` at the package
root — the combination a third-party facet author needs to write their own reader.

### Cross-domain reader: `declarations_from_signature` { #cross-domain-reader-declarations_from_signature }

`declarations_from_signature` (from the package root) reads *all* declared facets — unit, dims,
dtype, coords, and freq — into a single uniform `Declared` value per parameter. This is the
read-side counterpart to `annotate` (below), and their round-trip is exact:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated import declarations_from_signature
from xarray_annotated.schema import Dims, Dtype
from xarray_annotated.units import Unit

def node(
    x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("degC")],
) -> Annotated[xr.DataArray, Dims("time", "x")]: ...

inputs, output = declarations_from_signature(node)
# inputs == {"x": Declared(dims=Dims("time", "x"), dtype=Dtype("float64"), unit=Unit("degC"))}
# output == Declared(dims=Dims("time", "x"))
```

A bare-string unit shorthand is normalised to a `Unit` marker on read, so `.unit.unit` always
recovers the string. Parameters with no declared facet are omitted entirely.

This is the reader to use when checking a *graph* rather than a call: read each node's
declared inputs and outputs, compare them pairwise, and fail at assembly time rather than
at run time. `freq_compatible` (see
[What "the same frequency" means](declaring.md#what-the-same-frequency-means)) performs
the frequency half of that comparison with no array in hand.


## Writing annotations programmatically: `annotate`

`annotate` (from the package root) is the inverse of the readers: given facet values it returns a
real `Annotated` hint — useful for code generation or tools that build function signatures
dynamically:

```python
from typing import Annotated, get_args, get_origin
import xarray as xr
from xarray_annotated import annotate

hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64", freq="7D")
# Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x"), Dtype("float64"), Freq("7D")]

annotate() is xr.DataArray  # no-op when no facets given
```

Each facet accepts either a raw value or an already-built marker, so a caller holding a mix can
pass both without unwrapping:

```python
from xarray_annotated.units import Unit

annotate(unit=Unit("degC"), dims=("time", "x"))
```

Assign the result to a function's `__annotations__` and the `@declare_units` /
`@declare_schema` / `@declare_freq` decorators read it back exactly as if it were
hand-written.
