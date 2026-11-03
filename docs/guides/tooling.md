# Reading and writing declarations (advanced)

!!! note "You probably don't need this page"

    `@xra.enforce` is the recommended entry point, and most users need nothing else. Everything below is for *building tooling* on
    top of the declarations — skip it unless you are.

The primitives the decorator is built from are public too, for code that needs to
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
from xarray_annotated import Dims
from xarray_annotated.schema import check_schema

check_schema(da, Dims("time", "x"), name="da", on_mismatch=None, qualname=None)
```

`on_mismatch` takes the [default](policy.md#schema-on_mismatch) when `None`;
`name` labels the array in messages; it is a total no-op when the package is
[switched off](policy.md#enabled).

### `check_units`

Validates, and where allowed converts, a single array:

```python
from xarray_annotated.units import check_units

check_units(da, declared, name, qualname=None)
```

Given an input `da`, `check_units`:

1. reads the unit from `da.pint.units` if `da` is quantified, else `da.attrs["units"]`;
2. if present and parseable, re-stamps `attrs["units"] = declared` on the result when it
   names the same unit, converts `da` to `declared` when it is compatible and conversion
   is opted into (a `Unit(..., convert=True)` marker passed as `declared`), and
   raises `ValueError` otherwise;
3. if missing, unparseable or over-long, raises `ValueError`;
4. if present but **dimensionally incompatible** with `declared` (e.g. `"kg"` where
   `"Pa"` is declared), raises `pint.DimensionalityError` naming the offending
   variable — always, regardless of policy.

### `apply_output_units`

The output counterpart, and deliberately not symmetric — it *stamps* rather than converts:

```python
from xarray_annotated.units import apply_output_units

apply_output_units(da, declared, name, on_output=None, qualname=None)
```

Returns a shallow copy of `da` with `attrs["units"] = declared`, leaving `da` as it was.
Under [`on_output="strict"`](policy.md#units-on_output) a label that is present, parseable, and
different from `declared` raises instead. An absent, over-long, or unparseable label is
always stamped — absence is not evidence of a mismatch.

Reach for this when you are applying a declaration you read off a signature yourself.
**Always use the return value.** The label is only on what comes back, and a
[pint-quantified](troubleshooting.md#quantified-arrays) array is converted rather than
stamped.

### `check_freq`

Validates a single array's time axis and returns it **unchanged** (or raises
`FreqError`), taking the same shape of arguments:

```python
from xarray_annotated import Freq
from xarray_annotated.temporal import check_freq

check_freq(
    da, Freq("7D"), name="da", on_mismatch=None, on_uninferable=None, qualname=None
)
```

### Checking a declaration itself

`assert_valid_unit(unit, context)` / `assert_valid_schema(marker, context)` /
`assert_valid_freq(marker, context)` provide the same fail-fast declaration checks the
decorators run at import — useful if you are building markers dynamically and want the
typo caught where it was made.


## Reading declarations

These extract a function's declared properties without calling it — the single source
that both the decorators and any static checker consume, so a declaration is never
written twice.

### `declarations_from_signature`

Reads *all* declared facets — unit, dims, dtype, coords, and freq — into a single uniform
`Declared` value per parameter. This is the read-side counterpart to `annotate` (below),
and their round-trip is exact:

```python
from typing import Annotated, TypedDict
import xarray as xr
from xarray_annotated import Dims, Dtype, Unit, declarations_from_signature


class Output(TypedDict):
    gpp: Annotated[xr.DataArray, Unit("g m-2 d-1")]
    diagnostic: xr.DataArray


def node(
    x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("degC")],
    plain: xr.DataArray,
) -> Output: ...


inputs, output = declarations_from_signature(node)
# inputs == {"x": Declared(unit=Unit("degC"), dims=Dims("time", "x"), dtype=Dtype("float64"))}
# output == {"gpp": Declared(unit=Unit("g m-2 d-1"))}
```

Parameters and fields with no declared facet are omitted. A `TypedDict`/`dataclass` return
is read per-field; a single `Annotated[DataArray, ...]` return yields one `Declared` rather
than a dict; any other return yields `None`.

Each slot of a `Declared` holds that facet's marker or `None`, so `.unit.unit` recovers the
unit string and `.unit.convert` what the declaration says about conversion. `.schema` lists
the `dims`, `dtype` and `coords` markers that are set, which is what `check_schema` takes.

### `declaration_from_hint`

The same read for when you already hold one hint. It returns `None` when the hint declares
nothing, isn't `Annotated`, or annotates a non-`DataArray` base type:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated import Dims, Freq, Unit, declaration_from_hint

declaration_from_hint(Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")])
# Declared(unit=Unit("Pa"), dims=Dims("time"), freq=Freq("D"))
```

A wrapper exposing the wrapped class as `array_type` — such as a jaxtyping array type —
counts as a `DataArray` base type; see [Using with jaxtyping](jaxtyping.md).

To read a facet of your own, pass your own single-hint extractor to `walk_signature`, using
`unwrap_annotated` to see through the metadata. Both are at the package root.

### Checking a graph

To check a *graph* rather than a call, read each node's declared inputs and outputs with
`declarations_from_signature`, compare them pairwise, and fail at assembly time rather
than at run time. `freq_compatible` (see
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
from xarray_annotated import Unit

annotate(unit=Unit("degC"), dims=("time", "x"))
```

Assign the result to a function's `__annotations__` and `@xra.enforce` reads it back
exactly as if it were hand-written.

The first positional argument is the base type, which defaults to `xr.DataArray`. Pass a
jaxtyping array type there to keep dtype/shape checking on the generated hint —
`annotate(Float[xr.DataArray, "time x"], unit="Pa")` round-trips unchanged.
