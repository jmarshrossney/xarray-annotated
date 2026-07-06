# Usage

`xarray-annotated` lets you declare a property of a `DataArray` in a function
signature with `typing.Annotated`, then validate it automatically with a decorator.
There are four kinds of property, in two groups:

- **structural** — **dims**, **coords**, and **dtype** — checked (never mutated) by
  `@declare_schema`;
- **physical units** — checked *and converted* (via pint/CF) by `@declare_units`.

!!! warning

    Do **not** use `from __future__ import annotations` in modules that declare units
    or schema markers. Declarations are read as *runtime objects* out of the
    `Annotated` metadata; that import stringizes annotations, forcing a re-`eval` that
    fails (a `NameError` at decoration time) whenever a needed name — e.g. a
    `TYPE_CHECKING`-only `xarray` import — isn't resolvable at runtime. Python 3.14's
    deferred-annotation model removes this constraint.

## Concepts

All four properties work the same way, so what you learn for one transfers to the
others.

**Declare once, in the signature.** A property is declared as `Annotated` metadata on a
`DataArray` parameter or return — `Annotated[xr.DataArray, Dims("time", "x")]` (a
structural marker) or `Annotated[xr.DataArray, Unit("Pa")]` (a unit). The annotation is
the single source of truth, read once and never written twice.

**Two decorators.** The structural properties — **dims**, **coords**, and **dtype** —
are validated by `@declare_schema`; physical **units** by `@declare_units`. `schema`
only ever *checks* (arrays pass through unchanged); `units` also *converts* (e.g.
`"hPa"` → `"Pa"`). Stack both decorators to check structure and units at once — see
[Combining multiple checks](#combining-multiple-checks). Each decorator is a thin layer
over a public primitive (`check_schema` / `check_units`) that you can call by hand; see
[Advanced usage](#advanced-usage).

**Fail fast at decoration.** Each decorator validates its *declarations* when it is
applied (at import) — a typo'd unit or an unparseable dtype raises immediately, rather
than only when the function is first called, and regardless of policy.

**Policy.** Each decorator follows a small **policy** governing what happens on a
validation event. Every policy shares the package-wide **`enabled`** master switch:
`enabled=False` makes *both* decorators a total no-op (no validation, conversion, or
stamping). Each axis resolves once per call, in order:

1. its environment variable,
2. a process-wide override set with `set_policy(...)`,
3. the built-in default.

Use a domain's `policy(...)` as a context manager to scope overrides across a block and
restore them on exit:

```python
from xarray_annotated.units import policy as units_policy

with units_policy(enabled=False):   # disables *both* decorators for the block
    ...
```

The `enabled` switch (env `XARRAY_ANNOTATED_ENABLED`) is shared; the behavioural axes
described under each property below are domain-specific — `on_mismatch` for schema,
`on_missing`/`on_inexact` for units.

## Dims

### Declaring dims

Declare a DataArray's dimensions with the `Dims` marker in its `Annotated` metadata, and
apply `@declare_schema` to check every declared input and output on each call:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dims

@declare_schema
def standardise(
    x: Annotated[xr.DataArray, Dims("time", "x")],
) -> Annotated[xr.DataArray, Dims("time", "x")]:
    return x
```

`@declare_schema` reads the markers off the signature and, on each call, validates every
declared `DataArray` input and output. **It never mutates** — arrays pass through
unchanged; a mismatch raises, warns, or is ignored per policy. A `TypedDict` or
`dataclass` return is validated per-field; a bare `Annotated[DataArray, ...]` return is
validated directly. Non-`DataArray` arguments and returns pass through untouched.

Unlike units there is **no bare-string shorthand** — a plain string in the metadata is
treated as a description and ignored; only the typed markers are read.

### Strictness and the schema policy

**`Dims(*names, ordered=False)`** — by default the *set* of dims must match (extra or
missing dims fail); order is free, because xarray operations are order-independent until
you drop to numpy. Pass `ordered=True` to also pin the order (e.g. before `.values`,
`.stack`, or `apply_ufunc`):

```python
Annotated[xr.DataArray, Dims("time", "x", ordered=True)]
```

Because structural validation never converts, the schema policy has a single behavioural
axis on top of the shared [`enabled`](#concepts) switch: **`on_mismatch`**, governing
what happens when an array doesn't match a declaration. It is shared by all three
structural markers (`Dims`, `Coords`, `Dtype`).

| `on_mismatch`     | on a structural mismatch                           |
|-------------------|----------------------------------------------------|
| `error` (default) | raises `SchemaError`                               |
| `warn`            | emits `SchemaWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged               |

The axis resolves via `XARRAY_ANNOTATED_SCHEMA_ON_MISMATCH` (default `error` — a
structural mismatch usually signals a genuine wiring bug) as described under
[Concepts](#concepts). `SchemaError` is deliberately **not** a `ValueError`, so catching
a mismatch never accidentally swallows a malformed-declaration `ValueError`.

Override the policy per function:

```python
@declare_schema(on_mismatch="warn")
def lenient(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray: ...
```

**Per-marker override.** Any marker may carry its own `on_mismatch`, which wins over the
decorator/call argument and the policy default — so a wrong dtype can be a warning while
a wrong set of dims stays an error:

```python
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64", on_mismatch="warn")]
```

Effective severity resolves: **marker override → decorator/call argument → policy
default**.

## Coords

### Declaring coords

Declare required coordinates with the `Coords` marker, applied the same way with
`@declare_schema`:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Coords

@declare_schema
def anomalies(
    x: Annotated[xr.DataArray, Coords("time")],
) -> Annotated[xr.DataArray, Coords("time")]:
    return x - x.mean("time")
```

### Strictness

**`Coords(*names)`** — the named coordinates must be *present* (as labels, not merely
dims — a dim can exist without coordinate values). Extra coordinates are allowed.

Severity follows the shared schema policy `on_mismatch`
([above](#strictness-and-the-schema-policy)): a missing coordinate raises `SchemaError`
by default, or warns / is ignored per policy. A `Coords` marker may also carry its own
`on_mismatch` override.

## Dtype

### Declaring dtype

Declare an expected dtype with the `Dtype` marker, again applied with `@declare_schema`:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dtype

@declare_schema
def to_float(
    x: Annotated[xr.DataArray, Dtype("float64")],
) -> Annotated[xr.DataArray, Dtype("float64")]:
    return x
```

### Strictness

**`Dtype(dtype, exact=False)`** — by default matches by numpy *kind*: any float satisfies
`Dtype("float64")`, any integer `Dtype("int32")` — enough to catch an int/float or
bool/float mix-up without firing on `float64` vs `float32`. Pass `exact=True` to require
the precise dtype (e.g. to pin memory footprint or a typed sink):

```python
Annotated[xr.DataArray, Dtype("float32", exact=True)]
```

Severity follows the shared schema policy `on_mismatch`
([above](#strictness-and-the-schema-policy)), and a `Dtype` marker may carry its own
`on_mismatch` override.

## Units

### Declaring units

Declare a unit with the self-identifying `Unit` marker, and apply `@declare_units` to
validate, convert, and stamp declared inputs and outputs on each call:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import declare_units, Unit

@declare_units
def normalise_pressure(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p
```

On each call, under the active [policy](#the-units-policy), `@declare_units` validates
and converts every declared `DataArray` **input**, runs the function, then stamps each
declared `DataArray` **output** with its unit. A `TypedDict` or `dataclass` return is
stamped per-field; a bare `Annotated[DataArray, ...]` return takes that unit.
Non-`DataArray` arguments and returns pass through untouched.

`Unit` is the **recommended** form: it owns its own slot in the metadata, so it stays
unambiguous and order-independent even when other `Annotated`-based tooling — or a schema
marker (see [Combining multiple checks](#combining-multiple-checks)) — shares the
annotation.

#### Bare-string shorthand

When you're checking **only** units — no schema markers, no other `Annotated` metadata to
collide with — a bare string is accepted as a convenient shorthand:

```python
Annotated[xr.DataArray, "Pa"]
Annotated[xr.DataArray, "m s-1", "z component of velocity"]  # unit first; later string ignored
```

The unit must come **first**; any later string is treated as a human description and
ignored. A `Unit` marker always wins over a bare string when both are present:

```python
Annotated[xr.DataArray, "note", Unit("Pa")]  # resolves to "Pa"
```

Reach for `Unit(...)` as soon as the annotation is shared — its self-identifying slot
removes both the order dependence and any clash with a description string. This is also
why the schema markers have **no** bare-string form: a unit has a canonical string
spelling like `"Pa"`, whereas a structural property does not, so there a string can only
ever be prose (see [Dims](#dims)).

### The units policy

`@declare_units` follows the units policy, which has two behavioural axes on top of the
shared [`enabled`](#concepts) switch. Dimensional mismatches (e.g. `"kg"` where `"Pa"`
is declared) are never negotiable — they always raise `pint.DimensionalityError`.

Override the policy per function with keyword arguments (each defaults to the active
policy when omitted):

```python
@declare_units(on_missing="error", on_inexact="error")
def strict_node(x: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray: ...
```

#### `on_missing` — no parseable unit to check against

Governs only the "can't validate" cases: a missing or unparseable `units` attribute.

| `on_missing` | missing/unparseable units                     | dimensional mismatch |
|--------------|-----------------------------------------------|----------------------|
| `error`      | raises `ValueError`                           | always raises        |
| `warn`       | emits `UnitsWarning`, returns input unchanged | always raises        |
| `ignore`     | silently returns input unchanged              | always raises        |

#### `on_inexact` — value-changing conversion

By default (`convert`), a dimensionally compatible unit is silently converted
(`"hPa"` → `"Pa"`). `on_inexact` controls what happens when the actual unit is compatible
with the declared one but *not identical* — any conversion that would change the values,
including affine ones like `"K"` → `"degC"`:

| `on_inexact` | value-changing conversion                 |
|--------------|-------------------------------------------|
| `convert`    | performs the conversion silently          |
| `warn`       | converts, but emits `UnitsWarning`        |
| `error`      | raises `ValueError` instead of converting |

Equivalent spellings of the same unit (`"pascal"` for `"Pa"`) imply no value change and
always convert. `error` is useful when implicit conversion would hide a likely mistake
upstream, and you'd rather the caller fix the unit at the source.

The axes resolve via the environment variables `XARRAY_ANNOTATED_UNITS_ON_MISSING` and
`XARRAY_ANNOTATED_UNITS_ON_INEXACT` (defaults `on_missing="warn"`,
`on_inexact="convert"`) as described under [Concepts](#concepts). Scope overrides with
`policy(...)`:

```python
from xarray_annotated.units import policy

with policy(on_missing="error", on_inexact="warn"):
    ...
```

#### Choosing a registry: pint vs. CF/UDUNITS

Out of the box you get **plain pint** (`pint.get_application_registry()`), so standard
pint unit strings ("Pa", "degC", "m/s") parse with no setup.

CF-convention strings such as `"umol m-2 s-1"` or `"g m-2 d-1"` need cf-xarray's
UDUNITS-aware registry instead. Install the `[cf]` extra and activate it **once, at
startup**:

```python
from xarray_annotated.units import use_cf_units

use_cf_units()   # now "umol m-2 s-1", "g m-2 d-1" parse
```

Or supply any registry yourself:

```python
import pint
from xarray_annotated.units import set_registry

set_registry(pint.UnitRegistry())
```

pint has a single process-global application registry, so this is a one-time, startup
choice — not a per-array setting. Quantities created under two different registries
cannot be mixed (pint raises). Choose pint units *or* CF units for your entire codebase,
not a mixture.

## Combining multiple checks

A single `Annotated` hint can carry several markers, since a DataArray has all of dims,
coords, and dtype at once. `@declare_schema` reads and checks them all:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dims, Coords

@declare_schema
def detrend(
    x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time")],
) -> Annotated[xr.DataArray, Dims("time", "x")]:
    return x
```

To check **structure and units together**, stack the two decorators. The schema markers
and the `Unit` marker coexist in one `Annotated`: `@declare_schema` reads the typed
markers and ignores `Unit`, while `@declare_units` reads `Unit` and ignores the schema
markers. Prefer the `Unit` marker over the bare-string shorthand here, precisely because
the annotation is shared (see [Declaring units](#declaring-units)).

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dims, Coords, Dtype
from xarray_annotated.units import declare_units, Unit

@declare_units
@declare_schema
def process(
    x: Annotated[xr.DataArray, Dims("time", "x"), Coords("time"), Dtype("float64"), Unit("degC")],
) -> Annotated[xr.DataArray, Dims("time", "x"), Unit("degC")]:
    return x
```

The outer decorator's input handling runs first: here `@declare_units` converts the
input to `"degC"`, then `@declare_schema` validates the converted array's dims, coords,
and dtype before the body runs — and on the way out, the schema check runs before
`@declare_units` stamps the output unit. Both orders work; put `@declare_units` outermost
when you want the structural checks to see the array in its declared units.

## Advanced usage

The decorators are the recommended entry point. The primitives they call are public too,
for tools that need to validate an array by hand or inspect declarations statically
(e.g. build-time checks, documentation generation, custom consumers). Most users won't
need these.

### Validating directly: `check_schema` and `check_units`

`check_schema` validates a single array against a marker or list of markers and returns
it **unchanged** (or raises `SchemaError`):

```python
from xarray_annotated.schema import check_schema, Dims

check_schema(da, Dims("time", "x"), name="da", on_mismatch=None, qualname=None)
```

`on_mismatch` defaults to the active policy when `None`; `name` labels the array in
messages; it is a total no-op when the policy is disabled.

`check_units` validates *and converts* a single array:

```python
from xarray_annotated.units import check_units

check_units(da, declared, name, on_missing=None, on_inexact=None, qualname=None)
```

Given an input `da`, `check_units`:

1. reads `da.attrs["units"]`;
2. if present and parseable, converts `da` to `declared` and re-stamps
   `attrs["units"] = declared` on the result;
3. if missing or unparseable, follows the [`on_missing`](#on_missing-no-parseable-unit-to-check-against)
   axis;
4. if present but **dimensionally incompatible** with `declared` (e.g. `"kg"` where
   `"Pa"` is declared), raises `pint.DimensionalityError` naming the offending
   variable — always, regardless of policy.

`on_missing` and `on_inexact` may be passed per call; each defaults to the active policy
when `None`. `assert_valid_unit(unit, context)` / `assert_valid_schema(marker, context)`
provide the same fail-fast declaration checks the decorators run at import.

### Reading declarations: `schema_from_signature` and `units_from_signature`

These extract a function's declared properties without calling it — the single source
that both the decorators and any static checker consume, so a declaration is never
written twice.

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

`schema_from_signature` mirrors it, returning the *list* of markers on each
parameter/field (since a hint may declare several):

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

### Cross-domain reader: `declarations_from_signature`

`declarations_from_signature` (from the package root) reads *all* declared facets — unit, dims,
dtype, and coords — into a single uniform `Declared` value per parameter. This is the read-side
counterpart to `annotate` (below), and their round-trip is exact:

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

### Writing annotations programmatically: `annotate`

`annotate` (from the package root) is the inverse of the readers: given facet values it returns a
real `Annotated` hint — useful for code generation or tools that build function signatures
dynamically:

```python
from typing import Annotated, get_args, get_origin
import xarray as xr
from xarray_annotated import annotate

hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64")
# Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x"), Dtype("float64")]

annotate() is xr.DataArray  # no-op when no facets given
```

Each facet accepts either a raw value or an already-built marker, so a caller holding a mix can
pass both without unwrapping:

```python
from xarray_annotated.units import Unit

annotate(unit=Unit("degC"), dims=("time", "x"))
```

Assign the result to a function's `__annotations__` and the `@declare_units` / `@declare_schema`
decorators read it back exactly as if it were hand-written.
