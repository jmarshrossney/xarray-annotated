# Usage

`xarray-annotated` has two domains, imported and used independently: **`units`**
(physical units, checked and converted via pint/CF) and **`schema`** (structural
properties — dims, coords, dtype — checked, never mutated). Both work the same way:
declare a property on a `DataArray` in a function signature with `typing.Annotated`,
then apply it — with a decorator or a `check_*` primitive.

!!! warning

    Do **not** use `from __future__ import annotations` in modules that declare units
    or schema markers. Declarations are read as *runtime objects* out of the
    `Annotated` metadata; that import stringizes annotations, forcing a re-`eval` that
    fails (a `NameError` at decoration time) whenever a needed name — e.g. a
    `TYPE_CHECKING`-only `xarray` import — isn't resolvable at runtime. Python 3.14's
    deferred-annotation model removes this constraint.

## Concepts

The two domains share a common shape, so what you learn for one transfers to the other.

**Declare once, in the signature.** A property is declared as `Annotated` metadata on a
`DataArray` parameter or return — `Annotated[xr.DataArray, "Pa"]` (units) or
`Annotated[xr.DataArray, Dims("time", "x")]` (schema). The annotation is the single
source of truth, read once and never written twice.

**Apply it two ways.** Decorate the function (`@declare_units` / `@declare_schema`) to
validate the declared inputs and outputs automatically on every call, or call the
primitive (`check_units` / `check_schema`) to validate a single array by hand.

**Fail fast at decoration.** Each decorator validates its *declarations* when it is
applied (at import) — a typo'd unit or an unparseable dtype raises immediately, rather
than only when the function is first called, and regardless of policy.

**Policy.** Each domain has a small **policy** governing what happens on a validation
event. Every policy shares the package-wide **`enabled`** master switch: `enabled=False`
makes *both* domains a total no-op (no validation, conversion, or stamping). Each axis
resolves once per call, in order:

1. its environment variable,
2. a process-wide override set with `set_policy(...)`,
3. the built-in default.

Use a domain's `policy(...)` as a context manager to scope overrides across a block and
restore them on exit:

```python
from xarray_annotated.units import policy as units_policy

with units_policy(enabled=False):   # disables *both* domains for the block
    ...
```

The `enabled` switch (env `XARRAY_ANNOTATED_ENABLED`) is shared across domains; the
behavioural axes described under each domain below are domain-specific.

## Units

### Declaring units

A unit is the first `str` found in a `DataArray`'s `Annotated` metadata:

```python
Annotated[xr.DataArray, "degC"]
Annotated[xr.DataArray, "m s-1", "z component of velocity"]  # description after the unit is ignored
```

Equivalently, wrap it in the self-identifying `Unit` marker. This is the
order-independent form: the unit owns its own slot, so it stays unambiguous even when
other typed metadata shares the annotation — useful when composing with other
`Annotated`-based tooling.

```python
from xarray_annotated.units import Unit

Annotated[xr.DataArray, Unit("degC")]
Annotated[xr.DataArray, "note", Unit("Pa")]  # marker wins regardless of order → "Pa"
```

Both forms resolve to the same unit string; use whichever reads better. A `Unit` marker
takes priority over a bare string when both are present, and a description string still
comes *after* the unit in the bare-string form.

The declared units of a whole function are read back with
[`units_from_signature`](#reading-declarations-units_from_signature) — the single source
that both `@declare_units` and any static checker consume, so a unit is never written
twice.

`assert_valid_unit(unit, context)` fails fast at declaration time — a typo like
`"degrees_C"` raises `ValueError` immediately, rather than only surfacing when the
annotation is later used to validate data.

### Applying units: `declare_units`

`@declare_units` reads the declared unit off the signature, so it lives in the annotation
and nowhere else:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import declare_units

@declare_units
def normalise_pressure(
    p: Annotated[xr.DataArray, "Pa"],
) -> Annotated[xr.DataArray, "Pa"]:
    return p
```

On each call, under the active [policy](#the-units-policy), the wrapper validates and
converts every declared `DataArray` **input** via `check_units`, runs the function, then
stamps each declared `DataArray` **output** with its unit. A `TypedDict` or `dataclass`
return is stamped per-field; a bare `Annotated[DataArray, unit]` return takes that unit.
Non-`DataArray` arguments and returns pass through untouched.

Override the policy per function with keyword arguments (each defaults to the active
policy when omitted):

```python
@declare_units(on_missing="error", on_inexact="error")
def strict_node(x: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray: ...
```

`declare_units` is intentionally a thin convenience built from the public primitives. If
you need behaviour it doesn't cover — a build-time/static check, a custom value type —
assemble your own consumer from
[`units_from_signature`](#reading-declarations-units_from_signature),
[`check_units`](#validating-directly-check_units), and `assert_valid_unit` rather than
subclassing anything.

### The units policy

Both `@declare_units` and `check_units` follow the units policy, which has two
behavioural axes on top of the shared [`enabled`](#concepts) switch. Dimensional
mismatches are never negotiable — they always raise.

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
from xarray_annotated.units import policy, check_units

with policy(on_missing="error", on_inexact="warn"):
    check_units(da, "Pa", "vpd")
```

### Validating directly: `check_units`

`@declare_units` is the recommended entry point, but the `check_units` primitive it calls
is public too — reach for it when you want to validate an array by hand rather than
decorate a function:

```python
check_units(da, declared, name, on_missing=None, on_inexact=None, qualname=None)
```

Given an input `da`, `check_units`:

1. reads `da.attrs["units"]`;
2. if present and parseable, converts `da` to `declared` and re-stamps
   `attrs["units"] = declared` on the result;
3. if missing or unparseable, follows the
   [`on_missing`](#on_missing-no-parseable-unit-to-check-against) axis;
4. if present but **dimensionally incompatible** with `declared` (e.g. `"kg"` where
   `"Pa"` is declared), raises `pint.DimensionalityError` naming the offending variable —
   always, regardless of policy.

`on_missing` and `on_inexact` may be passed per call; each defaults to the active policy
when `None`. `name` names the array for error/warning messages.

### Choosing a registry: pint vs. CF/UDUNITS

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

### Reading declarations: `units_from_signature`

For tools that want to inspect a function's declared units statically (e.g. to generate
documentation, or to wire up validation automatically), `units_from_signature` extracts
them without needing to call the function:

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

Dataclass return types work identically. Only parameters — or fields of a
`TypedDict`/`dataclass` return type — with a unit-annotated `DataArray` contribute; a
plain `xr.DataArray` hint with no unit is ignored. A bare `Annotated[DataArray, unit]`
return annotation yields a single unit string rather than a dict.

## Schema

### Declaring structure: `Dims`, `Coords`, `Dtype`

Declare a DataArray's structure with typed markers in its `Annotated` metadata — several
at once, since a DataArray has all of dims, coords, and dtype:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import Dims, Coords, Dtype

Annotated[xr.DataArray, Dims("time", "x")]
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")]   # several markers at once
Annotated[xr.DataArray, Coords("time"), Dtype("float32")]
```

Unlike units there is **no bare-string shorthand** — a string in the metadata is treated
as a description and ignored; only the typed markers are read.

Each marker carries the strictness of its own check:

- **`Dims(*names, ordered=False)`** — by default the *set* of dims must match (extra or
  missing dims fail); order is free, because xarray operations are order-independent
  until you drop to numpy. Pass `ordered=True` to also pin the order (e.g. before
  `.values`, `.stack`, or `apply_ufunc`).
- **`Coords(*names)`** — the named coordinates must be *present* (as labels, not merely
  dims — a dim can exist without coordinate values). Extra coordinates are allowed.
- **`Dtype(dtype, exact=False)`** — by default matches by numpy *kind*: any float
  satisfies `Dtype("float64")`, any integer `Dtype("int32")` — enough to catch an
  int/float or bool/float mix-up without firing on `float64` vs `float32`. Pass
  `exact=True` to require the precise dtype (e.g. to pin memory footprint or a typed
  sink).

The declared markers of a whole function are read back with
[`schema_from_signature`](#reading-declarations-schema_from_signature).
`assert_valid_schema(marker, context)` fails fast at declaration time — an unparseable
dtype string or duplicate dim names raise `ValueError` immediately.

### Applying schema: `declare_schema`

`@declare_schema` reads the declared markers off the signature and, on each call,
validates every declared `DataArray` input and output. **It never mutates** — arrays pass
through unchanged; a mismatch raises, warns, or is ignored per policy.

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.schema import declare_schema, Dims, Dtype

@declare_schema
def standardise(
    x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64")],
) -> Annotated[xr.DataArray, Dims("time", "x")]:
    return x
```

A `TypedDict` or `dataclass` return is validated per-field; a bare
`Annotated[DataArray, ...]` return is validated directly. Non-`DataArray` arguments and
returns pass through untouched.

Override the policy per function:

```python
@declare_schema(on_mismatch="warn")
def lenient(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray: ...
```

### The schema policy

Because structural validation never converts, the schema policy has a single behavioural
axis on top of the shared [`enabled`](#concepts) switch: **`on_mismatch`**, governing what
happens when an array doesn't match a declaration.

| `on_mismatch`     | on a structural mismatch                          |
|-------------------|---------------------------------------------------|
| `error` (default) | raises `SchemaError`                              |
| `warn`            | emits `SchemaWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged              |

The axis resolves via `XARRAY_ANNOTATED_SCHEMA_ON_MISMATCH` (default `error` — a
structural mismatch usually signals a genuine wiring bug) as described under
[Concepts](#concepts). `SchemaError` is deliberately **not** a `ValueError`, so catching a
mismatch never accidentally swallows a malformed-declaration `ValueError`.

**Per-marker override.** Any marker may carry its own `on_mismatch`, which wins over the
decorator/call argument and the policy default — so a wrong dtype can be a warning while a
wrong set of dims stays an error:

```python
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64", on_mismatch="warn")]
```

Effective severity resolves: **marker override → decorator/call argument → policy
default**.

### Validating directly: `check_schema`

Like `check_units`, the primitive is public — use it to validate an array by hand:

```python
check_schema(da, declared, name, on_mismatch=None, qualname=None)
```

`declared` is a marker or a list of markers; `check_schema` runs each under the effective
severity and returns `da` **unchanged** (or raises `SchemaError`). `name` labels the array
in messages, and it is a total no-op when the policy is disabled.

### Reading declarations: `schema_from_signature`

Mirroring `units_from_signature`, `schema_from_signature` extracts a function's declared
markers without calling it:

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

Each declaration is the *list* of markers on that parameter/field, since a hint may
declare several. `TypedDict`/`dataclass` returns are read per-field, exactly as for units.
