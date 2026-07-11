# Usage

`xarray-annotated` lets you declare a property of a `DataArray` in a function
signature with `typing.Annotated`, then validate it automatically with a decorator.
There are five kinds of property, in three groups:

- **structural** — **dims**, **coords**, and **dtype** — checked (never mutated) by
  `@declare_schema`;
- **physical units** — checked *and converted* (via pint/CF) by `@declare_units`;
- **temporal frequency** — the spacing *and phase* of a time axis — checked (never
  mutated) by `@declare_freq`.

!!! warning

    Do **not** use `from __future__ import annotations` in modules that declare units,
    schema, or frequency markers. Declarations are read as *runtime objects* out of the
    `Annotated` metadata; that import stringizes annotations, forcing a re-`eval` that
    fails (a `NameError` at decoration time) whenever a needed name — e.g. a
    `TYPE_CHECKING`-only `xarray` import — isn't resolvable at runtime. Python 3.14's
    deferred-annotation model removes this constraint.

!!! warning "Alias a declaration with `=`, not with `type`"

    A declaration you reuse in several signatures is naturally worth naming. Give it a
    **plain assignment**, not a PEP 695 `type` statement:

    ```python
    Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]      # ✅ read
    type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")] # ❌ ignored
    ```

    A `type` alias is *lazy*: `get_type_hints(..., include_extras=True)` — how every
    reader in this package inspects a signature — hands back the alias object itself
    rather than the `Annotated` it wraps, so the markers inside are never seen. The
    failure is **silent**: the parameter simply looks undeclared, and the decorator
    validates nothing. A plain assignment is substituted eagerly, so the markers survive
    and every decorator reads them as if they had been written out in full.

## Concepts

All five properties work the same way, so what you learn for one transfers to the
others.

**Declare once, in the signature.** A property is declared as `Annotated` metadata on a
`DataArray` parameter or return — `Annotated[xr.DataArray, Dims("time", "x")]` (a
structural marker), `Annotated[xr.DataArray, Unit("Pa")]` (a unit), or
`Annotated[xr.DataArray, Freq("7D")]` (a frequency). The annotation is the single source
of truth, read once and never written twice.

**Three decorators.** The structural properties — **dims**, **coords**, and **dtype** —
are validated by `@declare_schema`; physical **units** by `@declare_units`; a time axis's
**frequency** by `@declare_freq`. `schema` and `temporal` only ever *check* (arrays pass
through unchanged); `units` also *converts* (e.g. `"hPa"` → `"Pa"`). Stack the decorators
to check several properties at once — see
[Combining multiple checks](#combining-multiple-checks). Each decorator is a thin layer
over a public primitive (`check_schema` / `check_units` / `check_freq`) that you can call
by hand; see [Advanced usage](#advanced-usage).

**Fail fast at decoration.** Each decorator validates its *declarations* when it is
applied (at import) — a typo'd unit or an unparseable dtype raises immediately, rather
than only when the function is first called, and regardless of policy.

**Policy.** Each decorator follows a small **policy** governing what happens on a
validation event. Every policy shares the package-wide **`enabled`** master switch:
`enabled=False` makes *every* decorator a total no-op (no validation, conversion, or
stamping). Each axis resolves once per call, in order:

1. its environment variable,
2. a process-wide override set with `set_policy(...)`,
3. the built-in default.

Use a domain's `policy(...)` as a context manager to scope overrides across a block and
restore them on exit:

```python
from xarray_annotated.units import policy as units_policy

with units_policy(enabled=False):   # disables *every* decorator for the block
    ...
```

The `enabled` switch (env `XARRAY_ANNOTATED_ENABLED`) is shared; the behavioural axes
described under each property below are domain-specific — `on_mismatch` for schema,
`on_missing`/`on_inexact` for units, `on_mismatch`/`on_uninferable` for temporal.

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

## Frequency

### Declaring a frequency

Declare the frequency of a DataArray's time axis with the `Freq` marker, and apply
`@declare_freq` to check every declared input and output on each call. The motivating
bug is a *phase* error — a resample that silently lands on the wrong weekday:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.temporal import declare_freq, Freq

@declare_freq
def weekly_mean(
    x: Annotated[xr.DataArray, Freq("D")],
) -> Annotated[xr.DataArray, Freq("W-SUN")]:
    return x.resample(time="W-WED").mean()   # raises FreqError: expected 'W-SUN', got 'W-WED'
```

The weekly means are perfectly regular — the *spacing* is right — but they are labelled
on Wednesdays, not Sundays, so anything downstream expecting week-ending-Sunday data is
now quietly misaligned. Declaring `Freq("W-SUN")` catches it at the boundary.

`@declare_freq` never mutates: it does not resample, and it does not stamp anything onto
the array. The frequency is *derived* from the time coordinate's values (via
`xarray.infer_freq`, so `cftime` calendars — 360-day, noleap — work too), which is why it
lives in its own domain rather than as a fourth schema marker.

The time axis is auto-detected as the array's sole datetime-like coordinate. If an array
carries two, name the one you mean: `Freq("7D", dim="time")`.

Unlike units there is **no bare-string shorthand** — a plain string in the metadata is a
unit or a description, never a frequency.

### What "the same frequency" means

Two things are compared, and they behave differently.

**Spacing** is always compared, and is compared *semantically* rather than by string.
pandas will infer a seven-day axis as `"W-WED"`, never as `"7D"`, so declarations must
see through the spelling:

| Declared    | Actual  | Result | Why                                           |
|-------------|---------|--------|-----------------------------------------------|
| `Freq("7D")`  | `W-WED` | ✅ pass | seven days either way                       |
| `Freq("D")`   | `24h`   | ✅ pass | same fixed spacing                          |
| `Freq("QE")`  | `3ME`   | ✅ pass | a quarter is three months                   |
| `Freq("ME")`  | `30D`   | ❌ fail | calendar months are not fixed-length days   |

**Phase** is compared only where the declaration pins it. The `End`/`Begin` convention
(`"ME"` vs `"MS"`) is always deliberate, so it is always compared. The *anchor* — the
`-WED` in `"W-WED"`, the `-MAR` in `"QE-MAR"` — is compared only when you **spell it**:

```python
Freq("W")       # weekly, any weekday  — accepts a W-WED axis
Freq("W-SUN")   # weekly, Sundays only — rejects a W-WED axis
```

!!! note "A deliberate divergence from pandas"

    pandas silently defaults an anchor you did not spell (`to_offset("W").freqstr` is
    `"W-SUN"`). `xarray-annotated` does not: an unspelled anchor means *"any"*, because a
    declaration you didn't write is not a constraint you meant. Override the inference
    either way with `anchored=`: `Freq("W", anchored=True)` means pandas' default
    (Sundays) and means it, while `Freq("W-SUN", anchored=False)` accepts any weekday.

`freq_compatible(a, b)` applies exactly this comparison to two declarations with **no
array in hand** — for a build-time check that a producer's output frequency can satisfy a
consumer's declared input:

```python
from xarray_annotated.temporal import Freq, freq_compatible

freq_compatible(Freq("7D"), Freq("W-WED"))     # True
freq_compatible(Freq("W-SUN"), Freq("W-WED"))  # False
```

### The temporal policy

A frequency declaration can fail in two genuinely different ways, so the policy has two
behavioural axes on top of the shared [`enabled`](#concepts) switch.

**`on_mismatch`** — the axis has a frequency, and it is not the declared one (the
declaration was *violated*). An array with no datetime coordinate, or an ambiguous pair of
them, is reported here too.

| `on_mismatch`     | on a frequency mismatch                          |
|-------------------|--------------------------------------------------|
| `error` (default) | raises `FreqError`                               |
| `warn`            | emits `FreqWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged             |

**`on_uninferable`** — no frequency could be determined at all: fewer than three
timestamps, or irregular spacing. The declaration was not violated; it was never
*tested*. The default is `warn`, not `error`, because a short axis is legitimate (a
two-timestep test fixture) but silently skipping a contract check deserves noise.

| `on_uninferable` | on an uninferable time axis                      |
|------------------|--------------------------------------------------|
| `error`          | raises `FreqError`                               |
| `warn` (default) | emits `FreqWarning`, returns the array unchanged |
| `ignore`         | silently returns the array unchanged             |

The axes resolve via `XARRAY_ANNOTATED_TEMPORAL_ON_MISMATCH` and
`XARRAY_ANNOTATED_TEMPORAL_ON_UNINFERABLE` as described under [Concepts](#concepts), and
either may be overridden per function — `@declare_freq(on_uninferable="error")` — or, for
`on_mismatch`, per marker: `Freq("D", on_mismatch="warn")`. Effective severity resolves
**marker override → decorator/call argument → policy default**, exactly as for schema.
`FreqError` is deliberately **not** a `ValueError`, so catching a mismatch never
accidentally swallows a malformed-declaration `ValueError` (an unparseable offset string
raises the latter, at decoration time).

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

The same holds for all three decorators: each domain reads **only its own markers**, so
one hint can declare a unit, a structure, and a frequency at once, and the decorators can
be stacked in any order:

```python
from xarray_annotated.temporal import declare_freq, Freq

@declare_units
@declare_schema
@declare_freq
def process(
    x: Annotated[xr.DataArray, Dims("time"), Unit("degC"), Freq("D")],
) -> Annotated[xr.DataArray, Dims("time"), Unit("degC"), Freq("W-SUN")]:
    return x.resample(time="W-SUN").mean()
```

## Advanced usage

The decorators are the recommended entry point. The primitives they call are public too,
for tools that need to validate an array by hand or inspect declarations statically
(e.g. build-time checks, documentation generation, custom consumers). Most users won't
need these.

### Validating directly: `check_schema`, `check_units`, and `check_freq`

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
when `None`.

`check_freq` validates a single array's time axis and returns it **unchanged** (or raises
`FreqError`), taking the same shape of arguments:

```python
from xarray_annotated.temporal import check_freq, Freq

check_freq(da, Freq("7D"), name="da", on_mismatch=None, on_uninferable=None, qualname=None)
```

`assert_valid_unit(unit, context)` / `assert_valid_schema(marker, context)` /
`assert_valid_freq(marker, context)` provide the same fail-fast declaration checks the
decorators run at import.

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
`freq_from_signature` does the same for the `Freq` marker (one marker, or `None`, per
parameter).

### Cross-domain reader: `declarations_from_signature`

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

### Writing annotations programmatically: `annotate`

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
