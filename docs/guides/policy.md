# Configuring validation

Declaring a property says what you expect. This page is about the other half: what
happens when the expectation isn't met, and how to change it — per marker, per function,
per block, or for a whole process.

Each of the three domains — `schema`, `units`, `temporal` — has its own **policy**: a
small set of axes governing the response to a validation event. They share one master
switch, `enabled`.


## How a setting is resolved { #resolution }

Every axis resolves **once per call**, taking the first of:

1. a **per-marker** override — `Dtype("float64", on_mismatch="warn")`;
2. a **per-function** argument — `@declare_schema(on_mismatch="warn")`, or the same
   keyword passed to a `check_*` primitive;
3. the **environment variable** for that axis;
4. a process-wide override set with `set_policy(...)` or scoped with `policy(...)`;
5. the built-in default.

The rule of thumb: the more specific the statement, the more it wins. A marker override
is a claim about *this particular declaration* and beats everything; the built-in default
is a claim about nothing in particular and loses to everything.

Not every axis is available at every level — only `on_mismatch` can be set on a marker,
since the other axes describe situations a single declaration can't anticipate.


## The `enabled` switch { #enabled }

`enabled` is shared by all three domains. Setting it to `False` makes **every** decorator
a total no-op: no validation, no conversion, no stamping, and no per-call overhead beyond
the wrapper itself.

```python
from xarray_annotated.units import policy as units_policy

with units_policy(enabled=False):   # disables *every* decorator for the block
    ...
```

The switch lives in one place, so reaching for it through any domain's `policy` or
`set_policy` sets it for all of them. Its environment variable is
`XARRAY_ANNOTATED_ENABLED`.

!!! warning "Disabling is not free of consequences"

    `@declare_units` normally *converts* inputs and *stamps* outputs. With `enabled=False`
    that stops too, so a function declaring `Unit("Pa")` will happily receive `hPa` and
    return an array whose `units` attribute is whatever the body left there.

    Disabling is appropriate for a hot loop over data you have already validated at the
    boundary. It is not a way to make a failing pipeline pass — for that you want
    `on_mismatch="warn"`, which keeps you informed.

Declaration checks are the exception: a typo'd unit or an unparseable dtype still raises
at decoration time, regardless of this switch. A malformed declaration is a bug in your
source, not a property of the data.


## Schema policy { #schema }

Because structural validation never converts, the schema policy has a single behavioural
axis on top of `enabled`, shared by all three structural markers (`Dims`, `Coords`,
`Dtype`).

### `on_mismatch` { #schema-on_mismatch }

| `on_mismatch`     | on a structural mismatch                           |
|-------------------|----------------------------------------------------|
| `error` (default) | raises `SchemaError`                               |
| `warn`            | emits `SchemaWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged               |

Environment variable: `XARRAY_ANNOTATED_SCHEMA_ON_MISMATCH`. The default is `error`,
because a structural mismatch usually signals a genuine wiring bug rather than a
tolerance question.

`SchemaError` is deliberately **not** a `ValueError`, so catching a mismatch never
accidentally swallows a malformed-declaration `ValueError`.

Per function:

```python
@declare_schema(on_mismatch="warn")
def lenient(x: Annotated[xr.DataArray, Dims("time", "x")]) -> xr.DataArray: ...
```

Per marker — so a wrong dtype can be a warning while a wrong set of dims stays an error:

```python
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64", on_mismatch="warn")]
```


## Units policy { #units }

Three behavioural axes on top of `enabled`. Note that **dimensional** mismatches (e.g.
`"kg"` where `"Pa"` is declared) are not governed by either — they always raise
`pint.DimensionalityError`.

Both may be overridden per function (each defaults to the active policy when omitted):

```python
@declare_units(on_missing="error", on_inexact="error")
def strict_node(x: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray: ...
```

### `on_missing` — no parseable unit to check against { #units-on_missing }

Governs only the "can't validate" cases: a missing or unparseable `units` attribute.

| `on_missing` | missing/unparseable units                     | dimensional mismatch |
|--------------|-----------------------------------------------|----------------------|
| `error`      | raises `ValueError`                           | always raises        |
| `warn`       | emits `UnitsWarning`, returns input unchanged | always raises        |
| `ignore`     | silently returns input unchanged              | always raises        |

Environment variable: `XARRAY_ANNOTATED_UNITS_ON_MISSING` (default `warn`).

### `on_inexact` — value-changing conversion { #units-on_inexact }

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

Environment variable: `XARRAY_ANNOTATED_UNITS_ON_INEXACT` (default `convert`).

### `on_output` — trusting a declared return value { #units-on_output }

The two axes above govern **inputs**. Outputs are treated differently: `@declare_units`
*stamps* a declared return — it overwrites `attrs["units"]` with the declared unit —
rather than checking it.

That asymmetry is forced, not an oversight. xarray's `attrs` are inert under arithmetic,
so a body that converts by scalar multiplication returns an array still carrying its
**input's** label:

```python
@declare_units
def to_pascals(p: Annotated[xr.DataArray, Unit("hPa")]) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p * 100.0     # values are Pa; attrs still say "hPa"
```

The values are right and the label is stale. Converting on the strength of that label
would multiply by 100 a second time, so stamping is the only safe default.

| `on_output`       | on a declared return value                                     |
|-------------------|----------------------------------------------------------------|
| `stamp` (default) | overwrites `attrs["units"]` with the declared unit, no checks   |
| `strict`          | a present, parseable label that differs from the declaration raises |

Environment variable: `XARRAY_ANNOTATED_UNITS_ON_OUTPUT` (default `stamp`).

!!! warning "`strict` is for unit-aware bodies only"

    A body doing manual scalar arithmetic — like `to_pascals` above — fails `strict`
    **whether or not it is correct**, because its stale label is indistinguishable from a
    wrong one. `strict` suits bodies that maintain truthful units: pass-through and
    subsetting functions, or computations on pint-quantified arrays.

    A manual-arithmetic body can opt in by clearing its own label (`out.attrs.pop("units",
    None)`), since an absent label is always stamped.

Note what no setting here can do: **outputs are never inspected by value.** A forgotten
conversion factor produces the same label as a correct one, so it is invisible to this
axis — see [Troubleshooting](troubleshooting.md#a-forgotten-conversion-is-not-caught-at-the-producer).


## Temporal policy { #temporal }

A frequency declaration can fail in two genuinely different ways, so the policy has two
behavioural axes on top of `enabled`.

### `on_mismatch` — the declaration was violated { #temporal-on_mismatch }

The axis has a frequency, and it is not the declared one. An array with no datetime
coordinate, or an ambiguous pair of them, is reported here too.

| `on_mismatch`     | on a frequency mismatch                          |
|-------------------|--------------------------------------------------|
| `error` (default) | raises `FreqError`                               |
| `warn`            | emits `FreqWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged             |

Environment variable: `XARRAY_ANNOTATED_TEMPORAL_ON_MISMATCH`. Also settable per marker:
`Freq("D", on_mismatch="warn")`.

`FreqError` is deliberately **not** a `ValueError`, so catching a mismatch never
accidentally swallows a malformed-declaration `ValueError` (an unparseable offset string
raises the latter, at decoration time).

### `on_uninferable` — the declaration was never tested { #temporal-on_uninferable }

No frequency could be determined at all: fewer than three timestamps, or irregular
spacing. The declaration was not violated; it was never *tested*.

| `on_uninferable` | on an uninferable time axis                      |
|------------------|--------------------------------------------------|
| `error`          | raises `FreqError`                               |
| `warn` (default) | emits `FreqWarning`, returns the array unchanged |
| `ignore`         | silently returns the array unchanged             |

Environment variable: `XARRAY_ANNOTATED_TEMPORAL_ON_UNINFERABLE`. The default is `warn`,
not `error`, because a short axis is legitimate — a two-timestep test fixture — but
silently skipping a contract check deserves noise.


## Scoping an override { #scoping }

### For a block

Each domain exports a `policy(...)` context manager, which restores the previous settings
on exit. This is the right tool for a notebook cell, a test, or a single stage of a
pipeline:

```python
from xarray_annotated.units import policy

with policy(on_missing="error", on_inexact="warn"):
    ...
```

### For a process

`set_policy(...)` applies until changed, and is intended to be called once at startup:

```python
from xarray_annotated.schema import set_policy

set_policy(on_mismatch="warn")
```

Each domain has its own `get_policy()` / `set_policy()` / `policy()` trio, importable from
`xarray_annotated.schema`, `.units` and `.temporal`. Import them under an alias if you
need more than one in the same module.

### From the environment

Useful when the choice belongs to the deployment rather than the code:

| Variable                                    | Domain   | Default     |
|---------------------------------------------|----------|-------------|
| `XARRAY_ANNOTATED_ENABLED`                  | all      | `true`      |
| `XARRAY_ANNOTATED_SCHEMA_ON_MISMATCH`       | schema   | `error`     |
| `XARRAY_ANNOTATED_UNITS_ON_MISSING`         | units    | `warn`      |
| `XARRAY_ANNOTATED_UNITS_ON_INEXACT`         | units    | `convert`   |
| `XARRAY_ANNOTATED_UNITS_ON_OUTPUT`          | units    | `stamp`     |
| `XARRAY_ANNOTATED_TEMPORAL_ON_MISMATCH`     | temporal | `error`     |
| `XARRAY_ANNOTATED_TEMPORAL_ON_UNINFERABLE`  | temporal | `warn`      |


## Not a policy: the unit registry

Which unit strings *parse* is a separate global setting, and not part of any policy —
see [Choosing a unit registry](registry.md). You need it only for CF-convention units
such as `"umol m-2 s-1"`.
