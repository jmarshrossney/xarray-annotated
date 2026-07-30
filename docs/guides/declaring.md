# Declaring properties

`xarray-annotated` lets you declare a property of a `DataArray` in a function
signature with `typing.Annotated`, then validate it automatically with a decorator.
There are five kinds of property, in three groups:

- **structural** — **dims**, **coords**, and **dtype** — checked (never mutated) by
  `@declare_schema`;
- **physical units** — checked *and converted* (via pint/CF) by `@declare_units`;
- **temporal frequency** — the spacing *and phase* of a time axis — checked (never
  mutated) by `@declare_freq`.

This page covers what each marker declares and what it considers a match. What happens
when a declaration is *violated* — raise, warn, or ignore — is a separate, cross-cutting
concern: see [Configuring validation](policy.md).


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
by hand; see [Reading and writing declarations](tooling.md).

**Fail fast at decoration.** Each decorator validates its *declarations* when it is
applied (at import) — a typo'd unit or an unparseable dtype raises immediately, rather
than only when the function is first called, and regardless of policy.

**Declarations are read as runtime objects.** The markers are pulled out of the
`Annotated` metadata by `get_type_hints(..., include_extras=True)`, not parsed from
source. Two consequences are worth knowing before you hit them: don't use
`from __future__ import annotations` in a module that declares markers, and alias a
reusable declaration with `=` rather than a PEP 695 `type` statement. Both are covered
in [Troubleshooting](troubleshooting.md).

**Policy governs the consequences.** Every check resolves a *policy* deciding whether a
violation raises, warns, or is silently ignored — overridable per marker, per function,
per block, or process-wide, and switchable off entirely. Each section below states its
default; [Configuring validation](policy.md) has the full set of axes, the environment
variables, and the scoping tools.


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

### Strictness

**`Dims(*names, ordered=False)`** — by default the *set* of dims must match (extra or
missing dims fail); order is free, because xarray operations are order-independent until
you drop to numpy. Pass `ordered=True` to also pin the order (e.g. before `.values`,
`.stack`, or `apply_ufunc`):

```python
Annotated[xr.DataArray, Dims("time", "x", ordered=True)]
```

A mismatch raises `SchemaError` by default; see
[the schema policy](policy.md#schema-on_mismatch) to warn or ignore instead, or to set a
different severity on one marker.

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

Severity follows the shared [schema policy](policy.md#schema-on_mismatch): a missing
coordinate raises `SchemaError` by default.

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

Severity follows the shared [schema policy](policy.md#schema-on_mismatch).

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

On each call, under the active [units policy](policy.md#units), `@declare_units` validates
and converts every declared `DataArray` **input**, runs the function, then stamps each
declared `DataArray` **output** with its unit. A `TypedDict` or `dataclass` return is
stamped per-field; a bare `Annotated[DataArray, ...]` return takes that unit.
Non-`DataArray` arguments and returns pass through untouched.

Inputs and outputs are treated differently, and the difference matters: an input is
**checked**, an output is **stamped**. The decorator trusts a function to return what it
declared, because a body that converts by arithmetic leaves `attrs` stale and re-checking
it would do more harm than good. The consequence worth remembering is that a declaration
on a *parameter* buys you verification, while a declaration on a *return* buys you a
label — so the useful place to declare a quantity is on whoever consumes it. The
[`on_output`](policy.md#units-on_output) axis can tighten this for bodies that maintain
their own units.

`Unit` is the **recommended** form: it owns its own slot in the metadata, so it stays
unambiguous and order-independent even when other `Annotated`-based tooling — or a schema
marker (see [Combining multiple checks](#combining-multiple-checks)) — shares the
annotation.

Out of the box, unit strings are parsed by plain pint. CF-convention strings such as
`"umol m-2 s-1"` need cf-xarray's UDUNITS-aware registry — a one-time startup choice,
covered under [Choosing a unit registry](registry.md).

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

### What counts as a match

A **dimensional** mismatch — `"kg"` where `"Pa"` is declared — always raises
`pint.DimensionalityError`, regardless of policy. There is no reading of that call under
which the caller meant what they wrote.

Everything else is negotiable and governed by the [units policy](policy.md#units): a
compatible-but-different unit is converted (`"hPa"` → `"Pa"`) by default, and a missing or
unparseable `units` attribute warns.

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

A violated declaration raises `FreqError` by default. A time axis whose frequency can't be
determined at all — fewer than three timestamps, or irregular spacing — is a separate case
that *warns* by default; see [the temporal policy](policy.md#temporal).

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

## Reusing a declaration

A declaration you write into several signatures is naturally worth naming. Give it a
**plain assignment**:

```python
Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]

@declare_units
@declare_schema
def f(p: Pressure) -> Pressure: ...
```

Do **not** use a PEP 695 `type` statement for this — `type Pressure = ...` is read lazily
and the markers inside are silently never seen. See
[Troubleshooting](troubleshooting.md#a-declaration-is-silently-ignored).
