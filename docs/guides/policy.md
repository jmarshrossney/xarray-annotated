# Configuring validation

Declaring a property says what you expect. This page is about the other half: what
happens when the expectation isn't met, and how to change it — per marker or per
function.

Each domain — `schema`, `units`, `temporal` — has a few **axes**, each governing the
response to one validation event. Behaviour is configured **only where it is declared**:
there is no process-wide setting and no environment variable that loosens or tightens a
check. The one global is the switch that turns the package [off](#enabled).


## How a setting is resolved { #resolution }

Every axis resolves **once per call**, taking the first of:

1. a **per-marker** override — `Dtype("float64", on_mismatch="warn")`,
   `Unit("degC", convert=True)`;
2. a **per-function** argument — `@xra.enforce(units_on_output="require")`, named
   `<domain>_<axis>`, or the unprefixed keyword passed to a `check_*` primitive;
3. the built-in default.

The more specific statement wins: a marker override is a claim about *this particular
declaration*, a keyword about the whole function. Only a few axes have a per-function
keyword; in particular, a mismatch can only be softened on the marker it applies to, so
every relaxed contract is stated where it is declared. Either way, reading the signature and
its decorator tells you how a function validates.

Not every axis is available on a marker. Every units axis and every `on_mismatch` can be;
`temporal`'s `on_uninferable` cannot, since a short or irregular axis is a property of the
data rather than of the declaration.

The defaults are chosen to be right for most code:

| Domain   | Axis             | Default   |
|----------|------------------|-----------|
| schema   | `on_mismatch`    | `error`   |
| units    | `convert`        | `False`   |
| units    | `on_output`      | `stamp`   |
| temporal | `on_mismatch`    | `error`   |
| temporal | `on_uninferable` | `warn`    |


## Switching the package off { #enabled }

`xra.enabled(False)` makes **every** decorator and `check_*` primitive a total no-op for
the block: no validation, no conversion, no stamping, and no per-call overhead beyond the
wrapper itself. The environment variable `XARRAY_ANNOTATED_ENABLED` does the same for a
whole process (`0`, `false`, `no` or `off`).

```python
import xarray_annotated as xra

with xra.enabled(False):
    ...
```

The block **beats the environment variable in both directions**: `xra.enabled(True)`
restores checking inside a process started with `XARRAY_ANNOTATED_ENABLED=false`. A block
applies to the current thread or async task only, and the innermost block wins. There is
deliberately no process-wide setter: a call at import scope that changes behaviour is the
kind of setting this package keeps out of the code.

!!! warning "Off means off, conversion included"

    `@xra.enforce` normally *converts* inputs and *stamps* outputs. When switched off that
    stops too, so a function declaring `Unit("Pa")` will happily receive `hPa` and return
    an array whose `units` attribute is whatever the body left there.

    Switching off is appropriate for a hot loop over data you have already validated at
    the boundary, or for tests that are not about units. If a test passes only with the
    package switched on, the code under test depends on a conversion — worth knowing. To
    make a failing pipeline pass, switch the axis on the declaration instead
    (`on_mismatch="warn"`), which keeps you informed.

Declaration checks are the exception: a typo'd unit or an unparseable dtype still raises
at decoration time, even when switched off. A malformed declaration is a bug in your
source, not a property of the data.


## Schema policy { #schema }

Because structural validation never converts, the schema domain has a single axis,
shared by all three structural markers (`Dims`, `Coords`, `Dtype`).

### `on_mismatch` { #schema-on_mismatch }

| `on_mismatch`     | on a structural mismatch                           |
|-------------------|----------------------------------------------------|
| `error` (default) | raises `SchemaError`                               |
| `warn`            | emits `SchemaWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged               |

The default is `error`, because a structural mismatch usually signals a genuine wiring bug rather than a
tolerance question.

`SchemaError` is deliberately **not** a `ValueError`, so catching a mismatch never
accidentally swallows a malformed-declaration `ValueError`.

Set it on the marker — so a wrong dtype can be a warning while a wrong set of dims stays
an error:

```python
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64", on_mismatch="warn")]
```


## Units policy { #units }

Inputs have an opt-in to conversion and no axis; outputs have one axis, `on_output`. Two
input failures are not configurable at all, and always raise:

- a **dimensional** mismatch (e.g. `"kg"` where `"Pa"` is declared) raises
  `pint.DimensionalityError`, because the values cannot be converted;
- a **missing**, unparseable or over-long `units` attribute raises `ValueError`, because
  the declaration cannot be checked. Attach the unit where the data enters the pipeline,
  or [switch checking off](#enabled) while prototyping.

### `convert` — value-changing conversion is opt-in { #units-convert }

An input whose unit is compatible with the declaration but *not identical* — any
conversion that would change the values, including affine ones like `"K"` → `"degC"` —
raises `ValueError` unless the declaration opts in:

```python
@xra.enforce
def partition(
    tair: Annotated[xr.DataArray, Unit("degC", convert=True)],
    total: Annotated[xr.DataArray, Unit("g m-2 yr-1")],
) -> xr.DataArray: ...
```

Whether a conversion is *correct* is a property of the quantity, not of the function or
the process, and one function can want both answers. Here the respiration term is written
for Celsius, so kelvin input should convert. An annual total is dimensionally compatible
with a daily mean, but converting one into the other would silently scale it by 365.25,
so it is refused.

Equivalent spellings of the same unit (`"pascal"` for `"Pa"`) imply no value change and
are always accepted. `convert` has no per-function keyword, and the bare-string shorthand
cannot opt in. Every conversion is [logged](#auditing) at `DEBUG`.

### `on_output` — trusting a declared return value { #units-on_output }

The settings above govern **inputs**. Outputs are treated differently: `@xra.enforce`
*stamps* a declared return — it overwrites `attrs["units"]` with the declared unit —
rather than checking it.

That asymmetry is forced, not an oversight. xarray's `attrs` are inert under arithmetic,
so a body that converts by scalar multiplication returns an array still carrying its
**input's** label:

```python
@xra.enforce
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p * 100.0  # values are Pa; attrs still say "hPa"
```

The values are right and the label is stale. Converting on the strength of that label
would multiply by 100 a second time, so stamping is the only safe default.

| `on_output`       | on a declared return value                                     |
|-------------------|----------------------------------------------------------------|
| `stamp` (default) | overwrites `attrs["units"]` with the declared unit, no checks   |
| `strict`          | a present, parseable label that differs from the declaration raises |
| `require`         | as `strict`, and an output with no parseable label of its own raises instead of being stamped |

The rows describe a return value labelled through `attrs`. A pint-quantified return is
*converted* to the declaration instead of stamped, under every setting (`strict` raises
rather than converting) — see [Quantified arrays](troubleshooting.md#quantified-arrays).

The default is `stamp`.

!!! warning "`strict` is for unit-aware bodies only"

    A body doing manual scalar arithmetic — like `to_pascals` above — fails `strict`
    **whether or not it is correct**, because its stale label is indistinguishable from a
    wrong one. `strict` suits bodies that maintain truthful units: pass-through and
    subsetting functions, or computations on pint-quantified arrays (whose units really
    are checked, since a `Quantity` cannot carry a stale label).

    A manual-arithmetic body can opt in by clearing its own label (`out.attrs.pop("units",
    None)`), since an absent label is always stamped.

Note what no setting here can do: **outputs are never inspected by value.** A forgotten
conversion factor produces the same label as a correct one, so it is invisible to this
axis — see [Troubleshooting](troubleshooting.md#a-forgotten-conversion-is-not-caught-at-the-producer).

### Wrapping a library that returns bare arrays { #wrapping-a-library }

A function wrapping a third-party library is in a different position from one doing its
own arithmetic. Its output declaration is not a description of its body but a claim about
someone else's convention, and the author can simply be wrong about it. Under `stamp` or
`strict` a bare return is **taken on trust**: it is stamped with the declared unit, and
every consumer downstream then compares that declaration with its own, never with the
values.

Attach the unit the library actually returns, at the boundary, and quantify:

```python
@xra.enforce(units_on_output="require")
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
    raw = some_library.gpp(ppfd.values)  # documented as ug m-2 s-1
    out = xr.DataArray(raw, dims=ppfd.dims, coords=ppfd.coords)
    return out.assign_attrs(units="ug m-2 s-1").pint.quantify()
```

The declared unit is now checked against something real: a compatible unit is converted
(here by a factor of 86.4e-3), and an incompatible one raises `DimensionalityError`.
`require` makes forgetting this impossible to do silently — a bare return raises, with a
message pointing at `.pint.quantify()`. So does returning something that is not a
`DataArray` at all (the library's raw numpy array, `None`, or a `TypedDict` missing the
declared key), and a quantified return whose unit the active registry cannot parse. See
[Configuring a module](#for-a-module) to apply it to a whole module of wrappers.


## Temporal policy { #temporal }

A frequency declaration can fail in two genuinely different ways, so the domain has two
axes.

### `on_mismatch` — the declaration was violated { #temporal-on_mismatch }

The axis has a frequency, and it is not the declared one. An array with no datetime
coordinate, or an ambiguous pair of them, is reported here too.

| `on_mismatch`     | on a frequency mismatch                          |
|-------------------|--------------------------------------------------|
| `error` (default) | raises `FreqError`                               |
| `warn`            | emits `FreqWarning`, returns the array unchanged |
| `ignore`          | silently returns the array unchanged             |

The default is `error`. Also settable per marker: `Freq("D", on_mismatch="warn")`.

`FreqError` is deliberately **not** a `ValueError`, so catching a mismatch never
accidentally swallows a malformed-declaration `ValueError` (an unparseable offset string
raises the latter, at decoration time).

### `on_uninferable` — the declaration was never tested { #temporal-on_uninferable }

No frequency could be determined at all: fewer than three timestamps, or irregular
spacing. The declaration was not violated; it was never *tested*.

| `on_uninferable` | on an uninferable time axis                      |
|------------------|--------------------------------------------------|
| `error`          | raises `FreqError`                               |
| `warn` (default) | emits `UninferableFreqWarning`, returns the array unchanged |
| `ignore`         | silently returns the array unchanged             |

The default is `warn`, not `error`, because a short axis is legitimate — a two-timestep
test fixture — but silently skipping a contract check deserves noise. In a test suite
whose fixtures are short on purpose, silence it with a warnings filter rather than a
setting:

```toml
[tool.pytest.ini_options]
filterwarnings = ["ignore::xarray_annotated.temporal.UninferableFreqWarning"]
```

`UninferableFreqWarning` subclasses `FreqWarning`, so the filter leaves `on_mismatch="warn"`
warnings — genuine contract violations — visible, and existing `FreqWarning` filters still
match both.


## Configuring a module { #for-a-module }

A per-function keyword can be bound once with `functools.partial`, giving a module its own
decorator. This suits a module whose functions share a role, such as wrappers around a
library that returns bare arrays:

```python
import functools

import xarray_annotated as xra

enforce = functools.partial(xra.enforce, units_on_output="require")


@enforce
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]: ...
```

It is a per-function argument, so a marker's own override still beats it.


## Auditing what the package does { #auditing }

Every non-trivial action is logged through the standard `logging` module, to the
`xarray_annotated` logger and one child per domain (`xarray_annotated.units`,
`.schema`, `.temporal`). Nothing is logged at `WARNING` or above, so with logging's
defaults this is silent; turn it up to see everything:

```python
import logging

logging.basicConfig()
logging.getLogger("xarray_annotated").setLevel(logging.DEBUG)
```

`INFO` is what an audit is usually looking for: a schema or frequency check that failed or
could not run being let through. An `"ignore"` is always chosen, but a contract that went
unchecked is still worth seeing. `DEBUG` adds chosen or routine actions, including every conversion,
since conversion is always opted into.

| Event                                                              | Level   |
|--------------------------------------------------------------------|---------|
| schema or frequency mismatch passed under `"ignore"`               | `INFO`  |
| input converted under `convert=True`                               | `DEBUG` |
| output stamped over a different label                              | `DEBUG` |
| quantified output converted to its declaration                     | `DEBUG` |

Same-unit respellings (`"pascal"` → `"Pa"`) are not logged, and nothing is logged when the
package is switched off. Unlike a warning, a log record is emitted on every call rather
than once per call site, so a conversion inside a loop shows up every time.


## Not a policy: the unit registry

Which unit strings *parse* is a separate global setting, and not part of any policy —
see [Choosing a unit registry](registry.md). You need it only for CF-convention units
such as `"umol m-2 s-1"`.
