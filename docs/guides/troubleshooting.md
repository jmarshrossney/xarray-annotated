# Troubleshooting

Failure modes that are silent, or whose message doesn't obviously name its cause.


## A declaration is silently ignored { #a-declaration-is-silently-ignored }

**Symptom.** A decorated function validates nothing. No error, no warning; units aren't
converted and structural mismatches sail through. The parameter simply behaves as if it
had no annotation at all.

**Cause.** The declaration was aliased with a PEP 695 `type` statement:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]  # ❌ ignored


@xra.enforce
def f(p: Pressure) -> xr.DataArray: ...  # hPa in, hPa out — nothing happened
```

A `type` alias is *lazy*. `get_type_hints(..., include_extras=True)` — how every reader in
this package inspects a signature — hands back the alias object itself rather than the
`Annotated` it wraps, so the markers inside are never seen.

**Fix.** Use a plain assignment. It is substituted eagerly, so the markers survive and
every decorator reads them as if they had been written out in full:

```python
Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]  # ✅ read
```

This is the one failure in this package that is *completely* silent, which is why it's
first on this page. If a decorator seems inert, check for `type` before anything else.


## `NameError` at decoration time { #nameerror-at-decoration-time }

**Symptom.** Importing a module raises `NameError` from inside the `@xra.enforce`
decorator, often naming `xr` or another type that is plainly imported somewhere.

**Cause.** `from __future__ import annotations` in that module. Declarations are read as
*runtime objects* out of the `Annotated` metadata; that import stringizes annotations,
forcing a re-`eval` that fails whenever a needed name — e.g. a `TYPE_CHECKING`-only
`xarray` import — isn't resolvable at runtime.

**Fix.** Remove `from __future__ import annotations` from modules that declare markers, and
make sure any name used in an annotation is a real runtime import rather than a
`TYPE_CHECKING`-only one. Python 3.14's deferred-annotation model (PEP 649/749) removes
this constraint.


## `ValueError: ... is not a recognised unit` { #unrecognised-unit }

**Symptom.** At import, not at call time:

```
ValueError: g input 'x': declared unit 'umol m-2 s-1' is not a recognised unit
  (...) (call use_cf_units() to enable CF/UDUNITS units like 'umol m-2 s-1')
```

**Cause.** Either a genuine typo, or — for CF-convention strings like `"umol m-2 s-1"` and
`"g m-2 d-1"` — the default plain-pint registry, which doesn't understand UDUNITS
spellings.

**Fix.** For CF strings, install the [`[cf]` extra](../getting_started.md#installation) and
call `use_cf_units()` once at startup, before the modules that declare those units are
imported. See [Choosing a unit registry](registry.md) — the choice
is process-global, so it's pint units *or* CF units for the whole codebase.

This class of error is raised at *decoration* time by design, regardless of policy: a
malformed declaration is a bug in your source, not a property of your data. The same
applies to an invalid dtype (`ValueError: invalid dtype 'flaot64'`) or an unparseable
offset string.


## `UnitsWarning: ... unvalidated: no 'units' attribute` { #no-units-attribute }

**Symptom.**

```
UnitsWarning: [f] input 'x' unvalidated: no 'units' attribute (declared 'Pa')
```

The array passed through unconverted.

**Cause.** `@xra.enforce` validates units by reading `da.attrs["units"]`. With nothing to read,
it cannot confirm or convert anything. Note that xarray drops `attrs` through most
arithmetic by default, so an array that had units upstream may well not have them here.

A pint-quantified array does **not** trigger this: its unit lives in the data rather than
in `attrs`, and is read from `da.pint.units` instead. See
[Quantified arrays](#quantified-arrays).

**Fix.** Depends on what you want the warning to mean:

- stamp the unit at the point the array enters your code, so downstream checks have
  something to work with;
- set `xr.set_options(keep_attrs=True)` if the units are being lost to arithmetic;
- if unlabelled input is legitimate and you're happy to trust it, quieten the axis with
  [`on_missing="ignore"`](policy.md#units-on_missing);
- if it should never happen, promote it with `on_missing="error"`.

**Not** the same thing as a dimensional mismatch, which always raises regardless of this
setting.


## Pint-quantified arrays { #quantified-arrays }

If you use [pint-xarray](https://pint-xarray.readthedocs.io), your arrays may hold a
`pint.Quantity` rather than a plain ndarray, with the unit in the data and `attrs` empty.
These are supported, but the two directions behave differently, because a `Quantity`'s unit
travels with the values and so is always truthful, whereas an `attrs` label is not.

**As an input**, the unit is read from `da.pint.units`, converted to the declaration, and
the array handed to the function body **dequantified** with `attrs["units"]` set — the same
shape of value a plain input produces, so bodies need no special case. A leftover
`attrs["units"]` on a quantified array is ignored in favour of the `Quantity`.

**As a return value**, the array is *converted* to the declaration rather than stamped, and
comes back **still quantified**, with `attrs` untouched:

- `on_output="stamp"` converts if needed, and raises `DimensionalityError` if it cannot.
  This is the one situation where `stamp` can raise — for a `Quantity` a mismatch is a real
  error, not the expected staleness that makes stamping the right default for `attrs`.
- `on_output="strict"` raises on any unit other than the declared one.

If a function must return something dequantified — to put it in a frozen dataclass, say —
call `.pint.dequantify()` in the body.


## `DimensionalityError` that no policy will suppress { #dimensionalityerror }

**Symptom.**

```
DimensionalityError: Cannot convert from 'kilogram' ([mass])
  to 'pascal' ([mass] / [length] / [time] ** 2)
```

…and setting `on_missing` to `"ignore"` or `convert=True` doesn't help.

**Cause.** Working as intended. The units policy governs cases where validation is
*uncertain* — an absent unit, a lossy conversion. A dimensional mismatch is not uncertain:
there is no reading of the call under which mass was meant to be a pressure.

**Fix.** Fix the data or the declaration. If you genuinely need the call to proceed, the
only switch that will do it is [`xra.enabled(False)`](policy.md#enabled), which disables
all validation — read the warning there before reaching for it.


## A forgotten conversion is not caught at the producer { #a-forgotten-conversion-is-not-caught-at-the-producer }

**Symptom.** A function declares an output unit, its body omits the conversion that would
make that true, and nothing complains. The returned array is labelled correctly and holds
the wrong numbers.

```python
@xra.enforce
def daily_carbon(
    flux: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
    return flux.resample(time="D").mean()  # forgot * MOLAR_FACTOR
```

**Cause.** Outputs are [stamped, not checked](policy.md#units-on_output), and stamping
cannot help here even in principle. The correct body (`... * MOLAR_FACTOR`) and the buggy
one leave *identical* `units` attributes, because multiplying by a float cannot update a
string. The only thing distinguishing them is the values, which are never inspected.
`on_output="strict"` does not close this gap — it compares labels, and the labels agree.

**Fix.** Declare the quantity on whoever **consumes** it. An input declaration is checked,
so the wrong quantity is caught at the first boundary that expects the right one:

```python
@xra.enforce
def annual_budget(
    daily: Annotated[xr.DataArray, Unit("g m-2 d-1")],
) -> Annotated[xr.DataArray, Unit("g m-2 yr-1")]:
    return daily.sum()
```

Two caveats on relying on that. It needs `xr.set_options(keep_attrs=True)`, or the label
will have been dropped by arithmetic before it arrives. And it only fires when the
intermediate array reaches the consumer *without* having been stamped by a decorator in
between — a stamped array asserts the declared unit and will be believed.

This is a real limit rather than a bug: unit metadata records what an array *claims* to be,
and no amount of checking that claim can verify arithmetic. Declarations catch wiring
mistakes, not algebra.


## A wrapper's output units are wrong, and every check passes { #wrapper-output-units }

**Symptom.** A function wrapping a third-party library declares an output unit, the
library actually returns something else, and nothing complains — not at the wrapper, and
not at any consumer that declares the same unit.

**Cause.** The library returns bare arrays, and a bare declared output is stamped with the
declared unit under the default `on_output="stamp"` (and under `"strict"`). The
declaration is then the array's only unit, so every check downstream compares one
declaration with another. A consumer that declares the same wrong unit agrees with it.

**Fix.** Attach the unit the library documents, where its output enters, and quantify:
`xr.DataArray(raw).assign_attrs(units="ug m-2 s-1").pint.quantify()`. The declaration is
then checked against it and converted to. Set `units_on_output="require"` on wrappers so a
bare return raises instead of being stamped — see
[Wrapping a library that returns bare arrays](policy.md#wrapping-a-library).


## Validation stopped happening everywhere { #validation-stopped-happening }

**Symptom.** Checks that used to fire don't, across every domain at once, in one
environment but not another.

**Cause.** The package is [switched off](policy.md#enabled) — most often
`XARRAY_ANNOTATED_ENABLED` set in a deployment environment or a `.env` file, or a call
running inside an `xra.enabled(False)` block.

**Fix.** Check the environment variable first. An `xra.enabled(...)` block beats it, so
code wrapped in `xra.enabled(True)` is checked whatever the environment says.

If the goal was to reduce noise rather than to disable checking, set
`on_mismatch="warn"` on the declaration — when switched off `@xra.enforce` also stops
converting inputs and stamping outputs, so arrays keep whatever units they arrived with.
