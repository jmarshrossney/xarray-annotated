# Usage

This guide walks through the pieces: declaring units in signatures, applying them
with the `declare_units` decorator, the validation policy that governs both,
validating directly with `check_units`, choosing a registry, and reading declarations
back off a signature.

## Declaring units

A unit is the first `str` found in a `DataArray`'s `Annotated` metadata:

```python
Annotated[xr.DataArray, "degC"]
Annotated[xr.DataArray, "m s-1", "z component of velocity"]  # description after the unit is ignored
```

Equivalently, wrap it in the self-identifying `Unit` marker. This is the
order-independent form: the unit owns its own slot, so it stays unambiguous
even when other typed metadata shares the annotation — useful when composing
with other `Annotated`-based tooling.

```python
from xarray_signature_units import Unit

Annotated[xr.DataArray, Unit("degC")]
Annotated[xr.DataArray, "note", Unit("Pa")]  # marker wins regardless of order → "Pa"
```

Both forms resolve to the same unit string; use whichever reads better. A `Unit`
marker takes priority over a bare string when both are present, and a description
string still comes *after* the unit in the bare-string form.

The declared units of a whole function are read back with
[`units_from_signature`](#reading-declarations-off-a-signature-units_from_signature)
— the single source that both `@declare_units` and any static checker consume, so a
unit is never written twice.

`assert_valid_unit(unit, context)` fails fast at declaration time — a typo like
`"degrees_C"` raises `ValueError` immediately, rather than only surfacing when the
annotation is later used to validate data.

## Applying units from a signature: `declare_units`

`@declare_units` reads the declared unit off the signature, so it lives in the
annotation and nowhere else:

```python
from typing import Annotated
import xarray as xr
from xarray_signature_units import declare_units

@declare_units
def normalise_pressure(
    p: Annotated[xr.DataArray, "Pa"],
) -> Annotated[xr.DataArray, "Pa"]:
    return p
```

On each call, under the active [policy](#the-validation-policy), the wrapper validates
and converts every declared `DataArray` **input** via `check_units`, runs the
function, then stamps each declared `DataArray` **output** with its unit. A
`TypedDict` or `dataclass` return is stamped per-field; a bare
`Annotated[DataArray, unit]` return takes
that unit. Non-`DataArray` arguments and returns pass through untouched.

Every declared unit is checked against the registry **at decoration time**, so a typo
fails fast at import — regardless of policy — rather than only when the function first
runs.

Override the policy per function with keyword arguments (each defaults to the active
policy when omitted):

```python
@declare_units(on_missing="error", on_inexact="error")
def strict_node(x: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray: ...
```

When the policy is disabled (`enabled=False`, e.g. inside `with policy(enabled=False):`)
the wrapper is a total no-op: inputs are not converted and outputs are not stamped.

`declare_units` is intentionally a thin convenience built from the public primitives.
If you need behaviour it doesn't cover — a build-time/static check, a custom value
type — assemble your own consumer from
[`units_from_signature`](#reading-declarations-off-a-signature-units_from_signature),
[`check_units`](#validating-directly-check_units), and `assert_valid_unit` rather than
subclassing anything.

## The validation policy

Both `@declare_units` and `check_units` behave according to a **policy** with one knob
per event. Each is independent, and dimensional mismatches are never negotiable.

### `on_missing` — no parseable unit to check against

Governs only the "can't validate" cases: a missing or unparseable `units` attribute.

| `on_missing` | missing/unparseable units                     | dimensional mismatch |
|--------------|-----------------------------------------------|----------------------|
| `error`      | raises `ValueError`                           | always raises        |
| `warn`       | emits `UnitsWarning`, returns input unchanged | always raises        |
| `ignore`     | silently returns input unchanged              | always raises        |

### `on_inexact` — value-changing conversion

By default (`convert`), a dimensionally compatible unit is silently converted
(`"hPa"` → `"Pa"`). `on_inexact` controls what happens when the actual unit is
compatible with the declared one but *not identical* — any conversion that would
change the values, including affine ones like `"K"` → `"degC"`:

| `on_inexact` | value-changing conversion                 |
|--------------|-------------------------------------------|
| `convert`    | performs the conversion silently          |
| `warn`       | converts, but emits `UnitsWarning`        |
| `error`      | raises `ValueError` instead of converting |

Equivalent spellings of the same unit (`"pascal"` for `"Pa"`) imply no value change
and always convert. `error` is useful when implicit conversion would hide a likely
mistake upstream, and you'd rather the caller fix the unit at the source.

### `enabled` — the master switch

`enabled=False` turns off all validation: `check_units` returns its input unchanged,
and a `@declare_units` wrapper becomes a total no-op (no conversion, no stamping).

### Setting the policy

Each axis resolves once per call from (in order): its environment variable
(`XARRAY_SIGNATURE_UNITS_ON_MISSING` / `_ON_INEXACT` / `_ENABLED`), a process-wide
override set with `set_policy(...)`, or the default (`on_missing="warn"`,
`on_inexact="convert"`, `enabled=True`). Use `policy(...)` as a context manager to
scope overrides — all axes in one call — across every call inside the block:

```python
from xarray_signature_units import policy, check_units

with policy(on_missing="error", on_inexact="warn"):
    check_units(da, "Pa", "vpd")
```

This lets the same call site run strict in CI and lenient while exploring
interactively, without threading policy arguments through every function.

## Validating directly: `check_units`

`@declare_units` is the recommended entry point, but the `check_units` primitive it
calls is public too — reach for it when you want to validate an array by hand rather
than decorate a function:

```python
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

`on_missing` and `on_inexact` may be passed per call; each defaults to the active
policy when `None`. `name` names the array for error/warning messages.

## Choosing a registry: pint vs. CF/UDUNITS

Out of the box you get **plain pint** (`pint.get_application_registry()`), so
standard pint unit strings ("Pa", "degC", "m/s") parse with no setup.

CF-convention strings such as `"umol m-2 s-1"` or `"g m-2 d-1"` need cf-xarray's
UDUNITS-aware registry instead. Install the `[cf]` extra and activate it **once, at
startup**:

```python
from xarray_signature_units import use_cf_units

use_cf_units()   # now "umol m-2 s-1", "g m-2 d-1" parse
```

Or supply any registry yourself:

```python
import pint
from xarray_signature_units import set_registry

set_registry(pint.UnitRegistry())
```

pint has a single process-global application registry, so this is a one-time,
startup choice — not a per-array setting. Quantities created under two different
registries cannot be mixed (pint raises). Choose pint units *or* CF units for your
entire codebase, not a mixture.

## Reading declarations off a signature: `units_from_signature`

For tools that want to inspect a function's declared units statically (e.g. to
generate documentation, or to wire up validation automatically), `units_from_signature`
extracts them without needing to call the function:

```python
from typing import Annotated, TypedDict
import xarray as xr
from xarray_signature_units import units_from_signature

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

Dataclass return types work identically — `units_from_signature` reads per-field
annotations the same way:

```python
from dataclasses import dataclass

@dataclass
class Output:
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
unit-annotated `DataArray` contribute; a plain `xr.DataArray` hint with no unit
is ignored. A bare `Annotated[DataArray, unit]` return annotation yields a single
unit string rather than a dict.
