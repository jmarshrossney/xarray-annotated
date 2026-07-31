# Choosing a unit registry

Every unit string you declare has to be *parsed* by something. That something is a
[pint](https://pint.readthedocs.io/en/stable/) registry, and which one you use is a
one-time, process-wide decision.

## pint units (the default)

Out of the box you get `pint.get_application_registry()`, so standard pint unit strings
parse with no setup at all:

```python
Annotated[xr.DataArray, Unit("Pa")]
Annotated[xr.DataArray, Unit("degC")]
Annotated[xr.DataArray, Unit("m/s")]
```

If those cover your vocabulary, there is nothing on this page you need to do.

## CF-convention units

Earth-science data conventionally spells units the UDUNITS way — `"umol m-2 s-1"`,
`"g m-2 d-1"`, `"kg m-2"` — with negative exponents rather than a solidus, and with
symbols like `umol` that plain pint doesn't recognise. Declaring one of these against the
default registry fails at import:

```
ValueError: node input 'gpp': declared unit 'g m-2 d-1' is not a recognised unit
  (...) (call use_cf_units() to enable CF/UDUNITS units like 'g m-2 d-1')
```

Install the [`[cf]` extra](../getting_started.md#installation), which pulls in
[cf-xarray](https://cf-xarray.readthedocs.io), and activate its registry **once, at
startup**:

```python
from xarray_annotated.units import use_cf_units

use_cf_units()   # now "umol m-2 s-1", "g m-2 d-1" parse
```

"At startup" is meant literally: call it before importing the modules whose signatures
declare those units, because declarations are validated at decoration time. In practice
that means near the top of your entry point or package `__init__`, not inside the function
that needs it.

## Supplying your own

`set_registry` accepts any pint registry — a customised one, or a
[`pint.UnitRegistry`](https://pint.readthedocs.io/en/stable/api/base.html) with your own
definitions loaded:

```python
import pint
from xarray_annotated.units import set_registry

ureg = pint.UnitRegistry()
ureg.define("dobson_unit = 2.687e20 * meter**-2 = DU")
set_registry(ureg)
```

`set_registry` also calls `pint_xarray.setup_registry`, so the `.pint` accessor and this
package's parsing helpers can't drift apart. `get_registry()` returns whichever registry
is currently active.

## Registries are process-wide

pint has a single global application registry, and quantities created under two different
registries cannot be combined — pint raises rather than silently guessing. So this is not
a per-array or per-function setting, and there is deliberately no way to make it one.

**Choose pint units or CF units for an entire codebase, not a mixture.** If you are
consuming a library that has already called `use_cf_units()`, you are on CF units too.

This is the one piece of global state in the package that isn't a
[policy](policy.md) — policies decide what happens when a check fails, whereas the
registry decides what a unit string *means*.
