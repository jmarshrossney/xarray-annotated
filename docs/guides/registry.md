# Choosing a unit registry

A [pint](https://pint.readthedocs.io/en/stable/) registry decides which unit strings parse.
The choice is process-wide, and quantities from different registries cannot be mixed, so use one registry for a whole codebase.
A library that calls `use_cf_units()` makes that choice for you.

## pint units (the default)

`pint.get_application_registry()` needs no setup, and parses strings such as `"Pa"`, `"degC"` and `"m/s"`.

## CF-convention units

UDUNITS spellings use space-separated negative exponents (`"umol m-2 s-1"`, `"g m-2 d-1"`), which plain pint rejects.
Install the [`cf` extra](../getting_started.md#installation) and activate its registry:

```python
from xarray_annotated.units import use_cf_units

use_cf_units()
```

Call [`use_cf_units`][xarray_annotated.units.use_cf_units] near the top of your entry point or package `__init__`, before importing any module that declares such units, since declarations are validated when functions are decorated.

## Your own registry

```python
import pint
from xarray_annotated.units import set_registry

ureg = pint.UnitRegistry()
ureg.define("dobson_unit = 2.687e20 * meter**-2 = DU")
set_registry(ureg)
```

[`set_registry`][xarray_annotated.units.set_registry] keeps pint and the `.pint` accessor in sync, and [`get_registry`][xarray_annotated.units.get_registry] returns the active registry.
