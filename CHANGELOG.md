# Changelog

## 0.7.0

This release is a drastic simplification, and breaks most of the API.
A declaration now either holds or raises, and how a function validates can be read from its signature.

### Migrating from 0.6.0

| 0.6.0 | 0.7.0 |
|---|---|
| `@units.declare_units`, `@schema.declare_schema`, `@temporal.declare_freq` | One decorator, `@xra.enforce`, with no options |
| `from xarray_annotated.units import Unit`, and likewise for the other markers | Also available from the root: `from xarray_annotated import Coords, Dims, Dtype, Freq, Unit` |
| `units_from_signature`, `schema_from_signature`, `freq_from_signature`, `annotated_unit`, `annotated_schema`, `annotated_freq` | `xra.declarations_from_signature`, `xra.declaration_from_hint` |
| `Annotated[xr.DataArray, "Pa"]` declaring a unit | `Annotated[xr.DataArray, Unit("Pa")]`; a string is only a description |
| `xra.unwrap_annotated`, `xra.walk_signature` | Removed; read declarations with the readers above |
| `check_freq(da, [Freq("D")], name)` | `check_freq(da, Freq("D"), name)`; a hint holds one `Freq` |
| `set_policy`, `policy`, `get_policy`, `Policy`, and the `on_missing`, `on_inexact`, `on_output`, `on_mismatch` and `on_uninferable` settings and environment variables | Removed; every failed check raises |
| `units.policy(enabled=False)` and similar | `with xra.enabled(False):`, or `XARRAY_ANNOTATED_ENABLED` |
| `UnitsWarning`, `SchemaWarning`, `FreqWarning` | Removed; nothing warns |
| A compatible input unit is converted | It raises `ValueError`; call `xra.units.convert(da, "Pa")` first |
| A returned array's label is overwritten with the declared unit | It must already match; a body that changes the unit states it with `.assign_attrs(units=...)` |
| A quantified output is converted to the declared unit | It must already be in that unit; convert it in the body with `.pint.to(...)` |
| `Dtype("float64")` passes any float; `Dtype("float64", exact=True)` passes only `float64` | `Dtype("float")` passes any float; `Dtype("float64")` passes only `float64` |
| `Freq("W", anchored=True)`, `Freq("W-SUN", anchored=False)` | `Freq("W-SUN")`, `Freq("W")` |

### Newly raises

- **`TypeError`, on a call:** a declared value that is not a `DataArray`, such as `None`, an ndarray or a `Dataset`, unless its hint allows it with `| None` or another type. So does a return that is not the declared `TypedDict` or dataclass, or lacks a required `TypedDict` field.
- **`FreqError`, on a call:** an unindexed time axis, a custom index on it, a scalar time, or an irregular or two-point axis that contradicts the declaration. An axis `Freq` could not infer used to pass with a warning.
- **`ValueError`, at decoration:** a non-string `Unit` or `Freq` argument; a marker `enforce` cannot apply, such as inside a `tuple`, on a `Dataset` or in a generic alias; a return or parameter holding a class whose fields carry markers, other than one `TypedDict` or dataclass return; a declared `*args` or `**kwargs`; a second marker of one kind on a hint; and a `type` alias that refers to itself.

### Fixed

- Converting no longer writes corrupt data: a converted or quantified array loses its `.encoding`, as packing that described other values would overflow on write.
- Checking a unit no longer reads a lazily opened array from disk.
- Converting no longer fails on a coordinate labelled `degrees_north`.
- A returned dict or dataclass is no longer changed in place.
- An async function's output is checked on the awaited result.
