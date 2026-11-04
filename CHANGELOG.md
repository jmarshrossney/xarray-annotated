# Changelog

## 0.7.0

This release is a drastic simplification, and breaks most of the API.
A declaration now either holds or raises, and how a function validates can be read from its signature.

### Migrating from 0.6.0

**Search for string metadata before upgrading.** `Annotated[xr.DataArray, "Pa"]` declared a unit in 0.6.0 and is now only a description, so an unmigrated hint raises nothing and goes unchecked.
Nearly everything else below fails loudly.

| 0.6.0 | 0.7.0 |
|---|---|
| `Annotated[xr.DataArray, "Pa"]` | `Annotated[xr.DataArray, Unit("Pa")]` |
| `@units.declare_units`, `@schema.declare_schema`, `@temporal.declare_freq` | One decorator, `@xra.enforce`, with no options |
| `from xarray_annotated.units import Unit`, and likewise for the other markers | Also available from the root: `from xarray_annotated import Coords, Dims, Dtype, Freq, Unit` |
| `units_from_signature`, `schema_from_signature`, `freq_from_signature`, `annotated_unit`, `annotated_schema`, `annotated_freq` | `xra.declarations_from_signature`, `xra.declaration_from_hint`, which return `Declared`, so a unit string is `declared.unit.unit` |
| `xra.unwrap_annotated` | `typing.get_args(hint)[0]` |
| `xra.walk_signature` | Removed, with no public replacement |
| `set_policy`, `policy`, `get_policy`, `Policy`, and the `OnMissing`, `OnInexact`, `OnOutput`, `OnMismatch` and `OnUninferable` types | Removed; every failed check raises. To tolerate something, correct or remove the declaration |
| The `on_missing`, `on_inexact`, `on_output`, `on_mismatch` and `on_uninferable` settings, on decorators, markers and checkers | Removed, likewise |
| `XARRAY_ANNOTATED_UNITS_ON_MISSING`, `_UNITS_ON_INEXACT`, `_UNITS_ON_OUTPUT`, `_SCHEMA_ON_MISMATCH`, `_TEMPORAL_ON_MISMATCH`, `_TEMPORAL_ON_UNINFERABLE` | Removed, likewise |
| `units.policy(enabled=False)`, `set_policy(enabled=False)` | `with xra.enabled(False):`, or `XARRAY_ANNOTATED_ENABLED`. The block now beats the variable, and covers only the current thread or async task, so use the variable to switch off worker threads |
| `UnitsWarning`, `SchemaWarning`, `FreqWarning` | Removed; nothing warns |
| A compatible input unit is converted, optionally with a warning | It raises `ValueError`; call `xra.units.convert(da, "Pa")` first, which logs at `DEBUG` on the `xarray_annotated.units` logger |
| A returned array is stamped with the declared unit, whatever label it had, or none | It must already match: a missing or unparseable label or a compatible unit raises `ValueError`, and an incompatible one `DimensionalityError`. A body that changes the unit states it with `.assign_attrs(units=...)` |
| A quantified output is converted to the declared unit | It must already be in that unit; convert it in the body with `.pint.to(...)` |
| `Dtype("float64")` passes any float, `Dtype("int32")` any signed integer, and so on; `exact=True`, `Dtype.exact` | A name with a width is exact, so `dtype_compatible` is stricter too. `float`, `int`, `uint`, `complex`, `str`, `bytes`, `datetime64` and `timedelta64` mean the whole kind. Other kind-like spellings, such as `Dtype("U")` or `Dtype("M8")`, no longer match any width |
| `Freq("W", anchored=True)`, `Freq("W-SUN", anchored=False)`, `Freq.anchored` | `Freq("W-SUN")`, `Freq("W")` |
| `check_freq(da, [Freq("D")], name)` | `check_freq(da, Freq("D"), name)`; a hint holds one `Freq` |
| `check_units`, `check_schema`, `check_freq`, `apply_output_units` with a severity argument | `qualname` is now the fourth positional argument, so a severity passed by position is silently read as `qualname`. Remove it |
| `apply_output_units` changes the array it is given | It returns a copy, so always use the return value |

### Newly raises

- **`TypeError`, on a call:** a declared value that is not a `DataArray`, such as `None`, an ndarray or a `Dataset`, unless its hint allows it. `| None` allows `None`, and any other member, such as `| float`, allows anything else, unchecked. A return that is not the declared `TypedDict` or dataclass raises too, as does one lacking a required `TypedDict` field.
- **`ValueError`, on a call:** an input with a missing, unparseable or over-long unit, which used to pass with a warning. Attach it with `.assign_attrs(units=...)`.
- **`FreqError`, on a call:** an unindexed or 2-D time coordinate, a custom index on it, a scalar time, or an irregular or two-point axis that contradicts the declaration. An axis `Freq` could not infer used to pass with a warning.
- **`ValueError`, at decoration:** `Unit(None)`, `Freq(5)` or `Freq(None)`; a marker `enforce` cannot apply, such as inside a `tuple`, on a `Dataset` or in a generic alias; a return or parameter holding a class whose fields carry markers, other than one `TypedDict` or dataclass return; a declared `*args` or `**kwargs`; a second marker of one kind on a hint, which used to be first-wins; a union of two declared members; and a `type` alias that refers to itself.

### Newly checked

These were silently ignored, and now apply: markers on a union member (`Annotated[xr.DataArray, Unit("Pa")] | None`), in a `type` alias and in `NotRequired[...]`, and on the fields of a `TypedDict` or dataclass return under `| None`.

### Newly passes

- A scalar or auxiliary datetime coordinate beside the real time axis, which used to be ambiguous.
- An axis of zero or one points, with no warning.
- `Dtype("float64")` on big-endian data, as byte order is not compared.
- `freq_compatible(Freq("QE-MAR"), Freq("QE-JUN"))`, as they are the same four dates.

### Fixed

- Converting no longer writes corrupt data: a converted or quantified array loses its `.encoding`, as packing that described other values would overflow on write.
- Checking a unit no longer reads a lazily opened array from disk.
- Converting no longer fails on a coordinate labelled `degrees_north` under the default registry.
- A returned array, dict or dataclass is no longer changed in place.
- An async function's checks run when it is awaited, and its output is checked on the awaited result.
