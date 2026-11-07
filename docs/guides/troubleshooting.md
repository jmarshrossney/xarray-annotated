# Troubleshooting

## Errors

| Symptom | Cause | Fix |
|---|---|---|
| `NameError` at decoration | Annotations are evaluated at decoration, so a `TYPE_CHECKING`-only import fails, as does a function-local alias under `from __future__ import annotations` | Import names used in annotations at run time, on every Python version |
| `ValueError: ... is not a recognised unit`, at import | A typo, or a CF string on the default registry | Fix the typo, or [activate the CF registry](units.md#cf-convention-units) before importing the module |
| `UnitError: ... has no 'units' attribute` | The array has no label to check | Attach the unit where the data enters, or use [`xra.enabled(False)`](first_principles.md#enabled) while prototyping; no setting lets it through, because an unchecked unit looks like a checked one downstream |
| `ValueError: ... which check_annotations cannot apply`, at import | A marker somewhere it cannot be checked: inside a `tuple`, `list`, `dict` or `Callable`, on a base type other than `DataArray`, on a union member other than the `DataArray`, or in a generic alias such as `type Generic[T] = Annotated[T, Unit("Pa")]` | Move it onto an `Annotated[xr.DataArray, ...]`, and return several arrays as a `TypedDict` or dataclass |
| `ValueError: ... reads as declaring its markers for ... too`, at import | Markers sit on a union with another type, as in `Annotated[xr.DataArray | None, Unit("Pa")]` | Put the union outside the `Annotated`: `Annotated[xr.DataArray, Unit("Pa")] | None` |
| `ValueError: ... holds fields carrying ...`, at import | A parameter, a return such as a `NamedTuple`, `list[R]` or `R1 | R2`, or a field of a return, holds a class whose fields carry markers, which `xra.check_annotations` cannot check | Return one `TypedDict` or dataclass, optionally `| None`, with its arrays as direct fields, and declare a parameter's arrays as parameters of their own |
| `ValueError: type alias ... refers to itself`, at import | A `type` alias contains itself outside a container, such as `type R = R | None` | Remove the self-reference |
| `UnitError: input ... differs from declared` | The unit is compatible with the declared one but not equal to it | [Convert it](units.md#converting-an-input) where you call the function |
| `UnitError: output ... differs from declared`, or an output with no `units` | The body returned another unit, left a stale label, returned no label, or has the wrong calculation | Work out which unit the values are in before changing anything, as a new label does not correct them. Then [assign the unit on output](units.md#outputs) |
| `SchemaError`, or `FreqError: ... frequency mismatch` | The array does not match its `Dims`, `Coords`, `Dtype` or `Freq`, or its time axis is irregular | Fix the data, or correct the declaration. An axis with holes may want [`gaps_ok=True`](temporal.md#axes-with-gaps), and one on no month boundary [`Freq("month")`](temporal.md#calendar-steps-on-any-day) |
| `FreqError: ... got no step that short` | `gaps_ok=True` was declared, and no two timestamps are as close as the declared step, so the data may be coarser than declared | Declare the coarser frequency, or fix the data |
| `ContractError: ... declared a DataArray, got ...`, when called | The value is `None`, an ndarray, a `Dataset` or something else, as after `.values` in a body | Pass or return a `DataArray`, or add what the value may be to the hint, such as `| None` |
| `UnitError: ... cannot convert to declared` | The units cannot be converted | Fix the data or the declaration |
| `DimensionalityError`, from `xra.units.convert` or in a body | The units cannot be converted, or the arithmetic makes no sense | Fix the data or the arithmetic |
| `FreqError: ... no indexed datetime coordinate`, when called | The message names a datetime coordinate and its fix where it can tell. Otherwise there is no time coordinate, as after `.mean("time")`, or it is unindexed `cftime` or undecoded | Remove the `Freq` marker, [index it][xarray_annotated.Freq], or decode it |
| Checks stopped everywhere | The package is switched off | Check `XARRAY_ANNOTATED_ENABLED`, and look for an [`xra.enabled(False)`](first_principles.md#enabled) block |

Every failed check is a `ContractError`, and not a `ValueError`, so catching a mismatch never swallows a malformed declaration.

## A declaration is silently ignored { #a-declaration-is-silently-ignored }

This looks like a declaration but declares nothing, with no error or warning:

```python
Annotated[xr.DataArray, "Pa"]  # a string is a description; use Unit("Pa")
```

`declarations_from_signature(f)` shows what a function actually declares.

A string can describe a property that is not checked, as in `Annotated[xr.DataArray, Dims("time", "x"), "usually float64"]`.

## A wrong result passes every check

A body's output label is taken on trust; see [What a label cannot catch](units.md#what-a-label-cannot-catch).
A quantified result can still be the wrong calculation; see [What pint cannot catch](../tools/pint.md#what-pint-cannot-catch).
