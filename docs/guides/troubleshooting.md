# Troubleshooting

## Errors

| Symptom | Cause | Fix |
|---|---|---|
| `NameError` at decoration | Annotations are evaluated at decoration, so a `TYPE_CHECKING`-only import fails, as does a function-local alias under `from __future__ import annotations` | Import names used in annotations at run time, on every Python version |
| `ValueError: ... is not a recognised unit`, at import | A typo, or a CF string on the default registry | Fix the typo, or [activate the CF registry](registry.md) before importing the module |
| `ValueError: ... has no 'units' attribute` | The array has no label to check | Attach the unit where the data enters, or use [`xra.enabled(False)`](#enabled) while prototyping; no setting lets it through, because an unchecked unit looks like a checked one downstream |
| `ValueError: ... which enforce cannot apply`, at import | A marker somewhere it cannot be checked: inside a `tuple`, `list`, `dict` or `Callable`, on a base type other than `DataArray`, or in a generic alias such as `type Generic[T] = Annotated[T, Unit("Pa")]` | Move it onto an `Annotated[xr.DataArray, ...]`, and return several arrays as a `TypedDict` or dataclass |
| `ValueError: ... holds fields carrying ...`, at import | A parameter, a return such as a `NamedTuple`, `list[R]` or `R1 \| R2`, or a field of a return, holds a class whose fields carry markers, which `enforce` cannot check | Return one `TypedDict` or dataclass, optionally `\| None`, with its arrays as direct fields, and declare a parameter's arrays as parameters of their own |
| `ValueError: type alias ... refers to itself`, at import | A `type` alias contains itself outside a container, such as `type R = R \| None` | Remove the self-reference |
| `ValueError: input ... differs from declared` | The unit is compatible with the declared one but not equal to it | [Convert it](units.md#converting-an-input) where you call the function |
| `ValueError: output ... differs from declared`, or an output with no `units` | The body returned another unit, left a stale label, or returned no label | [Compute with pint](pint.md), or for bare numbers [state the unit it returns](pint.md#plain-arrays) |
| `SchemaError`, or `FreqError: ... frequency mismatch` | The array does not match its `Dims`, `Coords`, `Dtype` or `Freq`, or its time axis is irregular | Fix the data, or correct the declaration. An axis with holes may want [`gaps_ok=True`](declaring.md#axes-with-gaps), and one on no month boundary [`Freq("month")`](declaring.md#calendar-steps-on-any-day) |
| `FreqError: ... got no step that short` | `gaps_ok=True` was declared, and no two timestamps are as close as the declared step, so the data may be coarser than declared | Declare the coarser frequency, or fix the data |
| `TypeError: ... declared a DataArray, got ...`, when called | The value is `None`, an ndarray, a `Dataset` or something else, as after `.values` in a body | Pass or return a `DataArray`, or add what the value may be to the hint, such as `\| None` |
| `DimensionalityError` | The units cannot be converted | Fix the data or the declaration |
| `FreqError: ... no indexed datetime coordinate`, when called | The message names a datetime coordinate and its fix where it can tell. Otherwise there is no time coordinate, as after `.mean("time")`, or it is unindexed `cftime` or undecoded | Remove the `Freq` marker, [index it][xarray_annotated.Freq], or decode it |
| Checks stopped everywhere | The package is switched off | Check `XARRAY_ANNOTATED_ENABLED`, and look for an `xra.enabled(False)` block |

`SchemaError` and `FreqError` are not `ValueError`s, so catching a mismatch never swallows a malformed declaration.

## A declaration is silently ignored { #a-declaration-is-silently-ignored }

This looks like a declaration but declares nothing, with no error or warning:

```python
Annotated[xr.DataArray, "Pa"]  # a string is a description; use Unit("Pa")
```

In a union such as `Annotated[xr.DataArray, Unit("Pa")] | Annotated[int, Unit("kg")]`, only the `DataArray` member applies.

`declarations_from_signature(f)` shows what a function actually declares.

A string can describe a property that is not checked, as in `Annotated[xr.DataArray, Dims("time", "x"), "usually float64"]`.

## A wrong result passes every check

A body's output label is taken on trust; see [What a label cannot catch](units.md#what-a-label-cannot-catch).
A quantified result can still be the wrong calculation; see [What pint cannot catch](pint.md#what-pint-cannot-catch).

## Switching the package off { #enabled }

A declaration that does not hold for all of a function's inputs should be corrected or removed.
While prototyping, switch the package off:

```python
with xra.enabled(False):
    ...
```

Off is a total no-op, so a function declaring `Unit("Pa")` can receive `hPa`.
Malformed declarations still raise when a function is decorated.

- **For a process:** set `XARRAY_ANNOTATED_ENABLED` to `0`, `false`, `no` or `off`.
- **Precedence:** a block beats the variable in both directions, and the innermost block wins.
- **Scope:** a block applies to the current thread or async task only, and to dask's threaded and synchronous schedulers when it surrounds `.compute()`. It does not reach a new thread or another process.
