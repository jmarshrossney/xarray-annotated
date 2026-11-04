# Troubleshooting

## Errors

| Symptom | Cause | Fix |
|---|---|---|
| `NameError` at decoration | Annotations are evaluated at decoration, so a `TYPE_CHECKING`-only import fails, as does a function-local alias under `from __future__ import annotations` | Import names used in annotations at run time, on every Python version |
| `ValueError: ... is not a recognised unit`, at import | A typo, or a CF string on the default registry | Fix the typo, or [activate the CF registry](registry.md) before importing the module |
| `ValueError: ... has no 'units' attribute` | The array has no label to check | Attach the unit where the data enters, or use [`xra.enabled(False)`](policy.md#enabled) while prototyping; no setting lets it through, because an unchecked unit looks like a checked one downstream |
| `ValueError: ... which enforce cannot apply`, at import | A marker somewhere it cannot be checked: inside a `tuple`, `list`, `dict` or `Callable`, on a base type other than `DataArray`, or in a generic alias such as `type Generic[T] = Annotated[T, Unit("Pa")]` | Move it onto an `Annotated[xr.DataArray, ...]`, and return several arrays as a `TypedDict` or dataclass |
| `ValueError: ... holds fields carrying ...`, at import | A parameter, a return such as a `NamedTuple`, `list[R]` or `R1 \| R2`, or a field of a return, holds a class whose fields carry markers, which `enforce` cannot check | Return one `TypedDict` or dataclass, optionally `\| None`, with its arrays as direct fields, and declare a parameter's arrays as parameters of their own |
| `ValueError: type alias ... refers to itself`, at import | A `type` alias contains itself outside a container, such as `type R = R \| None` | Remove the self-reference |
| `ValueError: input ... differs from declared` | The unit is compatible with the declared one but not equal to it | [Convert it](conversion.md) where you call the function |
| `ValueError: output ... differs from declared`, or an output with no `units` | The body returned another unit, left a stale label, or returned no label | Fix the body, or [state the unit it returns](policy.md#units-outputs) |
| `TypeError: ... declared a DataArray, got ...`, when called | The value is `None`, an ndarray, a `Dataset` or something else, as after `.values` in a body | Pass or return a `DataArray`, or add what the value may be to the hint, such as `\| None` |
| `DimensionalityError` | The units cannot be converted | Fix the data or the declaration |
| `FreqError: ... no indexed datetime coordinate`, when called | The message names a datetime coordinate and its fix where it can tell. Otherwise there is no time coordinate, as after `.mean("time")`, or it is unindexed `cftime` or undecoded | Remove the `Freq` marker, [index it](declaring.md#which-coordinate-is-the-time-axis), or decode it |
| Checks stopped everywhere | The package is switched off | Check `XARRAY_ANNOTATED_ENABLED`, and look for an `xra.enabled(False)` block |

A quantified input is read from `da.pint.units`, follows the same rules as a plain one, and reaches the body dequantified with `attrs["units"]` set.
For outputs, see [Output units](policy.md#units-outputs).

## A declaration is silently ignored { #a-declaration-is-silently-ignored }

This looks like a declaration but declares nothing, with no error or warning:

```python
Annotated[xr.DataArray, "Pa"]  # a string is a description; use Unit("Pa")
```

In a union such as `Annotated[xr.DataArray, Unit("Pa")] | Annotated[int, Unit("kg")]`, only the `DataArray` member applies.

`declarations_from_signature(f)` shows what a function actually declares.

## A forgotten conversion can still get through { #a-forgotten-conversion-is-not-caught }

A body that leaves its input's label on the result raises, but values are never inspected, so a wrong result with the right label passes every check:

```python
@xra.enforce
def daily_carbon(
    flux: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
    daily = flux.resample(time="D").mean()  # forgot * MOLAR_FACTOR
    return daily.assign_attrs(units="g m-2 d-1")
```

The same goes for a unit attached to a wrapped library's bare array, which is taken on trust.
A consumer declaring `g m-2 d-1` then agrees with that label, so declaring the unit downstream does not help.
Declarations catch wiring mistakes, not algebra.

Two things do keep the unit tied to the values:

- **Let pint carry the unit.** Compute on a quantified array, or [attach the unit a library documents](policy.md#wrapping-a-library), and the return is checked against the declaration. Here the forgotten factor raises `DimensionalityError`:

    <!-- docs-test: skip -->
    ```python
    return flux.pint.quantify().resample(time="D").mean()
    ```

- **Leave `Unit` off the return of a stage you don't trust.** The array's existing label then reaches the next declared input, which checks it. The [worked example](../example.md) catches its missing stage this way.
