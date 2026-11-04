# Troubleshooting

## Errors

| Symptom | Cause | Fix |
|---|---|---|
| `NameError` at decoration | Annotations are evaluated at decoration, so a `TYPE_CHECKING`-only import fails, as does a function-local alias under `from __future__ import annotations` | Import names used in annotations at run time, on every Python version |
| `ValueError: ... is not a recognised unit`, at import | A typo, or a CF string on the default registry | Fix the typo, or [activate the CF registry](registry.md) before importing the module |
| `ValueError: ... has no 'units' attribute` | The array has no label to check | Attach the unit where the data enters, or use [`xra.enabled(False)`](policy.md#enabled) while prototyping; no setting lets it through, because an unchecked unit looks like a checked one downstream |
| `ValueError: ... which enforce cannot apply`, at import | A marker somewhere it cannot be checked: inside a `tuple`, `list`, `dict` or `Callable`, on a base type other than `DataArray`, or in a generic alias such as `type Generic[T] = Annotated[T, Unit("Pa")]` | Move it onto an `Annotated[xr.DataArray, ...]`, and return several arrays as a `TypedDict` or dataclass |
| `DimensionalityError` | The units cannot be converted, and `convert=True` does not change that | Fix the data or the declaration |
| Checks stopped everywhere | The package is switched off | Check `XARRAY_ANNOTATED_ENABLED`, and look for an `xra.enabled(False)` block |

A quantified input is read from `da.pint.units`, follows the same rules as a plain one, and reaches the body dequantified with `attrs["units"]` set.
For quantified outputs, see [`on_output`](policy.md#units-on_output).

## A declaration is silently ignored { #a-declaration-is-silently-ignored }

This looks like a declaration but declares nothing, with no error or warning:

```python
Annotated[xr.DataArray, "Pa"]  # a string is a description; use Unit("Pa")
```

A `TypedDict` used as a parameter type is not read field by field, and in a union such as `Annotated[xr.DataArray, Unit("Pa")] | Annotated[int, Unit("kg")]` only the `DataArray` member applies.

`declarations_from_signature(f)` shows what a function actually declares.

## A forgotten conversion is not caught { #a-forgotten-conversion-is-not-caught }

A function declares an output unit, its body omits the conversion or a wrapped library returns something else, and every check passes:

```python
@xra.enforce
def daily_carbon(
    flux: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
    return flux.resample(time="D").mean()  # forgot * MOLAR_FACTOR
```

The output is stamped with its declared unit, and values are never inspected.
A consumer declaring `g m-2 d-1` then agrees with that stamp, so declaring the unit downstream does not help, and `strict` cannot tell this body from a correct one.
Declarations catch wiring mistakes, not algebra.

Two things do keep the unit tied to the values:

- **Let pint carry the unit.** Compute on a quantified array, or [attach the unit a library documents](policy.md#wrapping-a-library), and the return is converted to the declaration. Here the forgotten factor raises `DimensionalityError`:

    <!-- docs-test: skip -->
    ```python
    return flux.pint.quantify().resample(time="D").mean()
    ```

- **Leave `Unit` off the return of a stage you don't trust.** The array's existing label then reaches the next declared input, which checks it. The [worked example](../example.md) catches its missing stage this way.
