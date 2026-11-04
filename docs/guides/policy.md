# Configuring validation

A value either matches its declaration or raises, so an array that got through was checked.
The one setting on a declaration is `convert`, and `xra.enabled` [switches the package off](#enabled) while prototyping.

| Event | Result | Setting |
|---|---|---|
| Structure mismatch | `SchemaError` | none |
| Frequency mismatch, irregular time axis, or no unambiguous time axis | `FreqError` | none |
| Compatible but different input unit | `ValueError` | [`convert=True`](conversion.md#opting-in) on the marker |
| Dimensionally wrong input unit | `DimensionalityError` | none |
| Missing, unparseable or over-long input unit | `ValueError` | none |
| [Output unit that does not match](#units-outputs) | Depends on the unit | none |

`SchemaError` and `FreqError` are not `ValueError`s, so catching a mismatch never swallows a malformed declaration.

A declaration that does not hold for all of a function's inputs should be corrected or removed.
A string in `Annotated` [declares nothing](declaring.md), so it can describe a property that is not checked:

```python
Annotated[xr.DataArray, Dims("time", "x"), "usually float64"]
```

## Output units { #units-outputs }

A function's author controls what it returns, so an output that does not match its declaration is a bug.
Nothing is converted or labelled on the author's behalf.

| Returned value | Result |
|---|---|
| Label or quantified unit equal to declared | Passes; a label is respelled as declared (`pascal` to `Pa`) |
| Different but compatible unit (`hPa`) | `ValueError` |
| Incompatible unit (`kg`) | `DimensionalityError` |
| No unit, or an unparseable or over-long one | `ValueError` |
| Not a `DataArray` | Passed through |

`convert` does not apply to outputs, so one marker can be shared with an input.
A quantified result stays quantified, and a body converts one itself with `.pint.to(...)`.

A label that differs from the declaration could mean two things, and the package cannot tell them apart:

- **A forgotten conversion:** the label is true, and the values are in the wrong unit.
- **A stale label:** `attrs` are inert under arithmetic, so a correct body can leave its input's label on the result.

Both raise, because a false alarm is loud and a mislabelled result is silent.
A body that changes the unit states the new one:

```python
@xra.enforce
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return (p * 100.0).assign_attrs(units="Pa")  # otherwise labelled "hPa"
```

Values are never inspected, so `assign_attrs` can silence a wrong body as easily as a right one: it makes the author state the unit but does not verify it.
A better approach is an open question in [this GitHub issue](https://github.com/jmarshrossney/xarray-annotated/issues/26).

## Wrapping a library that returns bare arrays { #wrapping-a-library }

A wrapper's output declaration is a claim about someone else's convention, and a bare array raises until the wrapper makes that claim.
Attach the unit the library documents, and let pint do any conversion:

```python
@xra.enforce
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
    raw = some_library.gpp(ppfd.values)  # documented as ug m-2 s-1
    out = xr.DataArray(raw, dims=ppfd.dims, coords=ppfd.coords)
    return out.assign_attrs(units="ug m-2 s-1").pint.quantify().pint.to("g m-2 d-1")
```

Here that is a factor of 0.0864.
`.pint.quantify()` also [reads the labels on coordinates](conversion.md#quantify-reads-coordinates).

## Switching the package off { #enabled }

```python
with xra.enabled(False):
    ...
```

Off is a total no-op: no checks and no conversion, so a function declaring `Unit("Pa")` can receive `hPa`.
Malformed declarations still raise when a function is decorated.

- **For a process:** set `XARRAY_ANNOTATED_ENABLED` to `0`, `false`, `no` or `off`.
- **Precedence:** a block beats the variable in both directions, and the innermost block wins.
- **Scope:** a block applies to the current thread or async task only.

## Logging { #auditing }

Each conversion under `convert=True` is logged at `DEBUG` to `xarray_annotated.units`, on every call.
Nothing is logged at `WARNING` or above, so this is silent by default.

```python
import logging

logging.basicConfig()
logging.getLogger("xarray_annotated").setLevel(logging.DEBUG)
```
