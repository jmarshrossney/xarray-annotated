# Configuring validation

Behaviour is set only where it is declared, so reading a signature and its decorator tells you how a function validates.

| Event | Setting | Values (default first) | On the marker | `enforce` keyword | `check_*` argument |
|---|---|---|---|---|---|
| Structure mismatch | `on_mismatch` | `error`, `warn`, `ignore` | yes | no | `check_schema` |
| Frequency mismatch, or no unambiguous time axis | `on_mismatch` | `error`, `warn`, `ignore` | yes | no | `check_freq` |
| Short or irregular time axis | `on_uninferable` | `warn`, `error`, `ignore` | no | `temporal_on_uninferable` | `check_freq` |
| Compatible but different input unit | [`convert`](conversion.md#opting-in) | `False`, `True` | yes | no | no (pass the marker) |
| Dimensionally wrong input unit | none | always raises `DimensionalityError` | | | |
| Missing, unparseable or over-long input unit | none | always raises `ValueError` | | | |
| [Output unit that does not match](#units-outputs) | none | always raises | | | |

The marker beats the keyword or argument, which beats the default.
A mismatch can only be softened, and a conversion only allowed, on the marker it applies to.

```python
Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64", on_mismatch="warn")]
```

What each severity does:

- **`error`** raises `SchemaError` or `FreqError`. Neither is a `ValueError`, so catching a mismatch never swallows a malformed declaration.
- **`warn`** emits `SchemaWarning`, `FreqWarning` or, for a short or irregular axis, its subclass `UninferableFreqWarning`.
- **`ignore`** passes the array through and [logs](#auditing) at `INFO`.

To silence short test fixtures while leaving real mismatches visible:

```toml
[tool.pytest.ini_options]
filterwarnings = ["ignore::xarray_annotated.temporal.UninferableFreqWarning"]
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

Nothing is logged at `WARNING` or above, so this is silent by default.
The loggers are `xarray_annotated.units`, `.schema` and `.temporal`.

```python
import logging

logging.basicConfig()
logging.getLogger("xarray_annotated").setLevel(logging.DEBUG)
```

| Event | Level |
|---|---|
| Schema or frequency event passed under `ignore` | `INFO` |
| Input converted under `convert=True` | `DEBUG` |

Records are emitted on every call, not once per call site like warnings.
