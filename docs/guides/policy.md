# Configuring validation

Behaviour is set only where it is declared, so reading a signature and its decorator tells you how a function validates.

| Event | Setting | Values (default first) | On the marker | `enforce` keyword | `check_*` argument |
|---|---|---|---|---|---|
| Structure mismatch | `on_mismatch` | `error`, `warn`, `ignore` | yes | no | `check_schema` |
| Frequency mismatch, or no unambiguous time axis | `on_mismatch` | `error`, `warn`, `ignore` | yes | no | `check_freq` |
| Short or irregular time axis | `on_uninferable` | `warn`, `error`, `ignore` | no | `temporal_on_uninferable` | `check_freq` |
| Compatible but different unit, on an input or a quantified output | [`convert`](conversion.md#opting-in) | `False`, `True` | yes | no | no (pass the marker) |
| Declared output unit | `on_output` | `stamp`, `strict`, `require` | yes | `units_on_output` | `apply_output_units` |
| Dimensionally wrong input unit | none | always raises `DimensionalityError` | | | |
| Missing, unparseable or over-long input unit | none | always raises `ValueError` | | | |

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

## `on_output` { #units-on_output }

| Returned value | `stamp` | `strict` | `require` |
|---|---|---|---|
| No label | stamped | stamped | `ValueError` |
| Label equals declared | stamped | stamped | stamped |
| Compatible label (`hPa`) | stamped | `ValueError` | `ValueError` |
| Incompatible label (`kg`) | stamped | `DimensionalityError` | `DimensionalityError` |
| Unparseable label | stamped | stamped | `ValueError` |
| Quantified, compatible unit | `ValueError`, or converted with `convert=True` | `ValueError` | `ValueError`, or converted with `convert=True` |
| Quantified, incompatible unit | `DimensionalityError` | `DimensionalityError` | `DimensionalityError` |
| Not a `DataArray` | passed through | passed through | `ValueError` |

Stamping is the default because `attrs` are inert under arithmetic:

```python
@xra.enforce
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p * 100.0  # values are Pa; attrs still say "hPa"
```

Converting on that stale label would multiply by 100 again.
For the same reason `strict` suits only bodies that keep their units truthful, such as pass-through functions or pint-quantified computations.
A body doing scalar arithmetic fails it whether or not it is correct, unless it clears its own label.

A quantified array's unit lives in the data and cannot go stale, so it is never stamped.
A different unit raises unless the declaration says `convert=True`, as for an input, and the result stays quantified.

## Wrapping a library that returns bare arrays { #wrapping-a-library }

A wrapper's output declaration is a claim about someone else's convention, and stamping takes it on trust.
Attach the unit the library documents, and use `require` so that forgetting raises:

```python
@xra.enforce(units_on_output="require")
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g m-2 d-1", convert=True)]:
    raw = some_library.gpp(ppfd.values)  # documented as ug m-2 s-1
    out = xr.DataArray(raw, dims=ppfd.dims, coords=ppfd.coords)
    return out.assign_attrs(units="ug m-2 s-1").pint.quantify()
```

The result is converted to the declared unit, here by a factor of 0.0864.
For a whole module of wrappers, bind the keyword once: `enforce = functools.partial(xra.enforce, units_on_output="require")`.

## Switching the package off { #enabled }

```python
with xra.enabled(False):
    ...
```

Off is a total no-op: no checks, no conversion and no stamping, so a function declaring `Unit("Pa")` can receive `hPa`.
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
| Output unit unreadable under `strict`, or quantified and unreadable; returned unchecked | `INFO` |
| Input or quantified output converted under `convert=True` | `DEBUG` |
| Output stamped over a different or unreadable label | `DEBUG` |

Records are emitted on every call, not once per call site like warnings.
