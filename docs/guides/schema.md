# Schema declarations

`Dims`, `Coords` and `Dtype` describe an array's structure.
A mismatch raises [`SchemaError`][xarray_annotated.schema.SchemaError], and a malformed declaration, such as duplicate names or an unknown dtype, raises `ValueError` when the function is decorated.

Use them together, with at most one of each in a hint:

```python
@xra.enforce
def column_mean(
    x: Annotated[xr.DataArray, Dims("time", "level"), Coords("level"), Dtype("float")],
) -> Annotated[xr.DataArray, Dims("time"), Dtype("float")]:
    return x.mean("level")
```

## Dims

[`Dims`][xarray_annotated.Dims] declares exactly which dimensions an array has, by name:

| Declared | Actual dims | Result | Why |
|---|---|---|---|
| `Dims("time", "x")` | `(time, x)` | pass | |
| `Dims("time", "x")` | `(x, time)` | pass | Order is free by default |
| `Dims("time", "x")` | `(time,)` | fail | A dim is missing |
| `Dims("time", "x")` | `(time, x, y)` | fail | Extra dims fail too |
| `Dims("time", "x", ordered=True)` | `(x, time)` | fail | The order is pinned |
| `Dims()` | `()` | pass | A scalar has no dims |

xarray operations do not depend on dim order until you drop to numpy, so pin it with `ordered=True` for a body that uses `.values`, `.stack` or `apply_ufunc`.

`Dims` checks names and not sizes.
For sizes, and their consistency across arguments, see [jaxtyping](../integrations/jaxtyping.md).

## Coords

[`Coords`][xarray_annotated.Coords] declares coordinates an array must carry, and any others are allowed.
A dimension can exist with no coordinate values, so this is a separate question from `Dims`:

```python
@xra.enforce
def at_site(x: Annotated[xr.DataArray, Dims("site"), Coords("site")]) -> None: ...


no_labels = xr.DataArray([1.0, 2.0], dims="site")
```

<!-- docs-test: raises SchemaError -->
```python
at_site(no_labels)
```

A declared coordinate need not be a dimension coordinate, so `Coords("station_name")` is fine on an array with dims `(site,)`.

## Dtype

[`Dtype`][xarray_annotated.Dtype] declares an exact dtype, or with a name that has no width, any dtype of that kind:

| Declared | Actual | Result | Why |
|---|---|---|---|
| `Dtype("float64")` | `float64` | pass | Either byte order |
| `Dtype("float64")` | `float32` | fail | The width differs |
| `Dtype("float")` | `float32` | pass | Any float |
| `Dtype("int")` | `uint8` | fail | `"int"` means signed, and `"uint"` unsigned |

The names with no width are `float`, `int`, `uint`, `complex`, `str`, `bytes`, `datetime64` and `timedelta64`.
This differs from numpy, where `np.dtype("float")` is `float64`.

## Reuse a declaration

Writing markers in the signature keeps each declaration in view, so it is a good default.
When signatures get long, or the same markers repeat across many functions, name a declaration with a `type` statement or an assignment:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]
```

Wrap a named declaration to add markers to it:

```python
type TypedPressure = Annotated[Pressure, Dtype("float64")]
```

An alias is checked exactly like the hint it stands for, but keep these in mind:

- **The markers move out of the signature.** A reader, an AI agent or generated documentation sees `Pressure` and has to look up what it declares, so name an alias after what it declares and keep it close to the code that uses it.
- **Only repetition pays for an alias.** It takes no marker arguments, so each distinct combination needs its own name, and a one-off function is usually clearer written out.
- **Alias what is shared.** A marker cannot be replaced, so a second `Unit` raises `ValueError`. Name the parts that repeat, such as dims and frequency, and add the rest where they are used.
- **Import aliases at run time.** `enforce` reads them at decoration, so an alias imported only under `TYPE_CHECKING` raises `NameError`.
