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
For sizes, and their consistency across arguments, see [jaxtyping](../tools/jaxtyping.md).

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
