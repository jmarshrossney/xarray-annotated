# Output unit pitfalls

An input unit declaration is checked against the label the caller's data carries.
An output one is stated by the developer, because the package cannot see what a body did.
The output check then compares two statements, the label and the declaration, and this page lists the ways both can agree and still be wrong.

Results below were measured on xarray 2026.9 and pint 0.26.

## Where an output label comes from

| Source | What it shows |
|---|---|
| Inherited from an input through xarray | Nothing about the result, only that xarray kept `attrs` |
| `.assign_attrs(units=...)` in the body | What the author believed |
| [`xra.stamp_output_units`][xarray_annotated.stamp_output_units] | What the declaration says, so the output check on it passes by construction |
| Computed by pint, then dequantified | A unit worked out from the arithmetic, which is the one case the check verifies; see [Using with pint](../tools/pint.md) |

So for most bodies the output check catches disagreement, such as a stale label, and the declaration's real job is what the next function's input is compared with.

## Labels xarray keeps and drops

| Operation | Label on the result |
|---|---|
| `p * 2`, `p / 101325.0`, `1.0 - p` | `p`'s |
| `p.mean()`, `p.sum()` | `p`'s |
| `np.hypot(u, v)`, `np.degrees(np.arctan2(u, v))` | `u`'s |
| `p / (287.05 * t)`, where `p` and `t` have different labels | None |
| `xr.apply_ufunc(f, a, b, ...)` | `a`'s, on every output |

A kept label on a new quantity is stale, and `stamp_output_units` leaves it in place, so the output check raises.
That is the loud case.
A dropped label is filled from the declaration, which is right only if the body is.

`apply_ufunc` is the common trap when wrapping a library, since every output gets the first input's label:

```python
import numpy as np

frac = xr.DataArray(np.ones((2, 3)), dims=("pixel", "time"), attrs={"units": "1"})
temp = xr.DataArray(np.ones((2, 3)), dims=("pixel", "time"), attrs={"units": "degC"})
total, product = xr.apply_ufunc(
    lambda x, y: (x + y, x * y),
    frac,
    temp,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[["time"], ["time"]],
)
total.attrs, product.attrs  # ({'units': '1'}, {'units': '1'})
```

Pass `keep_attrs=False` to `apply_ufunc` so the outputs arrive unlabelled, and a bare `stamp_output_units` then states them.
`overwrite=True` does the same, but also replaces a label the body set on purpose.

## A stale label or a forgotten conversion

The package cannot tell a label left behind by arithmetic from a label that is true on values in the wrong unit.
Both raise without `overwrite`, and `overwrite` silences both:

```python
@xra.check_annotations
@xra.stamp_output_units(overwrite=True)
def fraction(
    rh: Annotated[xr.DataArray, Unit("percent")],
) -> Annotated[xr.DataArray, Unit("1")]:
    return rh  # forgot to divide by 100


rh = xr.DataArray([60.0], dims="x", attrs={"units": "percent"})
fraction(rh).values  # [60.], labelled "1", with no error
```

`.assign_attrs(units="1")` would hide it just as well.
Where the operation is itself a conversion, return `xra.units.convert(rh, "1")` instead, so that the label follows the arithmetic and neither is needed.
Keep `overwrite` for operations that change the unit without being a conversion, such as dividing by a time step or taking a ratio.

## Keep `overwrite` as narrow as the code

`overwrite=True` on a record replaces the label of every field.
If one field needed it and another carries a stale label marking a forgotten conversion, both pass.
Name the fields instead, as in `overwrite=("altitude",)`, unless every field comes from one source of bare numbers.

## Bodies that compute with pint

Leave `stamp_output_units` off a body that computes with pint and dequantifies.
There a missing label is worth an error: pint lost the unit, as `rolling` and `ffill` do on a dask-backed array, or the body went wrong.
A body that converts before dequantifying raises on the lost unit at the conversion.
One that dequantifies without converting hands the check an unlabelled array, which raises, and a stamp would fill that gap silently.

## Equal units that are different quantities

Units are compared by conversion factor, so these pass for each other:

| Declared | Accepted | But |
|---|---|---|
| `J` | `N m` | Energy and torque |
| `rad/s` | `Hz` | An angular frequency is 2π times a frequency, and pint treats `rad` as dimensionless |
| `1` | `rad`, `sr` | Different meanings |

```python
@xra.check_annotations
def angular(w: Annotated[xr.DataArray, Unit("rad/s")]) -> None: ...


angular(xr.DataArray([1.0], dims="x", attrs={"units": "Hz"}))  # passes
```

A label that is present is never respelled, so `N m` stays `N m` on an output declared `J`.

## Stamping runs when checks are off

`stamp_output_units` is an operation, like `xra.units.convert`, so [`xra.enabled(False)`](first_principles.md#enabled) does not affect it.
Code that reads `attrs["units"]` later, or a file written from the result, sees the same label either way.
The label is still only what the declaration says.

## Decorator order

`stamp_output_units` goes below `xra.check_annotations`, so the label is stated before the check reads it.
The other order is refused when the function is decorated:

```python
def depth() -> Annotated[xr.DataArray, Unit("mm")]:
    return xr.DataArray([0.0], dims="x")
```

<!-- docs-test: raises ValueError -->
```python
xra.stamp_output_units(xra.check_annotations(depth))
```

## Wrapping a library that works on bare numbers

A stamp proves nothing about the values, and neither does a hand-written label.
What catches a wrong unit here is a test against the library itself:

- compare each declared unit with the library's own documented or machine-readable units;
- pin a few absolute values, which catches a change of convention that the library forgot to relabel;
- assert physical relations that hold in one unit system and not another, which says what moved when a pinned value fails.
