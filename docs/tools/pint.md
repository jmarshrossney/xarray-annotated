# Using with pint

xarray-annotated checks units, and has no opinion on how a body computes with them.
Plain arrays with a `units` attribute are fully supported, as covered in [Units declarations](../guides/units.md), so nothing on this page is required.

If you compute with [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io), the unit is held in the data and every operation works out the new one.
That closes some failure modes, such as stale labels and hand-written units, and [opens others](#where-pint-surprises), so whether it suits a body is a choice about that body.
This page covers how it fits with declarations.

```python
Q = xra.units.get_registry().Quantity

mass = xra.units.quantify(xr.DataArray([2.0, 4.0], dims="t", attrs={"units": "kg"}))
accel = xra.units.quantify(xr.DataArray([9.8, 9.8], dims="t", attrs={"units": "m/s^2"}))

force = mass * accel  # kg * m / s ** 2
xra.units.convert(force, "N")  # [19.6, 39.2] N
```

An operation that makes no sense raises:

<!-- docs-test: raises DimensionalityError -->
```python
mass + accel
```

## How checks treat a quantified array

`Unit` reads the unit a quantified array holds, without loading it, and requires the declared unit exactly, as it does on a plain array.
`xra.check_annotations` only checks, so a body receives whichever kind of array the caller passed.

What an output check proves depends on where the unit came from:

| Workflow | Input check | Output check |
|---|---|---|
| Labels only | Reads the label | Reads the label a body assigned by hand |
| Labelled inputs, [pint in the body](#in-a-declared-function) | Reads the label | Verifies a unit pint computed |
| [Quantified arrays between functions](#between-functions) | Reads the unit in the data | Verifies a unit pint computed |

## Helpers

`xra.units` has its own versions of three pint-xarray calls, which avoid problems those calls cause for labelled files:

| Use | Rather than | Because |
|---|---|---|
| [`xra.units.quantify`][xarray_annotated.units.quantify] | `.pint.quantify()` | It leaves coordinates plain. `.pint.quantify()` also quantifies them, and a label the registry cannot parse stops it. A quantified coordinate expects `.sel` labels with units, and on degrees a bare number is read as radians and silently selects the wrong row |
| [`xra.units.dequantify`][xarray_annotated.units.dequantify] | `.pint.dequantify()` | It clears `.encoding`, as packing such as `scale_factor` that described the old values silently corrupts the new ones on write. That includes packing that was still valid, so set it again before writing |
| [`xra.units.convert`][xarray_annotated.units.convert] | `.pint.to` | It keeps an array the kind it was and clears a stale packing, and it raises `DimensionalityError` rather than a `PintExceptionGroup` around one |

All three take a `DataArray`, not a `Dataset`, so work one variable at a time.
A label is written as `hPa` under [the CF registry](../guides/units.md#cf-convention-units) and as `hectopascal` on the default one, since each can read back its own spelling.

## In a declared function { #in-a-declared-function }

Call `xra.units.quantify` on each input you compute with, and return `xra.units.dequantify(xra.units.convert(result, unit))`:

```python
MOLAR_MASS_C = Q(12.011, "g / mol")  # (1)!


@xra.check_annotations
def to_mass_flux(
    flux: Annotated[xr.DataArray, Unit("umol / m**2 / s")],
) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    mass_flux = xra.units.quantify(flux) * MOLAR_MASS_C
    return xra.units.dequantify(xra.units.convert(mass_flux, "g / m**2 / day"))


flux = xr.DataArray([3.0], dims="t", attrs={"units": "umol / m**2 / s"})
to_mass_flux(flux)  # [3.113], labelled "gram / day / meter ** 2"
```

1.  A constant needs a unit too, as a bare number is dimensionless.

The caller passes and receives plain arrays, and the returned label is one pint computed.
Forget the molar mass, and the conversion fails:

<!-- docs-test: raises DimensionalityError -->
```python
xra.units.convert(xra.units.quantify(flux), "g / m**2 / day")
```

On a dask-backed array, `rolling` and `ffill` lose the unit with no warning, so run them before quantifying:

<!-- docs-test: skip -->
```python
smoothed = xra.units.quantify(x.rolling(time=3).mean())
```

`float()` and `.values` also strip the unit, with no more than a `UnitStrippedWarning`.
To leave pint on purpose, convert first and then read the magnitude, so the unit you meant is written down: `xra.units.convert(x, "hPa").data.magnitude`.

## Plain arrays between functions { #between-functions }

If you do compute with pint, pass plain arrays between functions and quantify only inside them:

- **Each body knows what it will receive:** a body written for plain arrays can give [a different answer](#where-pint-surprises) when handed a quantified one, with no error.
- **The unit at each boundary is the one declared:** on a quantified array, `p * 2` keeps the unit `p` arrived in, so a result's unit can depend on its inputs until it is converted.
- **Laziness is unaffected:** dask-backed arrays stay lazy either way, so open files with `chunks={}`.

Quantified arrays still pass every check, and `xra.units.quantify` returns one unchanged, so the functions above accept both kinds.
They are worth keeping where the code between your functions does arithmetic of its own.
If others may hand a quantified array to a body written for plain ones, start the body with `rh = xra.units.dequantify(rh)`.

## From file to file

In a script with no functions to declare, quantify when you read, compute, and dequantify when you are finished:

<!-- docs-test: skip -->
```python
ds = xr.open_dataset("in.nc", chunks={})
p = xra.units.quantify(ds["p"])

result = xra.units.convert(p, "hPa")

xra.units.dequantify(result).to_netcdf("out.nc")
```

Open with `chunks={}`, as quantifying reads an array into memory unless dask backs it.

## Where pint surprises { #where-pint-surprises }

Pint's arithmetic is consistent, but some of it is not what you would write by hand, and a body moved from plain arrays to quantified ones can change its answer:

| Expression | Result | Why |
|---|---|---|
| `rh / 100`, with `rh` in `percent` | `0.6 %` | Dividing by a number keeps the unit, so `1 - rh / 100` is 0.994. Write `1 - rh`, which is 0.4 |
| `p / 100`, with `p` in `Pa` | A hundredth of the pressure, in `Pa` | Dividing is a physical factor, and only `xra.units.convert(p, "hPa")` is the same pressure |
| `km / m` | `0.01 km / m`, for 1 km / 100 m | Prefixes are not simplified. `xra.units.convert(x, "dimensionless")` gives 10 |
| `km + m` | `1.1 km`, for 1 km + 100 m | The result takes the left operand's unit, so `m + km` is `1100 m` |
| `degC + degC`, `degC * 2` | `OffsetUnitCalculusError` | A temperature in `degC` is a point on a scale, not an amount |
| `degC - degC` | `Δ°C` | The difference between two points is an amount, and `degC + delta_degC` is `degC` again |

The pint docs explain [offset units such as temperature](https://pint.readthedocs.io/en/stable/user/nonmult.html), and are the place to look when a result surprises you.

## What pint cannot catch

Pint checks that units are consistent, not that the calculation is the one you meant.
Each of these runs without a unit error, except where noted:

| Mistake | What happens |
|---|---|
| `rate.resample(time="W-SUN").sum()` for a weekly total | 14, still labelled per day. Multiply by the time step first, as a total has no time in its denominator |
| Converting a rate to `g / m**2 / month` for a monthly total | 60.875, as pint's month is a fixed 30.44 days, where the real totals are 62 and 58 |
| Converting a daily rate to `g / m**2 / year` for an annual total | 730.5, which is true only if every day had that rate |
| Gaps in a total | `.sum()` counts a missing value as zero |
| A wrong label in a file | 1000 labelled `Pa` that is really `hPa` becomes 10 `hPa` |
| `mb` in a file | Parses as millibarn, and only a later conversion to `hPa` raises. The millibar is `mbar` |
| The wrong species | `g` and `mol` do not track carbon or CO₂, so the wrong molar mass passes |
| `differentiate` or `integrate` on a datetime axis | Values are scaled by `datetime_unit`, but the unit stays `m` rather than `m / s` |
| `rolling` or `ffill` on a dask-backed array | `.rolling(...).mean()`, `.sum()`, `.std()` and `.ffill(...)` drop the unit with no warning, as of xarray 2026.9 and pint-xarray 0.6. A declared output then raises on the missing label, unless `.assign_attrs` hides the loss. To keep the unit, compute on the plain array: `xra.units.quantify(xra.units.dequantify(x).rolling(time=3).mean())` |
