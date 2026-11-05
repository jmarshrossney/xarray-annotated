# Computing with pint

A `units` attribute is a claim.
xarray keeps it through arithmetic, so `p * 100` still says `hPa`, and `.assign_attrs(units=...)` says whatever it is told.
A [declaration](units.md) can check a label, but not whether the label is true.

On a [pint](https://pint.readthedocs.io/en/stable/)-quantified array the unit is held in the data, and every operation works out the new one.
**So use pint for the arithmetic inside your functions, and [pass plain arrays between them](#between-functions)**, because a declared unit that was computed is worth more than one that was assigned by hand.
A function with no unit arithmetic, or one that needs the bare numbers, [can stay plain](#plain-arrays).

```python
import numpy as np
import pandas as pd

Q = xra.units.get_registry().Quantity

mass = xra.units.quantify(xr.DataArray([2.0, 4.0], dims="t", attrs={"units": "kg"}))
accel = xra.units.quantify(xr.DataArray([9.8, 9.8], dims="t", attrs={"units": "m/s^2"}))

force = mass * accel  # kg * m / s ** 2
xra.units.convert(force, "N")  # [19.6, 39.2] N
```

Units follow through `**`, `np.sqrt`, reductions, `resample`, `groupby`, `where` and `xr.dot`, among others.
An operation that makes no sense raises:

<!-- docs-test: raises DimensionalityError -->
```python
mass + accel
```

Constants need units too, as a bare number is dimensionless: use `Q(101325.0, "Pa")` rather than `101325.0`.

## In a declared function { #in-a-declared-function }

`enforce` only checks, so a body receives exactly what the caller passed.
Pass plain arrays to the function and quantify inside it: call `xra.units.quantify` on each input you compute with, and return `xra.units.dequantify(xra.units.convert(result, unit))`.

```python
MOLAR_MASS_C = Q(12.011, "g / mol")


@xra.enforce
def to_mass_flux(
    flux: Annotated[xr.DataArray, Unit("umol / m**2 / s")],
) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    mass_flux = xra.units.quantify(flux) * MOLAR_MASS_C
    return xra.units.dequantify(xra.units.convert(mass_flux, "g / m**2 / day"))


flux = xr.DataArray([3.0], dims="t", attrs={"units": "umol / m**2 / s"})
to_mass_flux(flux)  # [3.113], labelled "gram / day / meter ** 2"
```

The arithmetic is now checked as well as the signature, and the caller receives a plain array whose label pint computed.
Forget the molar mass, and the conversion fails:

<!-- docs-test: raises DimensionalityError -->
```python
xra.units.convert(xra.units.quantify(flux), "g / m**2 / day")
```

A function with no unit arithmetic, such as a mean over time, needs neither call.

On a dask-backed array, `rolling` and `ffill` lose the unit with no warning, so run them before quantifying:

<!-- docs-test: skip -->
```python
smoothed = xra.units.quantify(x.rolling(time=3).mean())
```

See [What pint cannot catch](#what-pint-cannot-catch) for the details.

## Plain arrays between functions { #between-functions }

We recommend passing plain arrays between functions, and quantifying only inside them:

- **Each body knows what it will receive:** a body written for plain arrays can give [a different answer](#where-pint-surprises) when handed a quantified one, with no error.
- **The unit at each boundary is the one declared:** on a quantified array, `p * 2` keeps the unit `p` arrived in, so a result's unit can depend on its inputs until it is converted.
- **Laziness is unaffected:** dask-backed arrays stay lazy either way, so open files with `chunks={}`.

Quantified arrays still pass every check, as `Unit` reads the unit they hold without loading them, and `xra.units.quantify` returns one unchanged, so the functions above accept both kinds.
They are worth keeping where the code between your functions does arithmetic of its own.

`Unit` requires the declared unit exactly on either kind of array, so `hPa` raises against `Unit("Pa")`, even for a body that would give the right answer in any compatible unit.
Convert [where you call the function](units.md#converting-an-input).

## From file to file

In a script with no functions to declare, quantify when you read, compute, and dequantify when you are finished:

<!-- docs-test: skip -->
```python
ds = xr.open_dataset("in.nc", chunks={})
p = xra.units.quantify(ds["p"])

result = xra.units.convert(p, "hPa")

xra.units.dequantify(result).to_netcdf("out.nc")
```

- **Open with `chunks={}`:** quantifying reads an array into memory unless dask backs it.
- **Use `xra.units.quantify` to leave coordinates plain:** `.pint.quantify()` also quantifies them, and a label the registry cannot parse stops it. A quantified coordinate expects `.sel` labels with units, as in `.sel(lat=Q(20.0, "degree"), method="nearest")`. A bare number counts as dimensionless, which raises on metres, but on degrees is read as radians and silently selects the wrong row.
- **Use `xra.units.dequantify`, not `.pint.dequantify()`:** `.pint.to` keeps `.encoding`, and packing such as `scale_factor` that described the old values silently corrupts the new ones on write. `dequantify` clears it, including packing that was still valid, so set compression and packing again before writing.
- **Work one variable at a time:** these functions take a `DataArray`, not a `Dataset`.

A label is written as `hPa` under [the CF registry](registry.md) and as `hectopascal` on the default one, since each can read back its own spelling.

## Conversion or physical factor

On a plain array, `p / 100` could be a change of unit or a real factor of 100.
On a quantified one the two look different:

```python
p = xra.units.quantify(xr.DataArray([101325.0], dims="t", attrs={"units": "Pa"}))

xra.units.convert(p, "hPa")  # 1013.25 hPa: the same pressure
p / 100  # 1013.25 Pa: a hundredth of the pressure
```

`xra.units.convert` keeps an array the kind it was, plain or quantified.
Prefer it to `.pint.to`, which keeps a stale packing, and which raises a `PintExceptionGroup` around any `DimensionalityError`.

## Rates and totals

A per-week rate and a per-day rate are one quantity: 2 g/m²/day is 14 g/m²/week.
The period you averaged over is not part of the unit, so keep the source's denominator and let [`Freq`][xarray_annotated.Freq] say which period it is.
A total has no time in its denominator at all, and comes from multiplying by the time step:

```python
time = pd.date_range("2024-01-01", periods=60, freq="D")
rate = xra.units.quantify(
    xr.DataArray(
        np.full(60, 2.0),
        dims="time",
        coords={"time": time},
        attrs={"units": "g / m**2 / day"},
    )
)

weekly_mean = rate.resample(time="W-SUN").mean()  # 2 g / day / m ** 2
xra.units.convert(weekly_mean, "g / m**2 / week")  # 14 g / m ** 2 / week

weekly_total = (rate * Q(1.0, "day")).resample(time="W-SUN").sum()  # 14 g / m ** 2
```

Calendar months need their real lengths:

```python
monthly_mean = rate.resample(time="MS").mean()
days = xra.units.quantify(monthly_mean.time.dt.days_in_month.assign_attrs(units="day"))

monthly_mean * days  # [62, 58] g / m ** 2 for January and February 2024
```

## Plain arrays { #plain-arrays }

Pint does not fit everything.
A function may need the bare numbers, because a library or an empirical formula works on them.
Its author then states the unit of the output, which is the proper use of `.assign_attrs`:

```python
@xra.enforce
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    raw = some_library.gpp(ppfd.values)  # documented as ug m-2 s-1
    out = xr.DataArray(raw, dims=ppfd.dims, coords=ppfd.coords)
    return xra.units.convert(out.assign_attrs(units="ug / m**2 / s"), "g / m**2 / day")
```

Labelling a file's arrays when you read them is the same kind of statement.

A body written for plain arrays [may give a different answer](#where-pint-surprises) when handed a quantified one, so if others may call it that way, start the body with `rh = xra.units.dequantify(rh)`.

`float()` and `.values` also strip the unit, with no more than a `UnitStrippedWarning`.
To leave pint on purpose, convert first and then read the magnitude, so the unit you meant is written down: `xra.units.convert(x, "hPa").data.magnitude`.

## Where pint surprises { #where-pint-surprises }

Pint's arithmetic is consistent, but some of it is not what you would write by hand:

| Expression | Result | Why |
|---|---|---|
| `rh / 100`, with `rh` in `percent` | `0.6 %` | Dividing by a number keeps the unit, so `1 - rh / 100` is 0.994. Write `1 - rh`, which is 0.4 |
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
| `rate.resample(time="W-SUN").sum()` for a weekly total | 14, still labelled per day. Multiply by the time step first |
| Converting a rate to `g / m**2 / month` for a monthly total | 60.875, as pint's month is a fixed 30.44 days, where the real totals are 62 and 58 |
| Converting a daily rate to `g / m**2 / year` for an annual total | 730.5, which is true only if every day had that rate |
| Gaps in a total | `.sum()` counts a missing value as zero |
| A wrong label in a file | 1000 labelled `Pa` that is really `hPa` becomes 10 `hPa` |
| `mb` in a file | Parses as millibarn, and only a later conversion to `hPa` raises. The millibar is `mbar` |
| The wrong species | `g` and `mol` do not track carbon or CO₂, so the wrong molar mass passes |
| `differentiate` or `integrate` on a datetime axis | Values are scaled by `datetime_unit`, but the unit stays `m` rather than `m / s` |
| `rolling` or `ffill` on a dask-backed array | `.rolling(...).mean()`, `.sum()`, `.std()` and `.ffill(...)` drop the unit with no warning, as of xarray 2026.9 and pint-xarray 0.6. A declared output then raises on the missing label, unless `.assign_attrs` hides the loss. To keep the unit, compute on the plain array: `xra.units.quantify(xra.units.dequantify(x).rolling(time=3).mean())` |
