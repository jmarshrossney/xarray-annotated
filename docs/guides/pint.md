# Computing with pint

A `units` attribute is a claim.
xarray keeps it through arithmetic, so `p * 100` still says `hPa`, and `.assign_attrs(units=...)` says whatever it is told.
A [declaration](units.md) can check a label, but not whether the label is true.

On a [pint](https://pint.readthedocs.io/en/stable/)-quantified array the unit is held in the data, and every operation works out the new one.
**So use pint wherever it is feasible**, because a declared unit that was computed is worth more than one that was assigned by hand.
Plain arrays with a `units` attribute stay fully supported, for the [cases where pint does not fit](#plain-arrays).

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

## From file to file

Quantify when you read, compute, and dequantify when you are finished:

<!-- docs-test: skip -->
```python
ds = xr.open_dataset("in.nc", chunks={})
p = xra.units.quantify(ds["p"])

result = xra.units.convert(p, "hPa")

xra.units.dequantify(result).to_netcdf("out.nc")
```

- **Open with `chunks={}`:** quantifying reads an array into memory unless dask backs it.
- **Use `xra.units.quantify`, not `.pint.quantify()`:** the accessor also quantifies coordinates. One the registry cannot parse stops it, and on one in degrees, `.sel(lat=20.0, method="nearest")` reads the bare number as radians and silently selects the wrong row.
- **Use `xra.units.dequantify`, not `.pint.dequantify()`:** `.pint.to` keeps `.encoding`, and packing such as `scale_factor` that described the old values silently corrupts the new ones on write. `dequantify` clears it, including packing that was still valid, so set compression and packing again before writing.
- **Work one variable at a time:** these functions take a `DataArray`, not a `Dataset`.

A label is written as `hPa` under [the CF registry](registry.md) and as `hectopascal` on the default one, since each can read back its own spelling.

## In a declared function

`enforce` only checks, so a body receives exactly what the caller passed.
A body written for pint starts by quantifying each input, which accepts both kinds of array, and returns its result with no `.assign_attrs`:

```python
MOLAR_MASS_C = Q(12.011, "g / mol")


@xra.enforce
def to_mass_flux(
    flux: Annotated[xr.DataArray, Unit("umol / m**2 / s")],
) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    mass_flux = xra.units.quantify(flux) * MOLAR_MASS_C
    return xra.units.convert(mass_flux, "g / m**2 / day")


flux = xr.DataArray([3.0], dims="t", attrs={"units": "umol / m**2 / s"})
to_mass_flux(flux)  # [3.113] g / day / m ** 2
```

The arithmetic is now checked as well as the signature.
Forget the molar mass, and the conversion fails:

<!-- docs-test: raises DimensionalityError -->
```python
xra.units.convert(xra.units.quantify(flux), "g / m**2 / day")
```

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
The period you averaged over is not part of the unit, so keep the source's denominator and let [`Freq`](declaring.md#frequency) say which period it is.
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

Because `enforce` only checks, a body written for plain arrays can be handed a quantified one, and [may then give a different answer](#where-pint-surprises).
Starting with `xra.units.dequantify` makes it correct for both:

```python
@xra.enforce
def dryness(
    rh: Annotated[xr.DataArray, Unit("percent")],
) -> Annotated[xr.DataArray, Unit("dimensionless")]:
    rh = xra.units.dequantify(rh)
    return (1 - rh / 100).assign_attrs(units="dimensionless")


rh = xr.DataArray([60.0], dims="t", attrs={"units": "percent"})
dryness(rh)  # 0.4
dryness(xra.units.quantify(rh))  # 0.4
```

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
