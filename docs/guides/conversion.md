# Converting units

By default `xarray-annotated` checks units and never changes values: a function declaring `Unit("Pa")` refuses `hPa`, even though pint could convert it.
You have two ways to get the conversion, and the first is usually the better one.

- **[Convert explicitly](#converting-explicitly-with-pint)** with pint, where you know what the data is.
- **[Opt in on the declaration](#opting-in)** with `convert=True`, so that the function converts whatever compatible unit it is given.

Conversion from a declaration is a deliberately limited feature, and it may change or go away.
The open questions are in [this GitHub issue](https://github.com/jmarshrossney/xarray-annotated/issues/26), and your experience of using it is welcome there.

## Converting explicitly with pint

Convert the array before passing it in, so the decision is visible where it is made:

```python
@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)
    return alt.assign_attrs(units="m")


p = xr.DataArray([1013.0, 1000.0], dims="time", attrs={"units": "hPa"})
altitude(p.pint.quantify().pint.to("Pa"))
```

`enforce` reads a quantified input's unit from the data, and the body receives a plain array labelled `Pa`.
The `assign_attrs` call states the unit of the result, because xarray would otherwise leave the `Pa` label on it; see [Output units](policy.md#units-outputs).
To get a plain array back yourself, add `.pint.dequantify()`.

Inside a function that computes on quantified arrays, do the same before returning:

```python
@xra.enforce
def pressure_in_pa() -> Annotated[xr.DataArray, Unit("Pa")]:
    return p.pint.quantify().pint.to("Pa")
```

### When a coordinate has a unit label { #quantify-reads-coordinates }

`.pint.quantify()` reads the `units` label of every coordinate as well as the data, so one label pint cannot parse stops the conversion.
On the default registry that includes `degrees_north`.

- **[The CF registry](registry.md)** parses `degrees_north`, but not every label, and dequantifying respells others (`km` becomes `kilometer`).
- **`.pint.quantify({"lat": None})`** avoids the error, but deletes the label from the coordinate.
- **`check_units`** converts only the data, and leaves every coordinate as it was:

```python
p = p.assign_coords(lat=("time", [10.0, 20.0], {"units": "degrees_north"}))
altitude(xra.units.check_units(p, Unit("Pa", convert=True), "p"))
```

## Opting in

Add `convert=True` to the marker:

```python
@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Unit("Pa", convert=True)],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)
    return alt.assign_attrs(units="m")


altitude(p)
```

The opt-in lives only on the marker, so reading a declaration tells you whether it converts.
`enforce` has no keyword for it and there is no global setting, which is why it can be repetitive.
To shorten it, name it in your own code:

```python
import functools

ConvertedUnit = functools.partial(Unit, convert=True)

type Pressure = Annotated[xr.DataArray, Unit("Pa", convert=True)]
```

`convert` applies only to inputs.
An output must already match its declaration, since the function's author controls it, so a body that wants another unit converts it itself; see [Output units](policy.md#units-outputs).

Equivalent spellings (`pascal` for `Pa`) need no opt-in, because the values do not change.
A dimensionally wrong unit always raises `DimensionalityError`, and an input with a missing or unparseable unit always raises `ValueError`.

## Why conversion is not the default

Pint knows whether two units are compatible, but not whether converting between them is correct, because that depends on what the data means.
In each of these cases the conversion succeeds and the result is wrong.

### Averages and totals

An annual total and a daily mean rate can both be expressed as mass per area per time.
Here a function expects an annual total and is given a daily mean rate:

```python
@xra.enforce
def annual_budget(
    total: Annotated[xr.DataArray, Unit("g / m**2 / year", convert=True)],
) -> xr.DataArray:
    return total


daily_mean = xr.DataArray([2.0], dims="site", attrs={"units": "g / m**2 / day"})
annual_budget(daily_mean)  # 730.5
```

That is 2.0 multiplied by 365.25, which annualises the rate.
It is only the annual total if every day of the year had that same rate, and nothing in the result says this assumption was made.

Without `convert=True`, the same call raises:

<!-- docs-test: raises ValueError -->
```python
@xra.enforce
def annual_budget(
    total: Annotated[xr.DataArray, Unit("g / m**2 / year")],
) -> xr.DataArray:
    return total


annual_budget(daily_mean)
```

### Temperature differences

A temperature anomaly labelled `K` is a difference, but `degC` is an absolute scale, so converting applies the offset:

```python
@xra.enforce
def anomaly_in_celsius(
    anomaly: Annotated[xr.DataArray, Unit("degC", convert=True)],
) -> xr.DataArray:
    return anomaly


anomaly_in_celsius(xr.DataArray([5.0], dims="site", attrs={"units": "K"}))  # -268.15
```

To declare a difference, use `Unit("delta_degC", convert=True)`, which converts 5 K to 5.

## Other things to know

- **Switching off changes results:** under [`xra.enabled(False)`](policy.md#enabled) nothing is converted, so a body that relies on `convert=True` receives `hPa` and computes with it as if it were `Pa`.
- **Conversion copies:** a converted array has a new buffer, which matters for large arrays. An array already in the declared unit shares its buffer with the caller.
- **Lazy arrays:** checking a unit reads no data, so an array already in the declared unit stays lazy. Converting one that dask does not back reads it into memory.
- **Encoding:** a converted array has empty `.encoding`, because packing such as `scale_factor` described the old values and would corrupt the new ones on write. Coordinates keep theirs. Set compression and packing again before writing.
- **Auditing:** every conversion is [logged](policy.md#auditing) at `DEBUG` on the `xarray_annotated.units` logger.
