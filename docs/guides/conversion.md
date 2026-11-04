# Converting units

`xarray-annotated` checks units and never changes values: a function declaring `Unit("Pa")` refuses `hPa`, even though pint could convert it.
Convert where you know the conversion is right for the data, so the decision is visible where it is made:

```python
@xra.enforce
def altitude(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)
    return alt.assign_attrs(units="m")


p = xr.DataArray([1013.0, 1000.0], dims="time", attrs={"units": "hPa"})
altitude(xra.units.convert(p, "Pa"))
```

[`convert`][xarray_annotated.units.convert] takes an array labelled in `attrs` or quantified, and returns a plain one labelled in the new unit.

- **Coordinates are left alone:** only the data is converted, so a coordinate labelled `degrees_north` neither stops it nor changes.
- **Copies:** a converted array has a new buffer, and converting a lazy array that dask does not back reads it into memory. An array already in the unit shares its buffer and stays lazy.
- **Encoding:** a converted array has empty `.encoding`, because packing such as `scale_factor` described the old values and would corrupt the new ones on write. Coordinates keep theirs. Set compression and packing again before writing.
- **Errors:** a dimensionally wrong unit raises `DimensionalityError`, and a missing or unparseable one raises `ValueError`.
- **Auditing:** every conversion is logged at `DEBUG` on the `xarray_annotated.units` logger.

It is not a check, so [`xra.enabled(False)`](policy.md#enabled) does not affect it.

## Using pint directly

`enforce` reads a quantified input's unit from the data, and the body receives a plain array labelled as declared, so `altitude(p.pint.quantify().pint.to("Pa"))` works too.
A function that computes on quantified arrays can return one, as its unit [is checked the same way](policy.md#units-outputs).
Two things to watch:

- `.pint.quantify()` also reads the `units` label of every coordinate, so one that the registry cannot parse stops it. On the default registry that includes `degrees_north`.
- `.pint.to()` keeps `.encoding`. Passing through `enforce` clears it, but call `.drop_encoding()` on an array you dequantify yourself before writing it.

## Why conversion is not automatic

Pint knows whether two units are compatible, but not whether converting between them is correct, because that depends on what the data means.
In each of these cases the conversion succeeds and the result is wrong, which is why `enforce` raises instead of doing it for you.

### Averages and totals

An annual total and a daily mean rate can both be expressed as mass per area per time:

```python
daily_mean = xr.DataArray([2.0], dims="site", attrs={"units": "g / m**2 / day"})
xra.units.convert(daily_mean, "g / m**2 / year")  # 730.5
```

That is 2.0 multiplied by 365.25, which annualises the rate.
It is only the annual total if every day of the year had that same rate, and nothing in the result says this assumption was made.

### Temperature differences

A temperature anomaly labelled `K` is a difference, but `degC` is an absolute scale, so converting applies the offset:

```python
anomaly = xr.DataArray([5.0], dims="site", attrs={"units": "K"})
xra.units.convert(anomaly, "degC")  # -268.15
```

To declare a difference, use `Unit("delta_degC")`, which is the same unit as `K` and so needs no conversion.
