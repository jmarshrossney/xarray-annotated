# Units

`Unit` checks a value's unit and never changes its values.
Units are checked with [pint](https://pint.readthedocs.io/en/stable/), so spellings are compared by meaning, and strings such as `"umol m-2 s-1"` need [the CF registry](registry.md).

The unit is read from `attrs["units"]`, or from the data if the array is quantified.

| The unit is | Input | Output |
|---|---|---|
| The declared one, however it is spelled (`pascal` for `Pa`) | Passes | Passes |
| Compatible but different (`hPa` for `Pa`) | `ValueError` | `ValueError` |
| Incompatible (`kg` for `Pa`) | `DimensionalityError` | `DimensionalityError` |
| Missing, unparseable or over-long | `ValueError` | `ValueError` |

These are checks only.
The body receives the array you passed, quantified or not and with its label spelled as it was, and you receive the array the body returned.

## Where the unit comes from { #where-the-unit-comes-from }

`Unit` is most valuable where units are held in a `units` attribute, as is common in environmental datasets, because nothing else checks them.
How much a check proves depends on where the unit comes from:

| Workflow | Input check | Output check |
|---|---|---|
| Labels only | Reads the label | Reads the label a body assigned by hand |
| Labelled inputs, [pint in the body](pint.md#in-a-declared-function) (recommended) | Reads the label | Verifies a unit pint computed |
| [Quantified arrays between functions](pint.md#between-functions) | Reads the unit in the data | Verifies a unit pint computed |

A check on a label cannot tell whether the label is true.

## Converting an input

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

[`convert`][xarray_annotated.units.convert] takes an array labelled in `attrs` or quantified, and returns the same kind of array in the new unit.

- **Coordinates are left alone:** only the data is converted, so a coordinate labelled `degrees_north` neither stops it nor changes.
- **Copies:** a converted array has a new buffer, and converting a lazy array that dask does not back reads it into memory. An array already in the unit shares its buffer and stays lazy.
- **Encoding:** a converted array has empty `.encoding`, because packing such as `scale_factor` described the old values and would corrupt the new ones on write. Coordinates keep theirs. Set compression and packing again before writing.
- **Auditing:** every conversion is logged at `DEBUG` on the `xarray_annotated.units` logger.

It is not a check, so [`xra.enabled(False)`](troubleshooting.md#enabled) does not affect it.

### Why conversion is not automatic

Pint knows whether two units are compatible, but not whether converting between them is correct, because that depends on what the data means.
In each of these cases the conversion succeeds and the result is wrong.

An annual total and a daily mean rate can both be expressed as mass per area per time, but this only gives the annual total if every day of the year had that same rate:

```python
daily_mean = xr.DataArray([2.0], dims="site", attrs={"units": "g / m**2 / day"})
xra.units.convert(daily_mean, "g / m**2 / year")  # 730.5
```

A temperature anomaly labelled `K` is a difference, but `degC` is an absolute scale, so converting applies the offset.
To declare a difference, use `Unit("delta_degC")`, which is the same unit as `K`.

```python
anomaly = xr.DataArray([5.0], dims="site", attrs={"units": "K"})
xra.units.convert(anomaly, "degC")  # -268.15
```

## Output units { #outputs }

A function's author controls what it returns, so an output must already match its declaration, and nothing is converted or labelled on the author's behalf.

A label that differs could be a forgotten conversion, where the label is true and the values are in the wrong unit, or a stale label left behind by xarray, which keeps `attrs` through arithmetic.
The package cannot tell them apart, and both raise, because a false alarm is loud and a mislabelled result is silent.

Compute with pint where you can, so the unit follows from the arithmetic: see [Computing with pint](pint.md).
A body that needs the bare numbers [states the unit it returns](pint.md#plain-arrays) with `.assign_attrs`, as `altitude` does above.

Whether a label survives depends on xarray: inputs with conflicting labels leave none, as do `pascal` and `Pa`, and so does `xr.set_options(keep_attrs=False)`, so a body that relies on an inherited label will then raise.

### What a label cannot catch { #what-a-label-cannot-catch }

Values are never inspected, so `assign_attrs` is a claim rather than a check, and it can silence a wrong body as easily as a right one:

```python
@xra.enforce
def daily_carbon(
    flux: Annotated[xr.DataArray, Unit("mol / m**2 / s")],
) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    daily = flux.resample(time="D").mean()  # forgot the molar mass
    return daily.assign_attrs(units="g / m**2 / day")
```

Declarations catch wiring mistakes, not algebra.
Two things keep the unit tied to the values:

- **Let pint carry the unit.** Compute on a quantified array, and the return is checked against the declaration. Here the forgotten factor raises `DimensionalityError`; see [Computing with pint](pint.md#in-a-declared-function).

    <!-- docs-test: skip -->
    ```python
    return xra.units.quantify(flux).resample(time="D").mean()
    ```

- **Declare a unit only where a stage defines it.** A stage that does not change the unit, such as a mean over time, need not declare one, so the label reaches the next declared input, which checks it. The [worked example](../example.md) catches its missing stage this way.
