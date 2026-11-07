# Units declarations

[`Unit`][xarray_annotated.Unit] checks a value's unit and never changes its values.

## Choosing a unit registry

A [pint](https://pint.readthedocs.io/en/stable/) registry decides which unit strings parse.
The choice is process-wide, and quantities from different registries cannot be mixed, so use one registry for a whole codebase.
A library that calls `use_cf_units()` makes that choice for you.

### pint units (the default)

`pint.get_application_registry()` needs no setup, and parses strings such as `"Pa"`, `"degC"` and `"m/s"`.

### CF-convention units

UDUNITS spellings use space-separated negative exponents (`"umol m-2 s-1"`, `"g m-2 d-1"`), which plain pint rejects.
Install the [`cf` extra](../getting_started.md#installation) and activate its registry:

```python
from xarray_annotated.units import use_cf_units

use_cf_units()
```

Call [`use_cf_units`][xarray_annotated.units.use_cf_units] near the top of your entry point or package `__init__`, before importing any module that declares such units, since declarations are validated when functions are decorated.

### Your own registry

```python
import pint
from xarray_annotated.units import set_registry

ureg = pint.UnitRegistry()
ureg.define("dobson_unit = 2.687e20 * meter**-2 = DU")
set_registry(ureg)
```

[`set_registry`][xarray_annotated.units.set_registry] keeps pint and the `.pint` accessor in sync, and [`get_registry`][xarray_annotated.units.get_registry] returns the active registry.

## How units are compared

Units are compared with pint, so spellings are compared by meaning.
The unit is read from `attrs["units"]`, or from the data if the array is [pint-quantified](../tools/pint.md).

| The unit is | Input | Output |
|---|---|---|
| The declared one, however it is spelled (`pascal` for `Pa`) | Passes | Passes |
| Compatible but different (`hPa` for `Pa`) | `UnitError` | `UnitError` |
| Incompatible (`kg` for `Pa`) | `UnitError` | `UnitError` |
| Missing, unparseable or over-long | `UnitError` | `UnitError` |

## Assigning units on output { #outputs }

An output must already match its declaration when the function returns, and nothing is converted or labelled on the author's behalf.
How to get there depends on what the body does:

| The body | Do this |
|---|---|
| Keeps the unit, as with a mean over time | Nothing, as xarray keeps `attrs` |
| Computes a new quantity | State its unit with `.assign_attrs(units=...)` |
| Changes the unit of the same quantity | Call [`xra.units.convert`][xarray_annotated.units.convert] |
| Gets bare numbers from elsewhere | Build a `DataArray` and state the unit they are in |

Here the arithmetic produces a new quantity, so the body states its unit:

```python
@xra.check_annotations
def altitude(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 101325.0) ** 0.1903)
    return alt.assign_attrs(units="m")
```

And here a library works on bare numbers, whose unit the body states and then converts:

```python
@xra.check_annotations
def gpp(ppfd: xr.DataArray) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    raw = some_library.gpp(ppfd.values)  # documented as ug m-2 s-1
    out = xr.DataArray(raw, dims=ppfd.dims, coords=ppfd.coords)
    return xra.units.convert(out.assign_attrs(units="ug / m**2 / s"), "g / m**2 / day")
```

### Stale labels

xarray keeps `attrs` through arithmetic, so this result is in `Pa` and still labelled `hPa`:

```python
@xra.check_annotations
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p * 100.0


p = xr.DataArray([1013.0, 1000.0], dims="time", attrs={"units": "hPa"})
```

<!-- docs-test: raises UnitError -->
```python
to_pascals(p)
```

The package cannot tell a stale label from a forgotten conversion, where the label is true and the values are in the wrong unit.
Both raise, because a false alarm is loud and a mislabelled result is silent.
Here, return `xra.units.convert(p, "Pa")`, which changes the values and the label together.

The opposite can also happen: inputs with conflicting labels leave no label, as do `pascal` and `Pa`, and so does `xr.set_options(keep_attrs=False)`, so a body that relies on an inherited label will then raise.

### What a label cannot catch { #what-a-label-cannot-catch }

An output check shows that a body and its signature agree, not that the values are right.
Values are never inspected, so `assign_attrs` is a claim rather than a check, and it can silence a wrong body as easily as a right one:

```python
@xra.check_annotations
def daily_carbon(
    flux: Annotated[xr.DataArray, Unit("mol / m**2 / s")],
) -> Annotated[xr.DataArray, Unit("g / m**2 / day")]:
    daily = flux.resample(time="D").mean()  # forgot the molar mass
    return daily.assign_attrs(units="g / m**2 / day")
```

Declarations catch wiring mistakes, not algebra, so test the arithmetic like any other code.

This is the normal case for a stage that computes a new quantity: its body states the unit its signature declares, so its own output check catches little more than a typo.
The declaration is still useful further along, as what the next stage's input is compared with, both at run time and in [pre-flight checks](../advanced/preflight.md).

Two things narrow the gap:

- **Declare a unit only where a stage defines it.** A stage that does not change the unit, such as a mean over time, need not declare one, so the label reaches the next declared input, which checks it. The [worked example](../example.md) catches its missing stage this way.
- **Have the unit computed.** A body that computes with pint hands the output check a unit worked out from the arithmetic, at the cost of some failure modes of its own; see [Using with pint](../tools/pint.md).

## Converting an input

Convert where you know the conversion is right for the data, so the decision is visible where it is made:

```python
altitude(xra.units.convert(p, "Pa"))
```

[`convert`][xarray_annotated.units.convert] takes an array labelled in `attrs` or quantified, and returns the same kind of array in the new unit.

- **Coordinates are left alone:** only the data is converted, so a coordinate labelled `degrees_north` neither stops it nor changes.
- **Encoding:** a converted array has empty `.encoding`, because packing such as `scale_factor` described the old values and would corrupt the new ones on write. Set compression and packing again before writing.
- **Not a check:** [`xra.enabled(False)`](first_principles.md#enabled) does not affect it.

A conversion can succeed and still be wrong for the data, which is [why it is never automatic](../faq.md#why-raise-on-a-compatible-unit-instead-of-converting-it).
