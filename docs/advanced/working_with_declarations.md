# Working with declarations

A declaration is data as well as a check, so tools can read one from a signature, write one into a signature and compare it with another, all with no array in hand.

## Reading declarations

```python
from typing import Annotated, TypedDict

import xarray as xr
from xarray_annotated import Dims, Dtype, Unit, declarations_from_signature


class Output(TypedDict):
    gpp: Annotated[xr.DataArray, Unit("g m-2 d-1")]
    diagnostic: xr.DataArray


def node(
    x: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("degC")],
    plain: xr.DataArray,
) -> Output: ...


inputs, output = declarations_from_signature(node)
# inputs == {"x": Declared(unit=Unit("degC"), dims=Dims("time", "x"), dtype=Dtype("float64"))}
# output == {"gpp": Declared(unit=Unit("g m-2 d-1"))}
```

- **Omitted:** parameters and fields with no declaration.
- **Return shape:** a dict for a `TypedDict` or dataclass, including one written `R | None` or `R[int]`, one [`Declared`][xarray_annotated.Declared] for a single array, otherwise `None`.
- **Slots:** each slot of a `Declared` holds a marker or `None`, and `.schema` lists the structural markers.
- **One hint:** use [`declaration_from_hint`][xarray_annotated.declaration_from_hint].

## Writing declarations

[`annotate`][xarray_annotated.annotate] is the inverse: it builds a real `Annotated` hint, from raw values or markers.

```python
from xarray_annotated import annotate

hint = annotate(unit="Pa", dims=("time", "x"), dtype="float64", freq="7D")
# Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x"), Dtype("float64"), Freq("7D")]
```

Assign the result to a function's `__annotations__` and `@xra.check_annotations` reads it as if it were hand-written.
The first positional argument is the base type, so a jaxtyping array type can be passed there.

## Comparing declarations { #comparing }

Each declaration can look fine on its own and still disagree with the next stage.
[`declaration_conflicts`][xarray_annotated.declaration_conflicts] compares two of them, as read above:

```python
@xra.check_annotations
def to_pascals(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return xra.units.convert(p, "Pa")


@xra.check_annotations
def altitude(
    p: Annotated[xr.DataArray, Unit("hPa")],
) -> Annotated[xr.DataArray, Unit("m")]:
    alt = 44330.0 * (1.0 - (p / 1013.25) ** 0.1903)
    return alt.assign_attrs(units="m")


_, produced = xra.declarations_from_signature(to_pascals)
consumed, _ = xra.declarations_from_signature(altitude)

xra.declaration_conflicts(produced, consumed["p"])
# ["unit: Unit('Pa') and Unit('hPa') are different units"]
```

- **Partly annotated pipelines:** a facet declared on only one side cannot conflict, so you can adopt this one function at a time.
- **Units:** they conflict unless equal, matching `xra.check_annotations`. `units_compatible` alone would pass this example, which `xra.check_annotations` would then refuse.
- **`Coords`:** they never conflict, since each declaration only requires names to be present.
- **Several consumers:** compare all pairs of the declarations for one array, not each against the first. A loose `Dims` or `Dtype` is compatible with stricter ones that conflict with each other.
- **No conflict is not proof:** `Freq("D", gaps_ok=True)` is compatible with `Freq("D")`, which still raises on an axis with holes when the data arrives.
- **Malformed declarations:** these raise `ValueError` rather than conflict, and [`assert_valid_declaration`][xarray_annotated.assert_valid_declaration] asks only that.
- **The switch:** a conflict is a fact about declarations, so this ignores [`xra.enabled`](../guides/first_principles.md#enabled). A tool that should stay quiet can ask [`resolve_enabled`][xarray_annotated.resolve_enabled] first.

To run these comparisons along every edge of a pipeline with [`wiring_conflicts`][xarray_annotated.wiring_conflicts], and against its input files, see [Pre-flight checks](preflight.md#whole-graph).
