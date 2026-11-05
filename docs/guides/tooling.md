# Reading and writing declarations (advanced)

The building blocks of `@xra.enforce` are public, for validating an array by hand or for building tools on top of declarations.

## Validating by hand

| Function | Use |
|---|---|
| [`check_schema`][xarray_annotated.schema.check_schema], [`check_freq`][xarray_annotated.temporal.check_freq] | Validate one array; it is returned unchanged |
| [`check_units`][xarray_annotated.units.check_units] | Check one input; returns it unchanged |
| [`apply_output_units`][xarray_annotated.units.apply_output_units] | Check one output against a declaration; returns it unchanged |

## Checking declarations with no array

| Function | Use |
|---|---|
| [`assert_valid_unit`][xarray_annotated.units.assert_valid_unit], [`assert_valid_schema`][xarray_annotated.schema.assert_valid_schema], [`assert_valid_freq`][xarray_annotated.temporal.assert_valid_freq] | Raise `ValueError` on a malformed declaration |
| [`units_compatible`][xarray_annotated.units.units_compatible], [`units_equal`][xarray_annotated.units.units_equal] | Compare two units |
| [`dims_compatible`][xarray_annotated.schema.dims_compatible], [`dtype_compatible`][xarray_annotated.schema.dtype_compatible], [`freq_compatible`][xarray_annotated.temporal.freq_compatible] | Whether two markers can describe the same array |

To check a graph at assembly time, read each node's inputs and outputs with `declarations_from_signature` and compare them pairwise with these.

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

Assign the result to a function's `__annotations__` and `@xra.enforce` reads it as if it were hand-written.
The first positional argument is the base type, so a jaxtyping array type can be passed there.
