# Accessing declarations

A declaration is data as well as a check, so tools can read one from a signature and write one into a signature.

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
