# Using with jaxtyping

[jaxtyping](https://docs.kidger.site/jaxtyping/) checks dtype and shape at run time, via [beartype](https://beartype.readthedocs.io/).
Its array type sits in the base-type slot of `Annotated`, leaving the metadata slot free for markers:

<!-- docs-test: skip -->
```python
Annotated[Float[xr.DataArray, "time x"], Unit("Pa"), Dims("time", "x")]
#         └────────── base type ───────┘  └──────── metadata ────────┘
```

## The full stack

```python
from typing import Annotated

import xarray as xr
from beartype import beartype
from jaxtyping import Float, jaxtyped

import xarray_annotated as xra
from xarray_annotated import Dims, Unit

Pressure = Float[xr.DataArray, "time x"]


@jaxtyped(typechecker=beartype)
@xra.enforce
def total(
    p: Annotated[Pressure, Unit("Pa", convert=True), Dims("time", "x")],
    q: Annotated[Pressure, Unit("Pa", convert=True), Dims("time", "x")],
) -> Annotated[Pressure, Unit("Pa")]:
    return p + q
```

!!! warning "Put `@jaxtyped` outermost"

    It then sees the call as the caller made it, with `@xra.enforce` converting underneath.

Naming the array type (`Pressure`) rather than spelling it inline keeps pyright quiet, since it reads a shape string in an annotation as a forward reference.

## What to use for what

| Property | Use | Note |
|---|---|---|
| dtype | jaxtyping (`Float`, `Int`, `Float32`, unions) | More capable than `Dtype`, which needs no extra dependency |
| Sizes, and their consistency across arguments | jaxtyping | Its `"time x"` are positional size symbols, and names are never looked at |
| Dim names | `Dims` | Checks names but not sizes, so declaring both is not redundant |
| Coord names | `Coords` | |
| Units | `Unit` | |
| Temporal frequency | `Freq` | |

## Caveats

- **Use beartype, not typeguard.** On typeguard 4, jaxtyping's checks can fail to import or silently pass everything.
- **Base type:** `unwrap_annotated` returns the jaxtyping class rather than `xr.DataArray`, and `annotate(Pressure, unit="Pa")` keeps it.
