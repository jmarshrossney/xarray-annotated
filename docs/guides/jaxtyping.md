# Using with jaxtyping

[jaxtyping](https://docs.kidger.site/jaxtyping/) annotates the **dtype and shape** of an
array, enforced at runtime by [beartype](https://beartype.readthedocs.io/).
`xarray-annotated` declares what jaxtyping has no notion of: **units**, xarray **dim
names**, **coords**, and **temporal frequency**. The two compose directly.

## They are not competing syntaxes

`Float[xr.DataArray, "time x"]` looks like a rival to `Annotated`, but it is an ordinary
class with a custom `__instancecheck__`. It occupies the **base-type slot** — exactly
where `xr.DataArray` goes — leaving the metadata slot free:

```python
Annotated[Float[xr.DataArray, "time x"], Unit("Pa"), Dims("time", "x")]
#         └────────── base type ───────┘  └──────── metadata ────────┘
```

`get_type_hints(..., include_extras=True)` gives both sides what they need: jaxtyping
reads the base type, `xarray-annotated` reads the markers.

!!! note "Requires xarray-annotated ≥ 0.6"

    Earlier versions required the base type to be *exactly* `xr.DataArray` and silently
    ignored declarations behind anything else. The check is now duck-typed on an
    `array_type` attribute, so jaxtyping composes without `xarray-annotated` importing or
    depending on it.

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

That signature enforces four things:

| what | enforced by |
|---|---|
| both arrays are floating-point | jaxtyping + beartype |
| both arrays have the **same** `time` and `x` sizes | `@jaxtyped`'s symbol binding |
| dims are literally named `time` and `x` | `Dims` |
| inputs are pressures, converted to Pa if given as hPa | `Unit` |

!!! warning "Put `@jaxtyped` outermost"

    It then sees the call as the caller made it, with `@xra.enforce` converting
    underneath. Inverted, the type check runs against already-converted arrays.

Binding the array type to a name (`Pressure`) rather than spelling it inline keeps
pyright quiet: it reads a shape string inside an annotation as a forward reference.

## jaxtyping's "dims" are sizes, not names

The strings in `Float[xr.DataArray, "time x"]` are **positional size symbols**. They
constrain the array to be 2-D and bind consistent *lengths* across arguments, but say
nothing about what the dimensions are *called*:

```python
da = xr.DataArray(np.zeros((2, 3)), dims=("foo", "bar"))

isinstance(da, Float[xr.DataArray, "time x"])  # True — names are never looked at
```

So `Dims` checks names but not sizes, jaxtyping checks sizes but not names. Declaring
both is not redundant.

## What to use for what

| property | use |
|---|---|
| dtype | **jaxtyping** (`Float`, `Int`, `Float32`, dtype unions) — more capable than `Dtype` |
| shape / size consistency | **jaxtyping** — `xarray-annotated` has no equivalent |
| dim *names* | `Dims` |
| coord names | `Coords` |
| units | `Unit` |
| temporal frequency | `Freq` |

`Dtype` is not deprecated — it needs no extra dependency and participates in the
[`annotate` round-trip](tooling.md#writing-annotations-programmatically-annotate) — but
if you already have jaxtyping, express the dtype there.

## Caveats

- **Use beartype, not typeguard.** On typeguard 4, jaxtyping's checks either fail to
  import or silently pass everything, depending on how the array type is spelled. Ours
  keep working either way, so the stack looks healthy while the jaxtyping half is dead.
  This is upstream and reproduces with no `xarray-annotated` present.
- **`unwrap_annotated` returns the jaxtyping class**, not `xr.DataArray` — relevant if
  you have tooling that compares the unwrapped base type.
- **The [writer](tooling.md#writing-annotations-programmatically-annotate) takes the base
  type as its first argument**, so `annotate(Pressure, unit="Pa")` preserves it; the
  default `annotate(unit="Pa")` builds a plain `Annotated[xr.DataArray, ...]`.
- **`xarray-annotated` never imports jaxtyping.** The support is purely structural.
