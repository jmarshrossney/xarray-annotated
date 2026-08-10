# Using with jaxtyping

[jaxtyping](https://docs.kidger.site/jaxtyping/) annotates the **dtype and shape** of an
array, and has those checks enforced by a runtime type checker —
[beartype](https://beartype.readthedocs.io/) below, though see
[Other type checkers](#other-type-checkers). `xarray-annotated` declares the things
jaxtyping has no notion of: **units**, xarray **dim names**, **coords**, and **temporal
frequency**.

The two compose directly. Neither has to give anything up.

## They are not competing syntaxes

It is easy to assume `Float[xr.DataArray, "time x"]` is jaxtyping's own version of
`Annotated`, and therefore that you must choose one. It isn't. A jaxtyping array type is
an ordinary class with a custom `__instancecheck__`, so it occupies the **base-type
slot** — exactly where `xr.DataArray` goes — and the markers stay in the metadata slot
where they always were:

```python
Annotated[Float[xr.DataArray, "time x"], Unit("Pa"), Dims("time", "x")]
#         └────────── base type ───────┘  └──────── metadata ────────┘
```

`get_type_hints(..., include_extras=True)` hands both sides what they need: jaxtyping
reads the base type, `xarray-annotated` reads the markers.

!!! note "Requires xarray-annotated ≥ 0.6"

    Earlier versions required the base type to be *exactly* `xr.DataArray` and silently
    ignored declarations behind anything else — including a jaxtyping array type. The
    check is now duck-typed on an `array_type` attribute, so jaxtyping composes without
    `xarray-annotated` importing or depending on it. Any other library exposing the
    wrapped class the same way works too.

## The full stack

```python
from typing import Annotated

import xarray as xr
from beartype import beartype
from jaxtyping import Float, jaxtyped

from xarray_annotated.schema import Dims, declare_schema
from xarray_annotated.units import Unit, declare_units

Pressure = Float[xr.DataArray, "time x"]


@jaxtyped(typechecker=beartype)
@declare_units
@declare_schema
def total(
    p: Annotated[Pressure, Unit("Pa"), Dims("time", "x")],
    q: Annotated[Pressure, Unit("Pa"), Dims("time", "x")],
) -> Annotated[Pressure, Unit("Pa")]:
    return p + q
```

That signature is now enforcing four different things:

| what | enforced by |
|---|---|
| both arrays are floating-point | jaxtyping + beartype |
| both arrays have the **same** `time` and `x` sizes | `@jaxtyped`'s symbol binding |
| dims are literally named `time` and `x` | `Dims` |
| inputs are pressures, converted to Pa if given as hPa | `Unit` |

!!! warning "Decorator order matters"

    Put `@jaxtyped(typechecker=...)` **outermost**. It then sees the call as the caller
    made it, and `@declare_units` converts underneath it. Inverting the order means the
    type check runs against arrays that units has already converted and re-stamped.

## jaxtyping's "dims" are sizes, not names

This is the trap worth reading twice. The strings in `Float[xr.DataArray, "time x"]` are
**positional size symbols**. They constrain the array to be 2-D, and they bind `time` and
`x` to consistent *lengths* across every argument in the same `@jaxtyped` call. They say
nothing about what the dimensions are *called*:

```python
hint = Float[xr.DataArray, "time x"]
da = xr.DataArray(np.zeros((2, 3)), dims=("foo", "bar"))

isinstance(da, hint)  # True — the names are never looked at
```

That an xarray dim happens to be named `time` and a jaxtyping symbol happens to be
spelled `time` is a coincidence of spelling. If you want the name enforced, you still
need `Dims("time", "x")`.

The two are complementary in the useful direction, too: `Dims` checks names but not
sizes, and jaxtyping checks sizes but not names, so declaring both is not redundant.

## What to use for what

| property | use |
|---|---|
| dtype | **jaxtyping** (`Float`, `Int`, `Float32`, dtype unions) — strictly more capable than `Dtype` |
| shape / size consistency | **jaxtyping** — `xarray-annotated` has no equivalent |
| dim *names* | `Dims` |
| coord names | `Coords` |
| units | `Unit` |
| temporal frequency | `Freq` |

`Dtype` is not deprecated — it works with no extra dependency and participates in the
[`annotate` round-trip](tooling.md#writing-annotations-programmatically-annotate). But if
you are already using jaxtyping, express the
dtype there and let `@declare_schema` handle the xarray-specific facets.

## Other type checkers

Nothing here is specific to beartype: `xarray-annotated` reads the `Annotated` metadata
and never sees which checker enforces the base type. But the checkers differ in how they
attach, and one difference bites.

[typeguard](https://typeguard.readthedocs.io/) rewrites the function's AST rather than
wrapping it, and `@jaxtyped(typechecker=typeguard.typechecked)` cannot do that — it
receives an already-defined function object. jaxtyping emits

```
InstrumentationWarning: instrumentor did not find the target function -- not typechecking
```

and the dtype and shape checks then **silently do nothing**. The units and schema
decorators are unaffected (they wrap normally), which is what makes this easy to miss:
the stack looks like it works. Use typeguard's import hook instead, installed before the
module is imported:

```python
from typeguard import install_import_hook

install_import_hook("my_package")
```

With the hook in place the full stack behaves as it does under beartype. This is a
typeguard/jaxtyping interaction — it reproduces with no `xarray-annotated` decorators in
the stack at all.

## Caveats

- **`unwrap_annotated` returns the jaxtyping class**, not `xr.DataArray`. If you have
  tooling that compares the unwrapped base type against `xr.DataArray`, it needs to
  account for that.
- **The [writer](tooling.md#writing-annotations-programmatically-annotate) takes the base
  type as its first argument**, so
  `annotate(Float[xr.DataArray, "time x"], unit="Pa")` preserves it; the default
  `annotate(unit="Pa")` builds a plain `Annotated[xr.DataArray, ...]` and does not invent
  a jaxtyping base.
- **`xarray-annotated` never imports jaxtyping.** It is not a dependency, optional or
  otherwise; the support is purely structural.
