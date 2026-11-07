# ADR 0001: Exact units and plain arrays at function boundaries

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-05 |
| **Discussion** | [#31](https://github.com/jmarshrossney/xarray-annotated/issues/31), following [#26](https://github.com/jmarshrossney/xarray-annotated/issues/26) |

Results below were measured on xarray 2026.9, pint 0.26 and pint-xarray 0.6.

## Context

`Unit("hPa")` requires exactly that unit, so `Pa` raises `UnitError`.
A body that computes on quantified arrays gives the right answer for any compatible unit, so a declaration can refuse a call that would have run correctly without it.

That raises two questions:

1. Should a marker be able to declare only a physical dimension, such as "a pressure"?
2. Should arrays cross function boundaries quantified, or only be quantified inside a body?

## Decision

1. **`Unit` stays exact, and there is no dimension-only marker.** A caller with a different unit converts at the call site with `xra.units.convert`.
2. **We recommend plain arrays between functions, and take no position on how a body computes.** The declaration states the contract at the boundary. A body that computes with pint quantifies its inputs itself, so pint checks the arithmetic inside.
3. **Quantified arrays stay supported.** Every check reads the unit they hold, and `xra.units.quantify` returns one unchanged, so a body that quantifies its inputs accepts both kinds.

A function that computes with pint looks like this:

```python
Q = xra.units.get_registry().Quantity
R = Q(287.05, "J / kg / K")


@xra.check_annotations
def air_density(
    p: Annotated[xr.DataArray, Unit("Pa")],
    t: Annotated[xr.DataArray, Unit("K")],
) -> Annotated[xr.DataArray, Unit("kg / m**3")]:
    density = xra.units.quantify(p) / (R * xra.units.quantify(t))
    return xra.units.dequantify(xra.units.convert(density, "kg / m**3"))


p = xr.DataArray([1013.25], dims="t", attrs={"units": "hPa"})
t = xr.DataArray([288.15], dims="t", attrs={"units": "K"})
air_density(xra.units.convert(p, "Pa"), t)  # [1.225], labelled, and not quantified
```

## Reasons

### A dimension-only marker makes a promise about the body

An exact unit is a claim about the data, which a check can verify.
"Any pressure" is safe only if the body stays in pint, which a check cannot see.

| Body | `Pa` or `K` input | `hPa` or `degC` input | Failure |
|---|---|---|---|
| `p.data.magnitude / 100` | 1013.25 | 10.13 | Silent |
| `6.1 * np.exp(p.values / 1000)` | 6e44 | 16.8 | `UnitStrippedWarning` only |
| `(p / p0) ** 0.19`, same pressure | 1.0 | 0.417 `hPa ** 0.19 / Pa ** 0.19` | Correct, but misleading until converted |
| `p * 2` | Result in `Pa` | Result in `hPa` | Output unit depends on the input |
| `T * 2` | Works | `OffsetUnitCalculusError` | Loud, but one dimension is not one algebra |

Outputs need an exact unit in any case, since the output unit otherwise depends on the input.

A dimension also does not identify a quantity.
`J` and `N m`, `Hz` and `Bq`, and `percent` and `dimensionless` all pass a dimension check.

### The flexibility is rarely needed in a pipeline

Accepting any compatible unit pays off when input units vary and cannot be known in advance, which is the position of a library.
A pipeline usually knows its sources, so one `xra.units.convert` where the data enters removes the variation and records the decision in the code.

A library author can already accept any compatible unit, by leaving the input undeclared and starting the body with `xra.units.convert`.
That raises on a missing label or a wrong dimension and is safe for plain arrays, at the cost of a signature that no longer states the expectation.

### Mixing kinds of array is the main hazard

`enforce` only checks, so a body receives whichever kind of array the caller passed.
A body written for plain arrays can then give a different answer with no error: `1 - rh / 100` gives 0.4 for 60 `percent` on a plain array and 0.994 on a quantified one.
Passing plain arrays between functions removes this hazard.

End-to-end quantification protects only the code between functions, because the code inside can quantify for itself.
That is a real benefit where the code between functions does arithmetic of its own, which is why quantified arrays stay supported.

### Laziness does not decide it

| Step | With `chunks={}` | Without chunks |
|---|---|---|
| `check_units` | Reads nothing | Reads nothing |
| `xra.units.quantify` | Stays lazy | Reads the whole array |
| Arithmetic, `convert`, `dequantify`, `enforce` | No computes, with either kind of array at the boundary | Not measured, as `quantify` has already read the array |

Files should be opened with `chunks={}` under either approach.

### Exact outputs catch units that pint loses

On a dask-backed array, `.rolling(...).mean()`, `.sum()`, `.std()` and `.ffill(...)` drop the unit with no warning.
An exact output `Unit` raises on the missing label, unless `.assign_attrs` hides the loss.

## Alternatives

| Alternative | Why not |
|---|---|
| `Unit("hPa", convert=True)` | Removed after [#26](https://github.com/jmarshrossney/xarray-annotated/issues/26). Switching the package off changes results, and it does nothing for a pint body, which never needed the conversion |
| A separate dimension-only marker | Deferred, for the reasons above. Adding it later breaks nothing, whereas removing it would |
| `Unit("[pressure]")` | Gives one marker two meanings, with different guarantees, which tools reading declarations would have to tell apart |
| Quantified arrays throughout | Supported, but not recommended by default, because of the hazard above |

MetPy takes the dimension-based approach, with [`check_units("[pressure]")`](https://unidata.github.io/MetPy/v1.5/api/generated/metpy.units.html), and pint offers [a similar decorator](https://pint.readthedocs.io/en/stable/advanced/wrapping.html).
These are good fits for a library whose callers' units are unknown.

## Consequences

- **Explicit conversion:** a body that would work for any compatible unit still refuses all but one, and each caller with a different unit needs a `convert`.
- **Boilerplate:** a function that computes with pint has a `quantify` for each input and a `dequantify(convert(...))` on return, and the output unit is written in both the signature and the body.
- **Stable results:** `enabled(False)` does not change what a function computes, since nothing relies on `enforce` to convert.
- **Limit of the checks:** a check reads the label in the file, and pint starts from it, so a wrong label stays wrong.

## When to revisit

Reconsider a dimension-only marker if a function whose callers you do not control needs one, and leaving its input undeclared falls short.
It should then:

- be a separate marker from `Unit`, and raise at decoration if combined with it;
- apply to inputs only, with an exact `Unit` on outputs;
- refuse plain arrays, whose numbers depend on the unit.

[#31](https://github.com/jmarshrossney/xarray-annotated/issues/31) covers the remaining questions, including the marker's name and spelling.
