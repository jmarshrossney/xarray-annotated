# xarray-annotated

`xarray-annotated` enables run-time validation of `xarray.DataArray` properties declared in function signatures via [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated).

- **`xarray_annotated.schema`** — structural **schema** (dims, coords, dtype), validate-only — checked, never mutated.
- **`xarray_annotated.units`** — physical **units**, checked and *converted* via [pint](https://pint.readthedocs.io/en/stable/) / [pint-xarray](https://pint-xarray.readthedocs.io). ([cf-xarray](https://cf-xarray.readthedocs.io) is an optional dependency for CF/UDUNITS unit strings.)
- **`xarray_annotated.temporal`** — the **frequency** (and phase) of a time axis, validate-only — e.g. `Freq("W-SUN")` catches a weekly series that landed on Wednesdays.

These share a common `typing.Annotated` mechanism and a global policy switch, so the decorators can be stacked and toggled together.

For example:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import declare_units, Unit
from xarray_annotated.schema import declare_schema, Dims, Dtype

@declare_units
@declare_schema
def normalise_pressure(
    p: Annotated[xr.DataArray, Dims("time", "x"), Dtype("float64"), Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p

p = xr.DataArray([[1013.0, 1000.0]], dims=["time", "x"], attrs={"units": "hPa"})
normalise_pressure(p)
# <xarray.DataArray (time: 1, x: 2)>
# array([[101300., 100000.]])
# Attributes:
#     units:    Pa
```

For full user documentation please visit **[https://jmarshrossney.github.io/xarray-annotated/](https://jmarshrossney.github.io/xarray-annotated/)**.

## Motivations

`xarray.DataArray` objects carry properties that matter for correctness but are invisible to the type system: dimensions, coordinates, and dtype are structural assertions that every array makes, and the `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`) carries a physical unit. xarray itself doesn't enforce any of these contracts at call sites — it's easy to swap dims, feed an integer array where a float is expected, or mix `hPa`/`Pa` — and while [pint](https://pint.readthedocs.io/en/stable/) / [pint-xarray](https://pint-xarray.readthedocs.io) provide unit arithmetic and conversion, they don't *declare* or *enforce* unit expectations either. `xarray-annotated` moves all of these expectations into the function signature — the contract — and enforces them transparently at run time:

1. **Declare properties in one place** — `Annotated[xr.DataArray, Dims("time", "x"), Unit("Pa")]` — so the contract is written once, in the signature.
2. **Validate automatically** via `@declare_units` / `@declare_schema` (or their primitives `check_units` / `check_schema`), under a switchable policy.

Attaching metadata to type hints with `typing.Annotated` is an increasingly common pattern in the Python ecosystem, used by libraries such as [Pydantic](https://docs.pydantic.dev/), [FastAPI](https://fastapi.tiangolo.com/) and [Typer](https://typer.tiangolo.com/). Declared properties sit naturally alongside other typed metadata rather than in a separate schema.

## Installation

`xarray-annotated` can be installed directly from GitHub using `pip`, or tools such as `uv` that wrap around it.

```sh
uv add git+https://github.com/jmarshrossney/xarray-annotated.git
```

or

```sh
pip install git+https://github.com/jmarshrossney/xarray-annotated.git
```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) require the optional `cf` extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io):

```sh
uv add "xarray-annotated[cf] @ git+https://github.com/jmarshrossney/xarray-annotated.git"
```

Currently Python versions equal to or above 3.12 are supported.

## Overview of usage

Each domain works the same way:

1. **Declare** properties in the signature with typed markers inside `Annotated` — `Unit("Pa")`, `Dims("time", "x")`, `Dtype("float64")`, `Coords("time")`, `Freq("7D")`. A single hint can carry several markers at once.
2. **Apply** the declarations — decorate with `@declare_units` (validates + converts units, stamps outputs), `@declare_schema` (validates dims/coords/dtype, passes through unchanged), and/or `@declare_freq` (validates the time axis, passes through unchanged). Stack them to check everything.

**Units** validate and convert: `check_units` reads the array's `attrs["units"]`, converts to the declared unit, and re-stamps the attribute. A dimensional mismatch (`kg` where `Pa` is declared) always raises; the policy controls the response to missing units and inexact conversions.

**Schema** validates only: `check_schema` checks dims, coords, and dtype against the declared markers and returns the array unchanged (or raises `SchemaError`). Each marker carries its own strictness — dim order, exact vs. kind-matching dtype — and an optional per-marker severity override.

**Temporal** validates only: `check_freq` infers the frequency of the array's time coordinate and compares it with the declaration. Spacing is compared semantically (a `7D` axis satisfies `Freq("7D")` *and* `Freq("W-WED")`), while the phase is compared only where you spell it — so `Freq("W")` means "weekly, any weekday" but `Freq("W-SUN")` rejects a Wednesday-anchored series. `freq_compatible` applies the same comparison to two declarations with no array in hand.

Under the hood each decorator is a thin layer over a public primitive (`check_units` / `check_schema` / `check_freq`), so you can validate by hand where a decorator doesn't fit. Each domain also exposes a signature-reader (`units_from_signature` / `schema_from_signature` / `freq_from_signature`) for static inspection.

The package-level `annotate` function builds `Annotated` hints programmatically from facet values — the inverse of the readers — and `declarations_from_signature` reads all declared facets into a uniform `Declared` value.

See the [documentation](https://jmarshrossney.github.io/xarray-annotated/) for the full walkthrough — the validation policy, choosing a pint vs. CF registry, per-marker strictness, and combining checks.

## Philosophy

`xarray-annotated` is a deliberately thin validation layer: it adds property *declaration* (via `Annotated` markers) and *enforcement* (via decorators and policy) on top of the libraries that already handle the heavy lifting — pint/pint-xarray for unit arithmetic, xarray for data structures. It is not a units engine, a type checker, or a general accessor; those spaces are already well served. Schema and temporal are validate-only by design (they assert properties without ever converting, resampling, or mutating), while units validates *and* converts. All the domains share the same annotation mechanism and the same global policy switch, so they compose cleanly and toggle together.

I developed it to serve a specific purpose in my own work, and don't plan to make it significantly more complex or feature-rich — but please feel free to raise an [issue](https://github.com/jmarshrossney/xarray-annotated/issues) or open a [pull request](https://github.com/jmarshrossney/xarray-annotated/pulls) to suggest a change or feature.

## Development

```sh
uv sync   # install the package + dev dependencies into .venv
just      # lint, typecheck, test, build docs
```

See the `justfile` for individual targets (`just lint`, `just test`, `just test-cov`, ...).

### Pre-commit hooks

A `.pre-commit-config.yaml` is included to run the same linting (ruff) and type-checking (pyright) steps on every commit. Install the hooks with:

```sh
uv run pre-commit install
```

The hooks defined are:
- `uv-lock` — keeps the lockfile in sync with `pyproject.toml`.
- `pyright` — static type checking.
- `ruff-format` / `ruff-check` — formatting and linting with auto-fix.
