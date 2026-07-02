# xarray-annotated

`xarray-annotated` enables run-time validation and conversion of `xarray.DataArray` units declared in function signatures via [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated).

It bridges [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io) — which provide the units registry and the underlying conversion machinery — with the units you declare in a signature, so each declared unit is checked and coerced whenever the function runs.
([cf-xarray](https://cf-xarray.readthedocs.io) is an optional dependency for CF/UDUNITS unit strings.)

> [!NOTE]
> Right now `xarray-annotated` supports declaring and validating physical **units** only. The intention is to grow it to cover a DataArray's full xarray *schema* — its **dims, coords, and dtype** — under the same `typing.Annotated` mechanism.

For example:

```python
from typing import Annotated
import xarray as xr
from xarray_annotated.units import declare_units

@declare_units
def normalise_pressure(
    p: Annotated[xr.DataArray, "Pa"],
) -> Annotated[xr.DataArray, "Pa"]:
    return p

p = xr.DataArray([1013.0, 1000.0], attrs={"units": "hPa"})
normalise_pressure(p)
# <xarray.DataArray (dim_0: 2)>
# array([101300., 100000.])
# Attributes:
#     units:    Pa
```

For full user documentation please visit **[https://jmarshrossney.github.io/xarray-annotated/](https://jmarshrossney.github.io/xarray-annotated/)**.

## Motivations

`xarray.DataArray`s often carry a physical unit as a free-form string in their `units` attribute (`"hPa"`, `"degC"`, `"g m-2 d-1"`).
Nothing enforces that the array a function receives is actually in the unit that function expects, so it is quite easy to mess up — mixing `hPa`/`Pa` or `m`/`mm`, or even feeding in dimensionally incompatible data.
Tools like [pint](https://pint.readthedocs.io/en/stable/) and [pint-xarray](https://pint-xarray.readthedocs.io) offer robust mechanisms for validating and converting the units attached to a `DataArray`, but enforcement is left up to the user, and manually declaring and enforcing a units policy everywhere is quite a lot of effort.

`xarray-annotated` helps by providing a transparent and low-effort policy for the declaration and validation/conversion of units on `xarray.DataArray`s.
Specifically,

1. Letting you declare the expected unit of a `DataArray` **in the function signature** — `Annotated[xr.DataArray, "Pa"]` — so the unit is part of the contract, written in exactly one place.
2. Validating and converting arrays against those declarations at run time, under a switchable policy, via the `@declare_units` decorator (or the `check_units` primitive it is built on).

Attaching metadata to type hints with `typing.Annotated` is an increasingly common pattern in the Python ecosystem, used by libraries such as [Pydantic](https://docs.pydantic.dev/), [FastAPI](https://fastapi.tiangolo.com/) and [Typer](https://typer.tiangolo.com/).
A declared unit sits naturally alongside other typed metadata rather than in a separate schema.

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

There are two steps.

1. **Declare** the expected unit of each `DataArray` in your function signatures, as `Annotated[xr.DataArray, "<unit>"]` metadata.
2. **Apply** the declarations — decorate the function with `@declare_units` to validate and convert its inputs and stamp its outputs automatically, or call `check_units(...)` directly where you'd rather not decorate.

See the [documentation](https://jmarshrossney.github.io/xarray-annotated/) for the full walkthrough — the validation policy, choosing a pint vs. CF registry, and reading declarations off a signature for your own tooling.

## Philosophy

`xarray-annotated` is a deliberately thin layer over [pint-xarray](https://pint-xarray.readthedocs.io) (and, optionally, [cf-xarray](https://cf-xarray.readthedocs.io)): pint does all the arithmetic and conversion; this package only adds unit *declaration* and *validation*.
It is not a units engine, nor a general units accessor — that space is already well served.

This is by design: the aim is to work seamlessly alongside those tools without ever getting in the way.
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
