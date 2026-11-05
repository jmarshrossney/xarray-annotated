# Contributing

## Scope

`xarray-annotated` is a thin layer on top of `xarray` (for data structure) and `pint`/`pint-xarray` (for units).
The aim is to do one simple job well without getting in the way of other packages and workflows.

A check must be answerable from an array's metadata and indexes alone, without reading its values.
That is what lets contracts run before the data is loaded, so a feature that needs the values, such as a range or NaN constraint, is out of scope however useful it would be.

I developed it to serve a specific purpose in my own work, and don't plan to make it significantly more complex or feature-rich.

With that out of the way, please feel free to raise an [issue](https://github.com/jmarshrossney/xarray-annotated/issues) or open a [pull request](https://github.com/jmarshrossney/xarray-annotated/pulls) to suggest a change or feature.

## Development setup

```sh
uv sync   # install the package + dev dependencies into .venv
just      # lint, typecheck, test, build docs
```

See the `justfile` for individual targets (`just lint`, `just test`, `just test-cov`, ...).

A `.pre-commit-config.yaml` is included to run the same linting (ruff) and type-checking (pyright) steps on every commit.
Install the hooks with:

```sh
uv run pre-commit install
```
