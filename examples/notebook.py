# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.0",
#     "pint==0.25.3",
#     "xarray==2026.4.0",
#     "xarray-annotated==0.2.0",
# ]
#
# [tool.uv.sources]
# xarray-annotated = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(app_title="xarray-annotated example")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # xarray-annotated by example

    [`xarray-annotated`](https://github.com/jmarshrossney/xarray-annotated) lets you
    declare a property of a `DataArray` — its physical **units** or its **structure**
    (dims, coords, dtype) — right in a function signature with `typing.Annotated`, and
    have it validated (and, for units, converted) automatically by a decorator.

    This notebook walks through the basics with a small weather-station example:
    surface **pressure** and air **temperature** measured over time. We define just
    **two** decorated functions and reuse them to show what happens on both *valid* and
    *invalid* inputs. The invalid calls are wrapped in `try`/`except` so the notebook
    runs cleanly from top to bottom.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Note: we deliberately do *not* `from __future__ import annotations` — the
    # declarations are read as runtime objects out of the `Annotated` metadata, and
    # stringizing annotations would break that.
    from typing import Annotated

    import marimo as mo
    import numpy as np
    import pint
    import xarray as xr

    from xarray_annotated.schema import Coords, Dims, SchemaError, declare_schema
    from xarray_annotated.units import declare_units, policy

    return (
        Annotated,
        Coords,
        Dims,
        SchemaError,
        declare_schema,
        declare_units,
        mo,
        np,
        pint,
        policy,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Some sample data

    A DataArray typically carries its unit as a free-form string in `attrs["units"]`.
    Our station reports **pressure in hectopascals** (`hPa`) and **temperature in
    kelvin** (`K`) over five days — realistic, but not the units our analysis code
    below expects. That mismatch is exactly what `xarray-annotated` handles for us.
    """)
    return


@app.cell
def _(np, xr):
    time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")

    pressure = xr.DataArray(
        [1013.0, 1000.0, 1007.0, 995.0, 1002.0],
        dims="time",
        coords={"time": time},
        attrs={"units": "hPa"},
    )

    temperature = xr.DataArray(
        [290.1, 291.4, 289.7, 292.0, 290.3],
        dims="time",
        coords={"time": time},
        attrs={"units": "K"},
    )
    return pressure, temperature


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Checking and converting units

    Declare the unit a function *expects* as a string in the `Annotated` metadata of a
    `DataArray` parameter (and return), then apply `@declare_units`. On each call it
    validates and converts every declared input to the expected unit, and stamps the
    declared unit onto the output.
    """)
    return


@app.cell
def _(Annotated, declare_units, xr):
    @declare_units
    def standardise_pressure(
        p: Annotated[xr.DataArray, "Pa", "surface air pressure"],
    ) -> Annotated[xr.DataArray, "Pa"]:
        """Return the pressure in SI units (pascals)."""
        return p

    return (standardise_pressure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### A valid call

    Our data is in `hPa`; the function wants `Pa`. Those are dimensionally compatible,
    so `@declare_units` silently converts (multiplying by 100) and re-stamps the result
    as `Pa` — no manual bookkeeping at the call site.
    """)
    return


@app.cell
def _(pressure, standardise_pressure):
    standardised = standardise_pressure(pressure)
    standardised.values, standardised.units
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### An invalid call — caught

    Now suppose an upstream bug hands us **mass** (`kg`) where pressure was expected.
    That is not a compatible conversion, so `@declare_units` raises a
    `pint.DimensionalityError` — *regardless of policy*, because it signals genuinely
    wrong data. We catch it here so the notebook keeps running.
    """)
    return


@app.cell
def _(pint, standardise_pressure, xr):
    bad_pressure = xr.DataArray([12.0, 13.0], dims="time", attrs={"units": "kg"})

    try:
        standardise_pressure(bad_pressure)
    except pint.DimensionalityError as exc:
        print(f"Caught a {type(exc).__name__}:\n{exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Combining structure and units

    Structural checks work the same way, via `@declare_schema` and the `Dims`, `Coords`
    and `Dtype` markers — but they only *validate* (arrays pass through unchanged). You
    can **stack both decorators** to check structure and units at once.

    `temperature_anomaly` below requires a `("time",)` array that *has* a `time`
    coordinate and is in `degC`. Put `@declare_units` outermost so the structural check
    sees the array already in its declared units.
    """)
    return


@app.cell
def _(Annotated, Coords, Dims, declare_schema, declare_units, xr):
    @declare_units
    @declare_schema
    def temperature_anomaly(
        t: Annotated[xr.DataArray, Dims("time"), Coords("time"), "degC"],
    ) -> Annotated[xr.DataArray, Dims("time"), "degC"]:
        """Deviation of each reading from the time-mean temperature."""
        return t - t.mean("time")

    return (temperature_anomaly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### A valid call

    Our `temperature` is in `K` with a `time` dim and coordinate. `@declare_units`
    converts `K` → `degC` (an affine shift), then `@declare_schema` confirms the dims
    and coords, and the anomaly is returned in `degC` with the `time` dimension intact.
    """)
    return


@app.cell
def _(temperature, temperature_anomaly):
    anomaly = temperature_anomaly(temperature)
    anomaly
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### An invalid call — caught

    Here we pass an array whose dimension is `x`, not `time`. The structural contract is
    violated, so `@declare_schema` raises a `SchemaError` *before the body ever runs* —
    pinpointing the wiring bug instead of failing deep inside `.mean("time")`.
    """)
    return


@app.cell
def _(SchemaError, temperature_anomaly, xr):
    wrong_dims = xr.DataArray([1.0, 2.0, 3.0], dims="x", attrs={"units": "degC"})

    try:
        temperature_anomaly(wrong_dims)
    except SchemaError as exc:
        print(f"Caught a {type(exc).__name__}:\n{exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Tuning the policy

    Each domain has a small **policy** governing what happens on a validation event, and
    a `policy(...)` context manager to scope overrides. By default a value-changing unit
    conversion is silent; here we ask units to *warn* on any inexact conversion, so the
    `K` → `degC` shift is flagged while still converting.
    """)
    return


@app.cell
def _(policy, temperature, temperature_anomaly):
    import warnings

    with warnings.catch_warnings(record=True) as caught, policy(on_inexact="warn"):
        warnings.simplefilter("always")
        temperature_anomaly(temperature)

    for w in caught:
        print(f"{w.category.__name__}: {w.message}")
    return


if __name__ == "__main__":
    app.run()
