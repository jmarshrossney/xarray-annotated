# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cf-xarray==0.11.3",
#     "marimo",
#     "numpy==2.5.1",
#     "pint==0.25.3",
#     "xarray==2026.7.0",
#     "xarray-annotated==0.5.0",
# ]
#
# [tool.uv.sources]
# xarray-annotated = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(app_title="xarray-annotated: a worked pipeline")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # A worked pipeline

    This notebook processes a year of synthetic eddy-covariance flux data from a forest
    site into two products:

    1. an **annual carbon budget** — how much carbon the site took up over the year;
    2. **weekly GPP**, validated against a satellite retrieval.

    It is a realistic pipeline with a realistic set of bugs — the kind that survive code
    review because every line looks reasonable. We run it three times: as first written,
    then with declarations added, then corrected.

    This notebook lives in the repository at
    [`examples/notebook.py`](https://github.com/jmarshrossney/xarray-annotated/tree/main/examples).
    """)
    return


@app.cell
def _():
    import warnings
    from typing import Annotated, TypedDict

    import marimo as mo
    import numpy as np
    import pint
    import xarray as xr

    from xarray_annotated.schema import (
        Coords,
        Dims,
        Dtype,
        SchemaError,
        declare_schema,
    )
    from xarray_annotated.temporal import Freq, FreqError, declare_freq
    from xarray_annotated.units import Unit, declare_units, use_cf_units

    # Flux data is spelled the CF/UDUNITS way ("umol m-2 s-1"), which plain pint
    # cannot parse. This is a one-time, process-wide choice.
    use_cf_units()

    # xarray drops `attrs` through arithmetic by default, which would throw away the
    # unit metadata this pipeline depends on. Keep it.
    xr.set_options(keep_attrs=True)

    # Molar mass of carbon, g mol-1.
    M_C = 12.011

    # Seconds per day.
    SEC_PER_DAY = 86400.0

    # 1 umol m-2 s-1 sustained for a day, expressed as g C m-2 d-1.
    UMOL_S_TO_G_D = 1e-6 * M_C * SEC_PER_DAY
    return (
        Annotated,
        Coords,
        Dims,
        Dtype,
        Freq,
        FreqError,
        SchemaError,
        TypedDict,
        UMOL_S_TO_G_D,
        Unit,
        declare_freq,
        declare_schema,
        declare_units,
        mo,
        np,
        pint,
        warnings,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The site

    A temperate deciduous forest at 52°N, logging every **30 minutes** for one year.
    The logger gives us four series:

    | Variable | Unit | Note |
    |---|---|---|
    | `nee_raw` | `umol m-2 s-1` | net ecosystem exchange; **negative means uptake** |
    | `tair_raw` | `K` | air temperature, as stored |
    | `ppfd_raw` | `umol m-2 s-1` | photosynthetic photon flux density |
    | `qc_raw` | — | quality flag, `int8`: 0 good, 1 moderate, 2 bad |

    We also have a satellite GPP product to validate against, on a coarser grid:

    | Variable | Unit | Note |
    |---|---|---|
    | `sat_gpp` | `g m-2 d-1` | weekly mean GPP, **week-ending Sunday** |

    The data is synthetic but physically plausible: a real diurnal and seasonal
    cycle, a Q10 respiration response, and a saturating light response.
    """)
    return


@app.cell(hide_code=True)
def _(UMOL_S_TO_G_D, np, xr):
    def generate_synthetic_data():
        _rng = np.random.default_rng(20240301)

        _time = np.arange("2023-01-01", "2024-01-01", np.timedelta64(30, "m"), dtype="datetime64[s]")
        _n = _time.size
        _doy = _time.astype("datetime64[D]").astype(int) - np.datetime64("2023-01-01", "D").astype(int)
        _hour = (_time.astype("datetime64[s]").astype(int) % 86400) / 3600.0

        # Solar geometry -> photosynthetic photon flux density
        _decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (_doy - 80) / 365.25)
        _lat = np.deg2rad(52.0)
        _ha = np.deg2rad(15.0 * (_hour - 12.0))
        _sin_elev = np.clip(
            np.sin(_lat) * np.sin(_decl) + np.cos(_lat) * np.cos(_decl) * np.cos(_ha),
            0,
            None,
        )
        _ppfd = (2100.0 * _sin_elev * (0.85 + 0.15 * _rng.normal(size=_n).clip(-1, 1))).clip(0)

        # Air temperature: seasonal + diurnal cycle, degC
        _tair_c = (
            9.5
            + 8.5 * np.sin(2 * np.pi * (_doy - 110) / 365.25)
            + 4.0 * np.sin(2 * np.pi * (_hour - 9) / 24.0)
            + 0.8 * _rng.normal(size=_n)
        )

        # Respiration: Q10 response. Photosynthesis: saturating light response, scaled by LAI.
        _reco = 2.60 * 2.0 ** ((_tair_c - 10.0) / 10.0)
        _lai = 0.25 + 0.75 * np.clip(np.sin(np.pi * (_doy - 90) / 190.0), 0, None)
        _gpp_max = 21.4 * _lai
        _gpp = np.where(_ppfd > 0, 0.055 * _ppfd * _gpp_max / (0.055 * _ppfd + _gpp_max), 0.0) / (
            1.0 + np.exp(-(_tair_c - 2.0))
        )

        _nee = _reco - _gpp + 0.35 * _rng.normal(size=_n)

        _qc = np.zeros(_n, dtype="int8")
        _qc[_rng.random(_n) < 0.07] = 1
        _qc[_rng.random(_n) < 0.02] = 2

        def _series(values, units, dtype="float64"):
            return xr.DataArray(
                values.astype(dtype),
                dims="time",
                coords={"time": _time},
                attrs={"units": units},
            )

        nee_raw = _series(_nee, "umol m-2 s-1")
        tair_raw = _series(_tair_c + 273.15, "K")
        ppfd_raw = _series(_ppfd, "umol m-2 s-1")
        qc_raw = _series(_qc, "1", dtype="int8")

        # The satellite product: weekly mean GPP, week-ending Sunday, as a mass flux.
        # Built from the true half-hourly GPP, with a retrieval error on top.
        _gpp_daily = _series(_gpp, "umol m-2 s-1").resample(time="D").mean() * UMOL_S_TO_G_D
        _gpp_weekly = _gpp_daily.resample(time="W-SUN").mean()
        sat_gpp = (_gpp_weekly * (1.0 + 0.08 * _rng.normal(size=_gpp_weekly.sizes["time"]))).assign_attrs(
            units="g m-2 d-1"
        )
        return nee_raw, ppfd_raw, qc_raw, sat_gpp, tair_raw

    return (generate_synthetic_data,)


@app.cell
def _(generate_synthetic_data, xr):
    nee_raw, ppfd_raw, qc_raw, sat_gpp, tair_raw = generate_synthetic_data()
    print(f"{nee_raw.sizes['time']} half-hourly records, {xr.infer_freq(nee_raw.time)}")
    print(f"flagged bad or moderate: {int((qc_raw > 0).sum())}")
    print(f"satellite GPP: {sat_gpp.sizes['time']} weekly means, {xr.infer_freq(sat_gpp.time)}")
    return nee_raw, ppfd_raw, qc_raw, sat_gpp, tair_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The pipeline as first written

    Five stages, no declarations anywhere. Each one states what it assumes in its
    docstring, which is where such things usually get written down.
    """)
    return


@app.cell
def _(nee_raw, ppfd_raw, qc_raw, sat_gpp, tair_raw):
    def screen_quality(flux, qc):
        """Drop records not flagged good. Assumes qc is the flag array for flux."""
        return flux.where(qc == 0)

    def partition_fluxes(nee, tair, ppfd):
        """Partition NEE into GPP and respiration. Assumes tair in degC."""
        reco = 2.60 * 2.0 ** ((tair - 10.0) / 10.0)
        gpp = (reco - nee).where(ppfd > 5.0, 0.0)  # no photosynthesis in the dark
        return gpp, reco

    def daily_carbon(flux):
        """Half-hourly flux -> daily mean carbon flux, g C m-2 d-1."""
        return flux.resample(time="D").mean()

    def weekly_mean(daily):
        """Daily -> weekly mean. Weeks end on Sunday."""
        return daily.resample(time="W-WED").mean()

    def compare_with_satellite(modelled, observed):
        """Bias and RMSE. Assumes both are weekly means on the same grid."""
        diff = modelled - observed
        return float(diff.mean()), float((diff**2).mean() ** 0.5)

    nee = screen_quality(nee_raw, qc_raw)
    gpp, _reco = partition_fluxes(nee, tair_raw, ppfd_raw)

    # Product 1: the annual carbon budget.
    print(f"annual NEE = {float(daily_carbon(nee).sum()):+.0f} g C m-2 yr-1   (negative = sink)")

    # Product 2: weekly GPP, against the satellite retrieval.
    gpp_weekly = weekly_mean(daily_carbon(gpp))
    _bias, _rmse = compare_with_satellite(gpp_weekly, sat_gpp)
    print(
        f"\nweekly GPP = {gpp_weekly.sizes['time']} weeks from "
        f"{str(gpp_weekly.time.values[0])[:10]}, mean {float(gpp_weekly.mean()):.3g}"
    )
    print(f"vs satellite: bias {_bias:+.2f}, rmse {_rmse:.2f} g C m-2 d-1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Nothing raised. Every one of those four docstring assumptions is false.

    - **`tair` in degC.** It arrives in kelvin, so the Q10 exponent is `(290-10)/10 = 28`
      rather than `(17-10)/10 = 0.7`. This one screams — mean weekly GPP of `2.6e8` where
      4 would be respectable — and you find it in minutes.
    - **`g C m-2 d-1`.** `daily_carbon` resamples but never converts moles to grams, so its
      output is still a molar flux. The factor happens to be **1.04**, so the annual NEE of
      `-474` is within 4% of the right answer and looks entirely respectable. It is the
      most dangerous number in the notebook.
    - **Weeks end on Sunday.** They end on Wednesday.
    - **Both on the same grid.** They are not. xarray aligns on the time coordinate, finds
      that Wednesdays and Sundays never coincide, and every statistic comes back `nan`.

    That last one is the interesting case. A `nan` is a signal, but an anonymous one: it
    tells you something is wrong somewhere, and nothing about what or where. Working out
    that two regular 53-point weekly series are three days out of phase is an afternoon.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declaring what each stage assumes

    We rebuild the pipeline stage by stage. Each assumption moves out of the docstring and
    into the signature, where it is enforced rather than merely stated.

    ### Stage 1: quality screening

    A half-hourly series with a real `time` coordinate, and an `int8` flag array alongside
    it. Returns `float64`, because gaps become NaN and an integer array cannot hold NaN.
    """)
    return


@app.cell
def _(
    Annotated,
    Coords,
    Dims,
    Dtype,
    Freq,
    Unit,
    declare_freq,
    declare_schema,
    declare_units,
    xr,
):
    @declare_units
    @declare_schema
    @declare_freq
    def screen_quality_new(
        flux: Annotated[
            xr.DataArray,
            Dims("time"),
            Coords("time"),
            Unit("umol m-2 s-1"),
            Freq("30min"),
        ],
        qc: Annotated[xr.DataArray, Dims("time"), Dtype("int8")],
    ) -> Annotated[
        xr.DataArray,
        Dims("time"),
        Dtype("float64"),
        Unit("umol m-2 s-1"),
        Freq("30min"),
    ]:
        """Drop records not flagged as good quality."""
        return flux.where(qc == 0)

    return (screen_quality_new,)


@app.cell
def _(nee_raw, qc_raw, screen_quality_new):
    nee_clean = screen_quality_new(nee_raw, qc_raw)
    print(f"gaps introduced: {int(nee_clean.isnull().sum())}")
    return (nee_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What it catches.** Two bad exports. One has a `time` *dimension* but no `time`
    *coordinate* — common when a file is read with the index column mislabelled, and
    otherwise survives until something calls `.resample`, several stages away. The other is
    from a site that logs hourly: a perfectly good file, for a different pipeline.
    """)
    return


@app.cell
def _(FreqError, SchemaError, nee_raw, qc_raw, screen_quality_new):
    _no_coord = nee_raw.drop_vars("time")
    _hourly = nee_raw.resample(time="60min").mean()

    try:
        screen_quality_new(_no_coord, qc_raw)
    except SchemaError as exc:
        print(f"{type(exc).__name__}: {exc}")

    try:
        screen_quality_new(_hourly, qc_raw.resample(time="60min").first())
    except FreqError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 2: flux partitioning

    *Assumes tair in degC* becomes `Unit("degC")`, and the kelvin input is now converted at
    the boundary rather than blowing up inside the body. Two outputs need a named structure
    to hang declarations on, so the tuple becomes a `TypedDict`, validated per field.
    """)
    return


@app.cell
def _(Annotated, Dims, TypedDict, Unit, declare_units, xr):
    class Partitioned(TypedDict):
        """Gross fluxes, both sign-positive."""

        gpp: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]
        reco: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]

    @declare_units
    def partition_fluxes_new(
        nee: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
        tair: Annotated[xr.DataArray, Unit("degC")],
        ppfd: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
    ) -> Partitioned:
        """Partition NEE into GPP and respiration via a Q10 model."""
        reco = 2.60 * 2.0 ** ((tair - 10.0) / 10.0)
        gpp = (reco - nee).where(ppfd > 5.0, 0.0)  # no photosynthesis in the dark
        return {"gpp": gpp, "reco": reco}

    return (partition_fluxes_new,)


@app.cell
def _(nee_clean, partition_fluxes_new, ppfd_raw, tair_raw):
    fluxes = partition_fluxes_new(nee_clean, tair_raw, ppfd_raw)

    print(f"mean GPP  = {float(fluxes['gpp'].mean()):.2f} {fluxes['gpp'].units}")
    print(f"mean RECO = {float(fluxes['reco'].mean()):.2f} {fluxes['reco'].units}")
    return (fluxes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Those are plausible values, and no conversion was written anywhere in the pipeline.
    Here is the same model without the declaration in front of it:
    """)
    return


@app.cell
def _(tair_raw):
    _reco_if_kelvin = 2.60 * 2.0 ** ((tair_raw - 10.0) / 10.0)
    print(f"mean RECO given K = {float(_reco_if_kelvin.mean()):.3e} umol m-2 s-1")
    print("(a plausible value is ~2.8)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 3: daily carbon flux

    Writing the docstring's *g C m-2 d-1* down as `Unit("g m-2 d-1")` is what exposes the
    second bug. The body resamples and nothing else, so it returns a *molar* flux; the
    declaration claims a *mass* flux. No library can bridge those on its own — that needs
    the molar mass of carbon — so the conversion has to be written, and now there is a
    statement in the signature saying it must be.

    `Freq("30min")` in and `Freq("D")` out pin the temporal side.
    """)
    return


@app.cell
def _(Annotated, Freq, UMOL_S_TO_G_D, Unit, declare_freq, declare_units, xr):
    @declare_units
    @declare_freq
    def daily_carbon_new(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("30min")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")]:
        """Mean half-hourly molar flux -> daily carbon mass flux."""
        return flux.resample(time="D").mean() * UMOL_S_TO_G_D

    return (daily_carbon_new,)


@app.cell
def _(daily_carbon_new, nee_clean):
    nee_daily = daily_carbon_new(nee_clean)
    print(f"{nee_daily.sizes['time']} daily means, {nee_daily.units}")
    return (nee_daily,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What it catches — with a caveat worth understanding.** By default a declared output
    is *stamped*, not checked: the array comes back labelled `g m-2 d-1` whatever the body
    did. Had we written the declaration and left the body alone, the label would have been
    a lie and nothing would have complained.

    The `on_output="strict"` policy checks instead of stamping. It works here because a
    bare `resample().mean()` carries its input's label forward truthfully:
    """)
    return


@app.cell
def _(Annotated, Freq, Unit, declare_freq, declare_units, nee_clean, pint, xr):
    @declare_units(on_output="strict")
    @declare_freq
    def daily_carbon_unconverted(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("30min")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")]:
        """The original body, with the declaration it should have had."""
        return flux.resample(time="D").mean()

    try:
        daily_carbon_unconverted(nee_clean)
    except pint.DimensionalityError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Moles are not grams under any reading, so this raises rather than warns.

    But `strict` is not a free upgrade. Add the conversion and the body becomes
    `... .mean() * UMOL_S_TO_G_D` — scalar arithmetic, which xarray applies to the values
    and not to `attrs`. The result is numerically right and still labelled `umol m-2 s-1`,
    and `strict` would now reject the *correct* function. That is why stamping is the
    default, and why **declarations on inputs are worth more than declarations on
    outputs**: the check that catches this in the pipeline proper is the one on the
    consumer, in stage 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What it catches — silently lost metadata.** `attrs` are dropped by xarray arithmetic
    unless you ask otherwise. We set `keep_attrs=True` at the top of this notebook; here
    is the same call without it, where the unit never reaches the check.
    """)
    return


@app.cell
def _(daily_carbon_new, nee_clean, warnings, xr):
    with (
        warnings.catch_warnings(record=True) as _caught,
        xr.set_options(keep_attrs=False),
    ):
        warnings.simplefilter("always")
        daily_carbon_new(nee_clean * 1.0)

    for _w in _caught:
        print(f"{_w.category.__name__}: {_w.message}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 4: weekly aggregation

    Weekly GPP, for comparison against the satellite product. The anchor is not a
    parameter: this function exists to produce week-ending-Sunday means, so `W-SUN` is
    written into the body and declared on the return. A stage should say what it *does*.
    """)
    return


@app.cell
def _(
    Annotated,
    Freq,
    Unit,
    daily_carbon_new,
    declare_freq,
    declare_units,
    fluxes,
    xr,
):
    @declare_units
    @declare_freq
    def weekly_mean_new(
        daily: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")]:
        """Mean daily flux within each week ending on a Sunday."""
        return daily.resample(time="W-SUN").mean()

    gpp_daily = daily_carbon_new(fluxes["gpp"])
    weekly_gpp = weekly_mean_new(gpp_daily)

    print(f"{weekly_gpp.sizes['time']} weeks, anchored {str(weekly_gpp.time.values[0])[:10]} (a Sunday)")
    return gpp_daily, weekly_gpp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 5: comparison against the satellite product

    The last stage compares modelled weekly GPP against the retrieval. Both arguments have
    to be on the same weekly grid for the comparison to mean anything, so both declare
    `Freq("W-SUN")` — and it is here, on the *inputs*, that the phase is enforced. The
    producer says what it does; the consumer says what it needs.
    """)
    return


@app.cell
def _(
    Annotated,
    Freq,
    TypedDict,
    Unit,
    declare_freq,
    declare_units,
    sat_gpp,
    weekly_gpp,
    xr,
):
    class Comparison(TypedDict):
        """Model-minus-observation summary statistics."""

        bias: float
        rmse: float

    @declare_units
    @declare_freq
    def compare_with_satellite_new(
        modelled: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],
        observed: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],
    ) -> Comparison:
        """Bias and RMSE of modelled weekly GPP against the satellite retrieval."""
        _diff = modelled - observed
        return {"bias": float(_diff.mean()), "rmse": float((_diff**2).mean() ** 0.5)}

    comparison = compare_with_satellite_new(weekly_gpp, sat_gpp)

    print(f"bias = {comparison['bias']:+.3f} g m-2 d-1, rmse = {comparison['rmse']:.3f} g m-2 d-1")
    return compare_with_satellite_new, comparison


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What it catches.** Both of the first pipeline's silent bugs, arriving at the one
    stage that checks its inputs: weekly GPP that was never converted out of moles, and
    weekly GPP on the wrong day of the week.
    """)
    return


@app.cell
def _(FreqError, compare_with_satellite_new, fluxes, gpp_daily, pint, sat_gpp):
    _never_converted = fluxes["gpp"].resample(time="D").mean().resample(time="W-SUN").mean()
    _wrong_weekday = gpp_daily.resample(time="W-WED").mean()

    try:
        compare_with_satellite_new(_never_converted, sat_gpp)
    except pint.DimensionalityError as exc:
        print(f"{type(exc).__name__}: {exc}")

    try:
        compare_with_satellite_new(_wrong_weekday, sat_gpp)
    except FreqError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Compare those two messages against the `nan` the first pipeline produced. Same bugs,
    same stage — but each error names the parameter, the expectation and the reality, and
    arrives before the arithmetic rather than after it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The corrected pipeline

    End to end, with every stage declaring what it needs.
    """)
    return


@app.cell
def _(comparison, daily_carbon_new, fluxes, gpp_daily, nee_daily, weekly_gpp):
    _reco_daily = daily_carbon_new(fluxes["reco"])

    print(f"annual NEE  = {float(nee_daily.sum()):+8.0f} g C m-2 yr-1   (negative = sink)")
    print(f"annual GPP  = {float(gpp_daily.sum()):+8.0f} g C m-2 yr-1")
    print(f"annual RECO = {float(_reco_daily.sum()):+8.0f} g C m-2 yr-1")
    print(f"\nweekly GPP: {weekly_gpp.sizes['time']} weeks from {str(weekly_gpp.time.values[0])[:10]} (a Sunday)")
    print(f"vs satellite: bias {comparison['bias']:+.2f}, rmse {comparison['rmse']:.2f} g C m-2 d-1")
    return


@app.cell
def _(weekly_gpp):
    weekly_gpp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    An annual NEE near **-490 g C m-2 yr-1**, against a GPP of ~1490 and respiration of
    ~1060 — a mid-latitude deciduous forest behaving like one. And a satellite comparison
    that actually compares like with like.

    Nothing here was clever. The four assumptions were already written down, in the
    docstrings, where they were worth nothing; moving them into the signature is the whole
    of the change. The annotations do look cluttered, but the clutter is dense, relevant
    information, and it is checked on every call.

    Two honest caveats. Declared outputs are *stamped* rather than checked by default, so
    a stage whose output nobody consumes under a declaration is a stage nobody is checking.
    And plenty of mistakes remain out of reach --- a flipped sign convention on NEE would
    have sailed through all of this. But the ones that *can* be checked mechanically,
    should be.
    """)
    return


if __name__ == "__main__":
    app.run()
