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
    review because every line looks reasonable. 

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
    ## A flux processing pipeline

    The pipeline splits into five stages, which are python functions.
    In typical fashion, assumptions are stated in docstrings and comments.
    """)
    return


@app.cell
def _():
    def screen_quality(flux, qc):
        """Drop records not flagged good. `qc` is the flag array for flux."""
        return flux.where(qc == 0)

    def partition_fluxes(nee, tair, ppfd):
        """Partition NEE into GPP and respiration. Assumes tair in degC."""
        reco = 2.60 * 2.0 ** ((tair - 10.0) / 10.0)
        gpp = (reco - nee).where(ppfd > 5.0, 0.0)  # no photosynthesis in the dark
        return gpp, reco

    def daily_carbon(flux):
        """Half-hourly flux -> daily mean carbon flux."""
        return flux.resample(time="D").mean()

    def weekly_mean(daily):
        """Daily -> weekly mean. Assumes week ends on Wednesday."""
        return daily.resample(time="W-WED").mean()

    def compare_with_satellite(modelled, observed):
        """Bias and RMSE. Assumes both are weekly means on the same grid."""
        diff = modelled - observed
        return float(diff.mean()), float((diff**2).mean() ** 0.5)

    return (
        compare_with_satellite,
        daily_carbon,
        partition_fluxes,
        screen_quality,
        weekly_mean,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Chained together, they turn the logger output into the two products.
    """)
    return


@app.cell
def _(
    compare_with_satellite,
    daily_carbon,
    nee_raw,
    partition_fluxes,
    ppfd_raw,
    qc_raw,
    sat_gpp,
    screen_quality,
    tair_raw,
    weekly_mean,
):
    nee = screen_quality(nee_raw, qc_raw)
    gpp, _ = partition_fluxes(nee, tair_raw, ppfd_raw)

    # Product 1: the annual carbon budget.
    print(f"annual NEE = {float(daily_carbon(nee).sum()):+.0f} g C m-2 yr-1   (negative = sink)")

    # Product 2: weekly GPP, against the satellite retrieval.
    gpp_weekly = weekly_mean(daily_carbon(gpp))
    bias, rmse = compare_with_satellite(gpp_weekly, sat_gpp)
    print(
        f"\nweekly GPP = {gpp_weekly.sizes['time']} weeks from "
        f"{str(gpp_weekly.time.values[0])[:10]}, mean {float(gpp_weekly.mean()):.3g}"
    )
    print(f"vs satellite: bias {bias:+.2f}, rmse {rmse:.2f} g C m-2 d-1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Hmm. The NEE looks plausible, but something has clearly gone wrong with GPP (it's enormous)
    and the statistics are somehow `nan`.

    No errors or warnings were raised. Time to go on a bug hunt I guess.

    ### Some time later...

    Bugs located. Three of the documented assumptions were false.

    `Assumes tair in degC.`
    : It arrives in kelvin, so the Q10 exponent is `(290-10)/10 = 28`
      rather than `(17-10)/10 = 0.7`. A mean weekly GPP of `2.6e8` where
      4 would be respectable — easy to debug.

    `Assumes week ends on Wednesday.`
    : The satellite product ends its weeks on Sunday.

    `Assumes both are weekly means on the same grid.`
    : They are not. xarray aligns on the time coordinate, finds
      that Wednesdays and Sundays never coincide, and every statistic comes back `nan`.

    Two of those are one-line fixes — convert the temperature at the call site, and anchor
    the weekly resample to Sunday — and the third goes away with them.
    """)
    return


@app.cell
def _(
    compare_with_satellite,
    daily_carbon,
    nee_raw,
    partition_fluxes,
    ppfd_raw,
    qc_raw,
    sat_gpp,
    screen_quality,
    tair_raw,
):
    def weekly_mean_sunday(daily):
        """Daily -> weekly mean. Week ends on Sunday, like the satellite product."""
        return daily.resample(time="W-SUN").mean()

    _tair_degc = tair_raw - 273.15

    _nee = screen_quality(nee_raw, qc_raw)
    _gpp, _ = partition_fluxes(_nee, _tair_degc, ppfd_raw)

    print(f"annual NEE = {float(daily_carbon(_nee).sum()):+.0f} g C m-2 yr-1   (negative = sink)")

    _gpp_weekly = weekly_mean_sunday(daily_carbon(_gpp))
    _bias, _rmse = compare_with_satellite(_gpp_weekly, sat_gpp)
    print(
        f"\nweekly GPP = {_gpp_weekly.sizes['time']} weeks from "
        f"{str(_gpp_weekly.time.values[0])[:10]}, mean {float(_gpp_weekly.mean()):.3g}"
    )
    print(f"vs satellite: bias {_bias:+.2f}, rmse {_rmse:.2f} g C m-2 d-1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""endingendingending
    A carbon sink of a few hundred grams per square metre per year, weekly GPP on the right
    grid averaging `3.88`, and a satellite comparison whose bias is a fraction of the rmse.
    Every number is the right order of magnitude and the right sign.

    Three bugs found, three bugs fixed, no errors, no warnings. Ship it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The same pipeline with declarations

    Here are the same five stages, with each assumption moved out of the docstring and into
    the signature, where it is actually checkable at run-time. One of them returns more than
    one array, so a `TypedDict` gives the declarations somewhere to hang; the comparison
    returns a second one for its two statistics.

    Nothing else changes: same models, same arithmetic, same bodies as the code we just
    finished debugging. The declarations only write down what was already believed to be
    true. We then walk through the stages one at a time to see what each one buys.
    """)
    return


@app.cell
def _(
    Annotated,
    Coords,
    Dims,
    Dtype,
    Freq,
    TypedDict,
    Unit,
    declare_freq,
    declare_schema,
    declare_units,
    xr,
):
    class Partitioned(TypedDict):
        """Gross fluxes, both sign-positive."""

        gpp: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]
        reco: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]

    class Comparison(TypedDict):
        """Model-minus-observation summary statistics."""

        bias: float
        rmse: float

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

    @declare_units
    @declare_freq
    def daily_carbon_new(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("30min")],
    ) -> Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("D")]:
        """Half-hourly flux -> daily mean carbon flux."""
        return flux.resample(time="D").mean()

    @declare_units
    @declare_freq
    def weekly_mean_new(
        daily: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("D")],
    ) -> Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("W-SUN")]:
        """Mean daily flux within each week ending on a Sunday."""
        return daily.resample(time="W-SUN").mean()

    @declare_units
    @declare_freq
    def compare_with_satellite_new(
        modelled: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],
        observed: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],
    ) -> Comparison:
        """Bias and RMSE of modelled weekly GPP against the satellite retrieval."""
        _diff = modelled - observed
        return {"bias": float(_diff.mean()), "rmse": float((_diff**2).mean() ** 0.5)}

    return (
        compare_with_satellite_new,
        daily_carbon_new,
        partition_fluxes_new,
        screen_quality_new,
        weekly_mean_new,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 1: quality screening

    A half-hourly series with a real `time` coordinate, and an `int8` flag array alongside
    it. Returns `float64`, because gaps become NaN and an integer array cannot hold NaN.
    """)
    return


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
    the boundary rather than blowing up inside the body. The two outputs needed a named
    structure to hang declarations on, so the tuple became the `Partitioned` `TypedDict`,
    validated per field.
    """)
    return


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

    Half-hourly to daily, so `Freq("30min")` in and `Freq("D")` out. The body resamples and
    does nothing else, so the unit it was handed is the unit it returns: `umol m-2 s-1`
    both sides. The declarations describe the function exactly.
    """)
    return


@app.cell
def _(daily_carbon_new, nee_clean):
    _nee_daily = daily_carbon_new(nee_clean)
    print(f"{_nee_daily.sizes['time']} daily means, labelled {_nee_daily.units}")
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

    Averaging does not change a unit either, so this one is `umol m-2 s-1` in and out too.
    """)
    return


@app.cell
def _(daily_carbon_new, fluxes, weekly_mean_new):
    _gpp_daily_molar = daily_carbon_new(fluxes["gpp"])
    weekly_gpp_molar = weekly_mean_new(_gpp_daily_molar)

    print(f"{weekly_gpp_molar.sizes['time']} weeks, anchored {str(weekly_gpp_molar.time.values[0])[:10]} (a Sunday)")
    print(f"mean weekly GPP = {float(weekly_gpp_molar.mean()):.2f} {weekly_gpp_molar.units}")
    return (weekly_gpp_molar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Stage 5: comparison against the satellite product

    The last stage compares modelled weekly GPP against the retrieval. Both arguments have
    to be on the same weekly grid for the comparison to mean anything, so both declare
    `Freq("W-SUN")` — and it is here, on the *inputs*, that the phase is enforced. The
    producer says what it does; the consumer says what it needs.

    The units are not ours to choose either. The satellite product is published as
    `g m-2 d-1`; that is a fact about someone else's file, and a comparison is only
    meaningful if both sides are in it. So both parameters declare it.
    """)
    return


@app.cell
def _(compare_with_satellite_new, pint, sat_gpp, weekly_gpp_molar):
    try:
        compare_with_satellite_new(weekly_gpp_molar, sat_gpp)
    except pint.DimensionalityError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The bug nobody was hunting

    A fourth bug, and it was never on the list. Nothing converts moles to grams — not in
    this pipeline, and not in the one we spent the afternoon debugging. `daily_carbon`
    resamples and does nothing else. The conversion needs the molar mass of carbon, so no
    library can supply it unasked, and nobody asked.

    Every stage was *internally* consistent: a molar flux went in, a molar flux came out,
    and each stage's declarations described it correctly. That is why no earlier check
    fired, and it is why the bug is realistic. It only becomes visible where the pipeline
    meets a number whose units it does not get to choose.

    Notice what the failure did *not* require. The two signatures sit next to each other in
    the definitions above, one stage feeding the next:

    ```python
    def weekly_mean_new(...) -> Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("W-SUN")]
    def compare_with_satellite_new(modelled: Annotated[xr.DataArray, Unit("g m-2 d-1"), ...])
    ```

    One produces `umol m-2 s-1`; the next consumes `g m-2 d-1`. That is a contradiction you
    can *read*, in a diff, on a screen, without a runtime, without data, without running
    anything at all. The declarations turned a question about program behaviour into a
    question about two lines of text. Running the pipeline merely confirmed it.

    ### What it would have cost

    The bug hunt was driven by numbers that looked wrong. These numbers do not:

    | | molar (what we shipped) | mass (correct) |
    |---|---|---|
    | annual NEE, g C m-2 yr-1 | `-474` | `-491` |
    | mean weekly GPP, g m-2 d-1 | `3.88` | `4.03` |
    | bias vs satellite | `-0.36` | `-0.21` |

    The factor is `1e-6 × 12.011 × 86400 = 1.038`, so **every product is 3.6% low** — well
    inside what you would accept from a flux tower, and far inside the 8% spread of the
    retrieval it gets validated against. The plain pipeline's comparison, the one step whose
    entire job is to catch this sort of thing, reported a bias of `-0.36` against an rmse of
    `0.58` and raised no objection.

    And the annual carbon budget would never have been caught at all. It is compared against
    nothing; it is simply reported. The only reason the error surfaced anywhere is that one
    of the two products happens to be validated against somebody else's data.

    ### Fixing it

    Convert where the daily budget is formed. The declarations then tell you exactly how far
    the change reaches: stage 4 declared molar, so stage 4 has to move too — same body, new
    signature.
    """)
    return


@app.cell
def _(Annotated, Freq, UMOL_S_TO_G_D, Unit, declare_freq, declare_units, xr):
    @declare_units
    @declare_freq
    def daily_carbon_grams(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("30min")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")]:
        """Mean half-hourly molar flux -> daily carbon mass flux."""
        return flux.resample(time="D").mean() * UMOL_S_TO_G_D

    @declare_units
    @declare_freq
    def weekly_mean_grams(
        daily: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")]:
        """Mean daily flux within each week ending on a Sunday."""
        return daily.resample(time="W-SUN").mean()

    return daily_carbon_grams, weekly_mean_grams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **A caveat worth understanding.** By default a declared output is *stamped*, not
    checked: whatever the body returns comes back labelled `g m-2 d-1`. That is what makes
    the fix work. `* UMOL_S_TO_G_D` is scalar arithmetic, which xarray applies to the values
    and not to `attrs`, so the result is numerically right and still labelled
    `umol m-2 s-1` until the declaration relabels it.

    The `on_output="strict"` policy verifies the label instead of overwriting it — and would
    therefore reject the function we just wrote, whose `attrs` are stale by construction:
    """)
    return


@app.cell
def _(Annotated, Freq, UMOL_S_TO_G_D, Unit, declare_freq, declare_units, nee_clean, pint, xr):
    @declare_units(on_output="strict")
    @declare_freq
    def daily_carbon_strict(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1"), Freq("30min")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("D")]:
        """The corrected body — numerically right, and rejected anyway."""
        return flux.resample(time="D").mean() * UMOL_S_TO_G_D

    try:
        daily_carbon_strict(nee_clean)
    except pint.DimensionalityError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    So the choice is per stage: `strict` where the body passes units through untouched — it
    would have caught a stage that *claimed* a conversion it never made — and stamping where
    the body does its own unit arithmetic, as here. The alternative is to clear the stale
    `attrs` in the body and keep `strict`.

    With the conversion in place, the chain runs:
    """)
    return


@app.cell
def _(daily_carbon_grams, fluxes, weekly_mean_grams):
    gpp_daily = daily_carbon_grams(fluxes["gpp"])
    weekly_gpp = weekly_mean_grams(gpp_daily)

    print(f"{weekly_gpp.sizes['time']} weeks, anchored {str(weekly_gpp.time.values[0])[:10]} (a Sunday)")
    print(f"mean weekly GPP = {float(weekly_gpp.mean()):.2f} {weekly_gpp.units}")
    return gpp_daily, weekly_gpp


@app.cell
def _(compare_with_satellite_new, sat_gpp, weekly_gpp):
    comparison = compare_with_satellite_new(weekly_gpp, sat_gpp)

    print(f"bias = {comparison['bias']:+.3f} g m-2 d-1, rmse = {comparison['rmse']:.3f} g m-2 d-1")
    return (comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **What it catches.** The bug that produced the first pipeline's `nan` statistics:
    weekly means on the wrong day of the week, silently misaligned by xarray.
    """)
    return


@app.cell
def _(FreqError, compare_with_satellite_new, gpp_daily, sat_gpp):
    _wrong_weekday = gpp_daily.resample(time="W-WED").mean()

    try:
        compare_with_satellite_new(_wrong_weekday, sat_gpp)
    except FreqError as exc:
        print(f"{type(exc).__name__}: {exc}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Compare that against the `nan` the first pipeline produced. Same bug, same stage — but
    the error names the parameter, the expectation and the reality, and arrives before the
    arithmetic rather than after it.
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
def _(comparison, daily_carbon_grams, fluxes, gpp_daily, nee_clean, weekly_gpp):
    _nee_daily = daily_carbon_grams(nee_clean)
    _reco_daily = daily_carbon_grams(fluxes["reco"])

    print(f"annual NEE  = {float(_nee_daily.sum()):+8.0f} g C m-2 yr-1   (negative = sink)")
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

    Nothing here was clever. Three of the four assumptions were already written down in the
    docstrings, where they were worth nothing; moving them into the signature is the whole
    of the change. The annotations do look cluttered, but the clutter is dense, relevant
    information, and it is checked on every call.

    The fourth is the one that matters. It was never written down anywhere, in any form. A
    deliberate bug hunt with the numbers in front of us did not find it, because a 3.6%
    error in a flux budget does not look like an error. What found it was two declarations
    disagreeing at the point where the pipeline met data whose units it did not control —
    and once they are written down, that disagreement is legible on the page. The check at
    run-time confirmed what the signatures already said.

    That is the case for declaring your expectations: not the bugs you would have caught
    anyway, but the ones with no symptom to notice, made visible by writing down what each
    stage takes and returns.

    Two honest caveats. Declared outputs are *stamped* rather than checked by default, so
    a stage whose output nobody consumes under a declaration is a stage nobody is checking.
    And plenty of mistakes remain out of reach --- a flipped sign convention on NEE would
    have sailed through all of this. But the ones that *can* be checked mechanically,
    should be.
    """)
    return


if __name__ == "__main__":
    app.run()
