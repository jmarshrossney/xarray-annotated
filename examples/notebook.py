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
app = marimo.App(app_title="xarray-annotated: a worked example")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # A worked example

    This notebook processes a year of synthetic eddy-covariance flux data into two products:

    1. an **annual carbon budget** — how much carbon the site took up over the year;
    2. **weekly GPP**, validated against a satellite retrieval.

    It is a realistic pipeline with a realistic set of bugs --- some obvious, others that
    could plausibly sneak through because the outputs look reasonable.

    This notebook lives in the repository at
    [`examples/notebook.py`](https://github.com/jmarshrossney/xarray-annotated/tree/main/examples).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup

    We imagine a temperate deciduous forest site at roughly 52°N, logging data every
    **30 minutes** for one year. The logger gives us four series:

    | Variable | Unit | Note |
    |---|---|---|
    | `nee_raw` | `umol m-2 s-1` | net ecosystem exchange; **negative means uptake** |
    | `tair` | `K` | air temperature, as stored |
    | `ppfd` | `umol m-2 s-1` | photosynthetic photon flux density |
    | `qc` | — | quality flag, `int8`: 0 good, 1 moderate, 2 bad |

    We also have a satellite GPP product to validate against, on a coarser grid:

    | Variable | Unit | Note |
    |---|---|---|
    | `sat_gpp` | `g m-2 d-1` | weekly mean GPP, **week-ending Sunday** |

    Throughout, a mass flux written `g m-2 d-1` means grams *of carbon*: pint has no way
    to say "grams of carbon" rather than "grams", so the species lives in the prose.
    """)
    return


@app.cell
def _():
    import warnings
    from typing import Annotated, TypedDict

    import numpy as np
    import xarray as xr

    from xarray_annotated.schema import (
        Coords,
        Dims,
        Dtype,
        declare_schema,
    )
    from xarray_annotated.temporal import Freq, declare_freq
    from xarray_annotated.units import Unit, declare_units, use_cf_units

    # Catch an annoying warning from cf-xarray when matplotlib not available.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Import(s) unavailable to set up matplotlib support")

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
        TypedDict,
        UMOL_S_TO_G_D,
        Unit,
        declare_freq,
        declare_schema,
        declare_units,
        np,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The data is synthetic but physically plausible: a real diurnal and seasonal
    cycle, a Q10 respiration response, and a saturating light response.
    """)
    return


@app.cell(hide_code=True)
def _(UMOL_S_TO_G_D, np, xr):
    def get_data():
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
        tair = _series(_tair_c + 273.15, "K")
        ppfd = _series(_ppfd, "umol m-2 s-1")
        qc = _series(_qc, "1", dtype="int8")

        # The satellite product: weekly mean GPP, week-ending Sunday, as a mass flux.
        # Built from the true half-hourly GPP, with a retrieval error on top.
        _gpp_daily = _series(_gpp, "umol m-2 s-1").resample(time="D").mean() * UMOL_S_TO_G_D
        _gpp_weekly = _gpp_daily.resample(time="W-SUN").mean()
        sat_gpp = (_gpp_weekly * (1.0 + 0.08 * _rng.normal(size=_gpp_weekly.sizes["time"]))).assign_attrs(
            units="g m-2 d-1"
        )
        return nee_raw, ppfd, qc, sat_gpp, tair

    return (get_data,)


@app.cell
def _(get_data, xr):
    # Assume that get_data is defined elsewhere
    nee_raw, ppfd, qc, sat_gpp, tair = get_data()
    print(f"{nee_raw.sizes['time']} half-hourly records, {xr.infer_freq(nee_raw.time)}")
    print(f"flagged bad or moderate: {int((qc > 0).sum())}")
    print(f"satellite GPP: {sat_gpp.sizes['time']} weekly means, {xr.infer_freq(sat_gpp.time)}")
    return nee_raw, ppfd, qc, sat_gpp, tair


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A simple flux processing pipeline

    A simple flux processing pipeline for an eddy covariance site might look
    something like this.

    ```mermaid
    flowchart TD
        NEER["nee_raw<br/>umol m-2 s-1 · 30min"] --> SQ(["screen_quality"])
        QC["qc<br/>int8 · 30min"] --> SQ
        SQ --> NEE["nee<br/>umol m-2 s-1 · 30min"]

        NEE --> PF(["partition_fluxes"])
        TAIR["tair<br/>K · 30min"] --> PF
        PPFD["ppfd<br/>umol m-2 s-1 · 30min"] --> PF
        PF --> GPP["gpp<br/>umol m-2 s-1 · 30min"]
        PF --> RECO["reco<br/>umol m-2 s-1 · 30min"]

        NEE --> TMN(["to_mass_flux"]) --> NEEM["nee<br/>g m-2 d-1 · 30min"]
        GPP --> TMG(["to_mass_flux"]) --> GPPM["gpp<br/>g m-2 d-1 · 30min"]

        NEEM --> DCN(["daily_mean"]) --> NEED["nee daily<br/>g m-2 d-1 · D"]
        GPPM --> DCG(["daily_mean"]) --> GPPD["gpp daily<br/>g m-2 d-1 · D"]

        NEED --> SUM([".sum()"]) --> BUD["annual budget<br/>g C m-2 yr-1"]

        GPPD --> WM(["weekly_mean"]) --> GPPW["gpp weekly<br/>g m-2 d-1 · W-SUN"]
        GPPW --> CMP(["compare_with_satellite"])
        SAT["sat_gpp<br/>g m-2 d-1 · W-SUN"] --> CMP
        CMP --> BIAS["bias<br/>g m-2 d-1"]
        CMP --> RMSE["rmse<br/>g m-2 d-1"]
    ```

    Rounded boxes are functions, square boxes are data.

    Below, this pipeline is implemented using Python functions.
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

    def daily_mean(flux):
        """Half-hourly -> daily mean."""
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
        daily_mean,
        partition_fluxes,
        screen_quality,
        weekly_mean,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Some unconvincing results

    We run the pipeline to produce the two products.
    """)
    return


@app.cell
def _(
    compare_with_satellite,
    daily_mean,
    partition_fluxes,
    sat_gpp,
    screen_quality,
    weekly_mean,
):
    def run_pipeline(nee_raw, qc, tair, ppfd):
        nee = screen_quality(nee_raw, qc)
        gpp, _ = partition_fluxes(nee, tair, ppfd)

        # Product 1: the annual carbon budget.
        nee_annual = float(daily_mean(nee).sum())

        # Product 2: weekly GPP, against the satellite retrieval.
        gpp_weekly = weekly_mean(daily_mean(gpp))
        bias, rmse = compare_with_satellite(gpp_weekly, sat_gpp)

        return {
            "nee_annual": nee_annual,
            "gpp_weekly": gpp_weekly,
            "comparison": {"bias": bias, "rmse": rmse},
        }

    return (run_pipeline,)


@app.cell(hide_code=True)
def _(mo, nee_raw, ppfd, qc, run_pipeline, tair):
    _results = run_pipeline(nee_raw, qc, tair, ppfd)
    _gpp = _results["gpp_weekly"]
    _cmp = _results["comparison"]

    mo.md(f"""
    This produces:

    | Product | Value |
    |---|---|
    | Annual NEE | **{_results["nee_annual"]:+.0f}** g C m^-2^ yr^-1^ (negative = sink) |
    | Weekly GPP | {_gpp.sizes["time"]} weeks from {str(_gpp.time.values[0])[:10]}, mean **{float(_gpp.mean()):.3g}** umol m^-2^ s^-1^ |
    | Satellite comparison | bias **{_cmp["bias"]:+.2f}**, rmse **{_cmp["rmse"]:.2f}** g C m^-2^ d^-1^ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Hmm. The NEE looks plausible, but something has clearly gone wrong with GPP (it's enormous)
    and the statistics are somehow `nan`.

    No errors or warnings were raised. Time to go on a bug hunt I guess.

    **Some time later...**

    Bugs located. Three of the documented assumptions were false.

    `Assumes tair in degC.`
    : It arrives in Kelvin, so the Q10 exponent is `(290-10)/10 = 28`
      rather than `(17-10)/10 = 0.7`. A mean weekly GPP of `2.6e8` where
      4 would be respectable --- easy to debug.

    `Assumes week ends on Wednesday.`
    : The satellite product ends its weeks on Sunday --- easy.

    `Assumes both are weekly means on the same grid.`
    : They are not. xarray aligns on the time coordinate, finds
      that Wednesdays and Sundays never coincide, and every statistic comes back `nan`.
      Another easy one.

    These are one-line fixes: convert the temperature at the call site, and anchor
    the weekly resample to Sunday.
    """)
    return


@app.cell
def _(
    compare_with_satellite,
    daily_mean,
    partition_fluxes,
    sat_gpp,
    screen_quality,
):
    def weekly_mean_sunday(daily):
        """Daily -> weekly mean. Week ends on Sunday, like the satellite product."""
        return daily.resample(time="W-SUN").mean()

    def run_pipeline_patched(nee_raw, qc, tair, ppfd):
        tair_degc = tair - 273.15  # the Q10 model wants degC, not kelvin

        nee = screen_quality(nee_raw, qc)
        gpp, _ = partition_fluxes(nee, tair_degc, ppfd)

        # Product 1: the annual carbon budget.
        nee_annual = float(daily_mean(nee).sum())

        # Product 2: weekly GPP, now on the satellite's weekly grid.
        gpp_weekly = weekly_mean_sunday(daily_mean(gpp))
        bias, rmse = compare_with_satellite(gpp_weekly, sat_gpp)

        return {
            "nee_annual": nee_annual,
            "gpp_weekly": gpp_weekly,
            "comparison": {"bias": bias, "rmse": rmse},
        }

    return (run_pipeline_patched,)


@app.cell(hide_code=True)
def _(mo, nee_raw, ppfd, qc, run_pipeline_patched, tair):
    _results = run_pipeline_patched(nee_raw, qc, tair, ppfd)
    _gpp = _results["gpp_weekly"]
    _cmp = _results["comparison"]

    mo.md(f"""
    | Product | Value |
    |---|---|
    | Annual NEE | **{_results["nee_annual"]:+.0f}** g C m^-2^ yr^-1^ (negative = sink) |
    | Weekly GPP | {_gpp.sizes["time"]} weeks from {str(_gpp.time.values[0])[:10]}, mean **{float(_gpp.mean()):.3g}** umol m^-2^ s^-1^ |
    | Satellite comparison | bias **{_cmp["bias"]:+.2f}**, rmse **{_cmp["rmse"]:.2f}** g C m^-2^ d^-1^ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    A carbon sink of a few hundred grams per square metre per year, weekly GPP on the right
    grid averaging `3.88`, and a satellite comparison whose bias is a fraction of the rmse.
    Numbers are the right order of magnitude and the right sign.
    No errors or warnings.
    LGTM?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The same pipeline with declarations

    Here is the same pipeline with each assumption moved out of the docstring and into
    the signature, where it is actually checkable at run-time.

    !!! tip
        Hover the :lucide-circle-plus: markers for an explanation of the corresponding annotation.
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
    @declare_units
    @declare_schema
    @declare_freq
    def screen_quality_declared(
        flux: Annotated[
            xr.DataArray,
            Dims("time"),
            Coords("time"),  # (1)
            Unit("umol m-2 s-1"),
            Freq("30min"),  # (2)
        ],
        qc: Annotated[xr.DataArray, Dims("time"), Dtype("int8")],  # (3)
    ) -> Annotated[
        xr.DataArray,
        Dims("time"),
        Dtype("float64"),  # (4)
        Unit("umol m-2 s-1"),
        Freq("30min"),
    ]:
        """Drop records not flagged as good quality."""
        return flux.where(qc == 0)

    class Partitioned(TypedDict):  # (5)
        """Gross fluxes, both sign-positive."""

        gpp: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]
        reco: Annotated[xr.DataArray, Dims("time"), Unit("umol m-2 s-1")]

    @declare_units
    def partition_fluxes_declared(
        nee: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
        tair: Annotated[xr.DataArray, Unit("degC")],  # (6)
        ppfd: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
    ) -> Partitioned:
        """Partition NEE into GPP and respiration via a Q10 model."""
        reco = 2.60 * 2.0 ** ((tair - 10.0) / 10.0)
        gpp = (reco - nee).where(ppfd > 5.0, 0.0)  # no photosynthesis in the dark
        return {"gpp": gpp, "reco": reco}

    @declare_freq
    def daily_mean_declared(
        flux: Annotated[xr.DataArray, Freq("30min")],
    ) -> Annotated[xr.DataArray, Freq("D")]:  # (7)
        """Mean half-hourly flux within each day."""
        return flux.resample(time="D").mean()

    @declare_freq
    def weekly_mean_declared(
        daily: Annotated[xr.DataArray, Freq("D")],
    ) -> Annotated[xr.DataArray, Freq("W-SUN")]:  # (8)
        """Mean daily flux within each week ending on a Sunday."""
        return daily.resample(time="W-SUN").mean()

    @declare_units
    @declare_freq
    def compare_with_satellite_declared(
        modelled: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],  # (9)
        observed: Annotated[xr.DataArray, Unit("g m-2 d-1"), Freq("W-SUN")],  # (10)
    ) -> dict[str, float]:
        """Bias and RMSE of modelled weekly GPP against the satellite retrieval."""
        _diff = modelled - observed
        return {"bias": float(_diff.mean()), "rmse": float((_diff**2).mean() ** 0.5)}

    return (
        compare_with_satellite_declared,
        daily_mean_declared,
        partition_fluxes_declared,
        screen_quality_declared,
        weekly_mean_declared,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    1. `Dims` is about shape; `Coords` is about labels. An export can have a `time`
       dimension and no `time` coordinate — common when a file is read with the index
       column mislabelled, and otherwise survives until something calls `.resample`,
       several stages away.
    2. Half-hourly data, checked against the actual time coordinate. A site that logs
       hourly produces a perfectly good file, for a different pipeline.
    3. The quality flag is `int8`. A float flag has usually already been through
       arithmetic that turned missing records into NaN, and `qc == 0` then quietly
       screens nothing.
    4. `float64` out, not `int8`: gaps become NaN, and an integer array cannot hold NaN.
    5. Two outputs need somewhere to hang declarations, so the tuple became a `TypedDict`,
       validated field by field.
    6. The docstring's *"assumes tair in degC"*, made real. Kelvin is now converted at the
       boundary instead of blowing up the Q10 exponent inside the body.
    7. Aggregation changes the frequency and nothing else, so this stage declares a
       frequency and says nothing at all about units — it means whatever it is handed, and
       the silence is the statement.
    8. Same, one scale up. And the anchor is not a parameter: this function exists to
       produce week-ending-Sunday means, so `W-SUN` is written into the body *and* declared
       on the return. A stage should say what it does.
    9. The satellite product is published as `g m-2 d-1`. That is a fact about someone
       else's file, not a choice, and a comparison is only meaningful if both sides are in
       it — so both parameters declare it.
    10. Both arguments must also be on the same weekly grid, enforced here on the *inputs*.
        The producer says what it does; the consumer says what it needs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Two of the three bugs from the bug hunt cannot recur:

    - `tair` can be in either Kelvin or degrees C --- either way it is converted at the
      boundary (by `pint`) thanks to the `Unit("degC")` annotation.
    - The weekly anchor is declared on both the producer and the consumer.

    Beyond those, the checks reject inputs that never belonged in this pipeline
    at all, such as an export with a `time` dimension but no `time` coordinate,
    data from an hourly site.

    Let's run it.
    """)
    return


@app.cell
def _(
    compare_with_satellite_declared,
    daily_mean_declared,
    partition_fluxes_declared,
    sat_gpp,
    screen_quality_declared,
    weekly_mean_declared,
):
    def run_pipeline_declared(nee_raw, qc, tair, ppfd):
        nee = screen_quality_declared(nee_raw, qc)
        fluxes = partition_fluxes_declared(nee, tair, ppfd)

        # Product 1: the annual carbon budget.
        nee_annual = float(daily_mean_declared(nee).sum())

        # Product 2: weekly GPP, against the satellite retrieval.
        gpp_weekly = weekly_mean_declared(daily_mean_declared(fluxes["gpp"]))
        comparison = compare_with_satellite_declared(gpp_weekly, sat_gpp)

        return {
            "nee_annual": nee_annual,
            "gpp_weekly": gpp_weekly,
            "comparison": comparison,
        }

    return (run_pipeline_declared,)


@app.cell
def _(nee_raw, ppfd, qc, run_pipeline_declared, tair):
    run_pipeline_declared(nee_raw, qc, tair, ppfd)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The bug we missed

    The pipeline crashes with a `pint.errors.DimensionalityError` since it cannot
    reconcile the declared units, `g m-2 d-1`, with the `units` attribute stored in
    the input `DataArray`, `umol m-2 s-1`.

    Looking back at the diagram, it's clear what the problem is. `to_mass_flux` is right
    there, twice. It was in the design from the beginning.

    Sure, we probably should have written *"Assumes units of g C m^-2^ d^-1^"* in the
    original docstring for `compare_with_satellite`, but there's no guarantee we would
    have noticed, especially since the numbers came out looking highly plausible.

    Let's add the missing function and run the pipeline (hopefully) one final time.
    """)
    return


@app.cell
def _(Annotated, UMOL_S_TO_G_D, Unit, declare_units, xr):
    @declare_units
    def to_mass_flux(
        flux: Annotated[xr.DataArray, Unit("umol m-2 s-1")],
    ) -> Annotated[xr.DataArray, Unit("g m-2 d-1")]:
        """umol CO2 m-2 s-1 -> g C m-2 d-1, via the molar mass of carbon."""
        return flux * UMOL_S_TO_G_D

    return (to_mass_flux,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The corrected pipeline

    With the missing stage inserted, end to end, every stage declaring what it needs:
    """)
    return


@app.cell
def _(
    compare_with_satellite_declared,
    daily_mean_declared,
    partition_fluxes_declared,
    sat_gpp,
    screen_quality_declared,
    to_mass_flux,
    weekly_mean_declared,
):
    def run_pipeline_final(nee_raw, qc, tair, ppfd):
        nee = screen_quality_declared(nee_raw, qc)
        fluxes = partition_fluxes_declared(nee, tair, ppfd)

        nee_daily = daily_mean_declared(to_mass_flux(nee))
        gpp_daily = daily_mean_declared(to_mass_flux(fluxes["gpp"]))
        reco_daily = daily_mean_declared(to_mass_flux(fluxes["reco"]))

        # Product 1: the annual carbon budget.
        nee_annual = float(nee_daily.sum())

        # Product 2: weekly GPP, against the satellite retrieval.
        gpp_weekly = weekly_mean_declared(gpp_daily)
        comparison = compare_with_satellite_declared(gpp_weekly, sat_gpp)

        return {
            "nee_annual": nee_annual,
            "gpp_annual": float(gpp_daily.sum()),
            "reco_annual": float(reco_daily.sum()),
            "gpp_weekly": gpp_weekly,
            "comparison": comparison,
        }

    return (run_pipeline_final,)


@app.cell
def _(nee_raw, ppfd, qc, run_pipeline_final, tair):
    results = run_pipeline_final(nee_raw, qc, tair, ppfd)
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    _weekly = results["gpp_weekly"]
    _cmp = results["comparison"]

    mo.md(f"""
    | Product | Value |
    |---|---|
    | Annual NEE | **{results["nee_annual"]:+.0f}** g C m^-2^ yr^-1^ (negative = sink) |
    | Annual GPP | **{results["gpp_annual"]:+.0f}** g C m^-2^ yr^-1^ |
    | Annual RECO | **{results["reco_annual"]:+.0f}** g C m^-2^ yr^-1^ |
    | Weekly GPP | {_weekly.sizes["time"]} weeks from {str(_weekly.time.values[0])[:10]} (a Sunday), mean **{float(_weekly.mean()):.2f}** g C m^-2^ d^-1^ |
    | Satellite comparison | bias **{_cmp["bias"]:+.2f}**, rmse **{_cmp["rmse"]:.2f}** g C m^-2^ d^-1^ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    An annual NEE near **-490 g C m^-2^ yr^-1^** against a GPP of ~1490 and respiration
    of ~1060 is reasonable for a mid-latitude deciduous forest. Importantly, the satellite
    comparison finally compares like with like.

    Nothing here was clever. The original diagram was entirely correct, and 3/4 bugs were
    correctly called in docstrings. Without an enforcement mechanism, though, they were
    able to slip through.

    The fourth is the one that matters most since a 3.6% error in a flux budget does not
    obviously look like an error.

    Although this is quite powerful, there are a few gaps and sharp edges that are worth
    understanding before you lean on them. See the guides for more info.

    - [Declaring properties](guides/declaring.md)
    - [Configuring validation](guides/policy.md)
    - [Troubleshooting](guides/troubleshooting.md)
    """)
    return


if __name__ == "__main__":
    app.run()
