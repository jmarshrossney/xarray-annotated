# Getting Started

## Installation

`xarray-annotated` can be installed from PyPI using `pip`, or tools such as `uv` that wrap around it.

=== "uv (recommended)"

    ```sh
    uv add xarray-annotated
    ```

=== "pip"

    ```sh
    pip install xarray-annotated
    ```

CF-convention / UDUNITS unit strings (e.g. `"umol m-2 s-1"`) need the optional `cf` extra, which pulls in [cf-xarray](https://cf-xarray.readthedocs.io).
It also needs activating at startup --- see [Choosing a unit registry](guides/registry.md).

=== "uv (recommended)"

    ```sh
    uv add "xarray-annotated[cf]"
    ```

=== "pip"

    ```sh
    pip install "xarray-annotated[cf]"
    ```

The package requires Python `>=3.12`.
Python 3.12, 3.13 and 3.14 are tested in CI.


## Before and after

Vapour pressure deficit (VPD) measures how dry the air is, and is computed from air temperature and relative humidity.
Here is a function that averages it to daily values, written twice: once with its expectations in the docstring, and once with them declared in the signature.
The tabs on this page switch together, so you can compare the code and each result below.

=== "Plain xarray"

    ```python
    import numpy as np
    import xarray as xr


    def daily_vpd(tair: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
        """Daily mean vapour pressure deficit.

        Args:
            tair: Hourly air temperature in degC, with dims (time, site).
            rh: Hourly relative humidity in percent, with dims (time, site).

        Returns:
            Daily mean VPD in kPa, with dims (time, site).
        """
        es = 0.6108 * np.exp(17.27 * tair / (tair + 237.3))
        return (es * (1 - rh / 100)).resample(time="D").mean()
    ```

=== "With xarray-annotated"

    ```python
    from typing import Annotated

    import numpy as np
    import xarray as xr
    import xarray_annotated as xra
    from xarray_annotated import Dims, Freq, Unit


    @xra.enforce  # (1)!
    def daily_vpd(
        tair: Annotated[
            xr.DataArray,
            Dims("time", "site"),  # (2)!
            Unit("degC"),  # (3)!
            Freq("h"),  # (4)!
        ],
        rh: Annotated[xr.DataArray, Dims("time", "site"), Unit("percent"), Freq("h")],
    ) -> Annotated[xr.DataArray, Dims("time", "site"), Unit("kPa"), Freq("D")]:  # (5)!
        """Daily mean vapour pressure deficit."""
        es = 0.6108 * np.exp(17.27 * tair / (tair + 237.3))
        vpd = (es * (1 - rh / 100)).resample(time="D").mean()
        return vpd.assign_attrs(units="kPa")  # (6)!
    ```

    1.  `enforce` reads the declarations from the signature and checks them on every call.
    2.  These dimensions must be present, in any order, and no others.
    3.  The body always sees `degC`. A different unit such as `K` raises, even though it could be converted: see [Converting units](guides/conversion.md).
    4.  The `time` coordinate must be hourly.
    5.  The return value is checked too, and has to match.
    6.  This result has no unit of its own, so the body states it, and values are never inspected: see [Output units](guides/policy.md#units-outputs).

To try it, we need some data.
The values are constant at 20 °C and 60% humidity, so every correct result is the same: 0.935 kPa.

```python
import numpy as np
import pandas as pd
import xarray as xr


def hourly(value, units=None, freq="h"):
    time = pd.date_range("2024-07-01", "2024-07-03T23:00", freq=freq)
    return xr.DataArray(
        np.full((time.size, 2), value),
        dims=("time", "site"),
        coords={"time": time, "site": ["north", "south"]},
        attrs={"units": units} if units else {},
    )


tair = hourly(20.0, "degC")
rh = hourly(60.0, "percent")
```

### Temperature in kelvin

The plain function treats 293.15 as a Celsius temperature and returns a VPD of 3411 kPa, with no error.
The annotated function refuses the input, because `K` is not the declared `degC`.

=== "Plain xarray"

    ```python
    daily_vpd(hourly(293.15, "K"), rh).round(3)
    ```
    ```
    <xarray.DataArray (time: 3, site: 2)> Size: 48B
    array([[3411.424, 3411.424],
           [3411.424, 3411.424],
           [3411.424, 3411.424]])
    Coordinates:
      * time     (time) datetime64[us] 24B 2024-07-01 2024-07-02 2024-07-03
      * site     (site) <U5 40B 'north' 'south'
    ```

=== "With xarray-annotated"

    <!-- docs-test: raises ValueError -->
    ```python
    daily_vpd(hourly(293.15, "K"), rh).round(3)
    ```
    ```
    ValueError: [daily_vpd] input 'tair': unit 'K' differs from declared 'degC'.
      Converting would change the values; if that is correct for this quantity,
      pass .pint.quantify().pint.to('degC') or declare Unit('degC', convert=True).
    ```

    Temperature is safe to convert here, so convert it with pint and try again:

    ```python
    tair_k = hourly(293.15, "K")
    daily_vpd(tair_k.pint.quantify().pint.to("degC"), rh).round(3)
    ```
    ```
    <xarray.DataArray (time: 3, site: 2)> Size: 48B
    array([[0.935, 0.935],
           [0.935, 0.935],
           [0.935, 0.935]])
    Coordinates:
      * time     (time) datetime64[us] 24B 2024-07-01 2024-07-02 2024-07-03
      * site     (site) <U5 40B 'north' 'south'
    Attributes:
        units:    kPa
    ```

A declaration can also [opt in to converting](guides/conversion.md#opting-in) inputs itself, but it is off by default, because a compatible unit is not always the right one.

### Humidity with no units

Here `rh` is a fraction rather than a percentage, and nothing says so.
The plain function returns 2.324 instead of 0.935, and the VPD comes back labelled `degC`, because xarray kept the only `units` attribute it could find.
The annotated function refuses to guess.

=== "Plain xarray"

    ```python
    daily_vpd(tair, hourly(0.6)).round(3)
    ```
    ```
    <xarray.DataArray (time: 3, site: 2)> Size: 48B
    array([[2.324, 2.324],
           [2.324, 2.324],
           [2.324, 2.324]])
    Coordinates:
      * time     (time) datetime64[us] 24B 2024-07-01 2024-07-02 2024-07-03
      * site     (site) <U5 40B 'north' 'south'
    Attributes:
        units:    degC
    ```

=== "With xarray-annotated"

    <!-- docs-test: raises ValueError -->
    ```python
    daily_vpd(tair, hourly(0.6)).round(3)
    ```
    ```
    ValueError: [daily_vpd] input 'rh' has no 'units' attribute (declared 'percent').
      Attach the unit where the data enters the pipeline (`.assign_attrs(units=...)`),
      or switch checking off with `xra.enabled(False)`.
    ```

A unit that is dimensionally wrong, such as `tair` labelled `m`, raises `DimensionalityError` whatever the declaration says.

### Data at the wrong frequency

This 3-hourly data gives a plausible answer, and with constant values even the right one, but the function was written for hourly data.
Whether that matters depends on the analysis, and the annotated function makes it a decision rather than an accident.

=== "Plain xarray"

    ```python
    daily_vpd(hourly(20.0, "degC", freq="3h"), hourly(60.0, "percent", freq="3h")).round(3)
    ```
    ```
    <xarray.DataArray (time: 3, site: 2)> Size: 48B
    array([[0.935, 0.935],
           [0.935, 0.935],
           [0.935, 0.935]])
    Coordinates:
      * time     (time) datetime64[us] 24B 2024-07-01 2024-07-02 2024-07-03
      * site     (site) <U5 40B 'north' 'south'
    ```

=== "With xarray-annotated"

    <!-- docs-test: raises FreqError -->
    ```python
    daily_vpd(hourly(20.0, "degC", freq="3h"), hourly(60.0, "percent", freq="3h")).round(3)
    ```
    ```
    FreqError: [daily_vpd] 'tair' frequency mismatch on 'time': expected 'h', got '3h'
    ```

Whether a schema or frequency violation raises, warns or is ignored is [configurable](guides/policy.md).

## Shorter signatures

Spelling out every marker keeps a declaration in view where it applies, and is a good default.
When a signature gets long, or the same markers repeat across many functions, you can name the markers they share with a `type` statement and add the rest where they are used:

```python
from typing import Annotated

import numpy as np
import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Dims, Freq, Unit

type HourlyBySite = Annotated[xr.DataArray, Dims("time", "site"), Freq("h")]
type DailyBySite = Annotated[xr.DataArray, Dims("time", "site"), Freq("D")]


@xra.enforce
def daily_vpd(
    tair: Annotated[HourlyBySite, Unit("degC")],
    rh: Annotated[HourlyBySite, Unit("percent")],
) -> Annotated[DailyBySite, Unit("kPa")]:
    """Daily mean vapour pressure deficit."""
    es = 0.6108 * np.exp(17.27 * tair / (tair + 237.3))
    vpd = (es * (1 - rh / 100)).resample(time="D").mean()
    return vpd.assign_attrs(units="kPa")
```

This declares exactly what the longer version does.
See [Reusing a declaration](guides/declaring.md#reusing-a-declaration) for the trade-offs.

Next, see [Declaring properties](guides/declaring.md) for every marker, or the [worked example](example.md) for an end-to-end pipeline.
