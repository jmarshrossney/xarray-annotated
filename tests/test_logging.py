"""The audit trail: every non-trivial action is logged.

INFO marks a failed or untestable check let through (anything under
``ignore``); DEBUG marks chosen or routine actions, including every conversion,
since conversion is always opted into.
"""

import logging
from typing import Annotated

import numpy as np
import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Dims, Freq, Unit, schema, temporal, units

UNITS = "xarray_annotated.units"
SCHEMA = "xarray_annotated.schema"
TEMPORAL = "xarray_annotated.temporal"


def _da(unit: str | None = "hPa") -> xr.DataArray:
    attrs = {} if unit is None else {"units": unit}
    return xr.DataArray([1.0], dims="x", attrs=attrs)


def _records(caplog, logger):
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == logger]


@pytest.fixture
def audit(caplog):
    caplog.set_level(logging.DEBUG, logger="xarray_annotated")
    return caplog


class TestInputConversion:
    def test_marker_opted_in_conversion_is_debug(self, audit):
        units.check_units(_da(), Unit("Pa", convert=True), "p")
        [(level, msg)] = _records(audit, UNITS)
        assert level == logging.DEBUG
        assert "'hPa' -> 'Pa'" in msg

    def test_argument_opted_in_conversion_is_debug(self, audit):
        units.check_units(_da(), "Pa", "p", convert=True)
        [(level, _)] = _records(audit, UNITS)
        assert level == logging.DEBUG

    def test_conversion_through_enforce_is_debug(self, audit):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]): ...

        f(_da())
        [(level, _)] = _records(audit, UNITS)
        assert level == logging.DEBUG

    def test_quantified_input_conversion_is_logged(self, audit):
        units.check_units(_da().pint.quantify(), "Pa", "p", convert=True)
        [(level, _)] = _records(audit, UNITS)
        assert level == logging.DEBUG

    def test_same_unit_respelling_is_not_logged(self, audit):
        units.check_units(_da("pascal"), "Pa", "p")
        assert _records(audit, UNITS) == []

    def test_refused_conversion_is_not_logged(self, audit):
        with pytest.raises(ValueError, match="convert=True"):
            units.check_units(_da(), "Pa", "p")
        assert _records(audit, UNITS) == []

    def test_message_carries_qualname(self, audit):
        units.check_units(_da(), "Pa", "p", convert=True, qualname="mod.f")
        [(_, msg)] = _records(audit, UNITS)
        assert msg.startswith("[mod.f] input 'p'")


class TestIgnoredInputs:
    @pytest.mark.parametrize(
        ("unit", "fragment"),
        [
            (None, "no 'units' attribute"),
            ("x" * 300, "over-long"),
            ("not_a_unit", "unparseable"),
        ],
    )
    def test_ignore_is_info(self, audit, unit, fragment):
        units.check_units(_da(unit), "Pa", "p", on_missing="ignore")
        [(level, msg)] = _records(audit, UNITS)
        assert level == logging.INFO
        assert fragment in msg

    def test_warn_is_not_also_logged(self, audit):
        with pytest.warns(units.UnitsWarning):
            units.check_units(_da(None), "Pa", "p")
        assert _records(audit, UNITS) == []


class TestOutputs:
    def test_stamp_over_different_label_is_debug(self, audit):
        units.apply_output_units(_da("hPa"), "Pa", "return")
        [(level, msg)] = _records(audit, UNITS)
        assert level == logging.DEBUG
        assert "over label 'hPa'" in msg

    def test_stamp_over_unparseable_label_is_debug(self, audit):
        units.apply_output_units(_da("not_a_unit"), "Pa", "return")
        [(level, _)] = _records(audit, UNITS)
        assert level == logging.DEBUG

    @pytest.mark.parametrize("label", [None, "Pa", "pascal"])
    def test_stamp_over_no_or_same_label_is_not_logged(self, audit, label):
        units.apply_output_units(_da(label), "Pa", "return")
        assert _records(audit, UNITS) == []

    def test_quantified_output_conversion_is_debug(self, audit):
        out = units.apply_output_units(_da().pint.quantify(), "Pa", "return")
        np.testing.assert_allclose(out.pint.magnitude, [100.0])
        [(level, msg)] = _records(audit, UNITS)
        assert level == logging.DEBUG
        assert "quantified 'hPa' -> declared 'Pa'" in msg


class TestIgnoredMismatches:
    def test_schema_ignore_is_info(self, audit):
        schema.check_schema(_da(), Dims("time"), "p", on_mismatch="ignore")
        [(level, msg)] = _records(audit, SCHEMA)
        assert level == logging.INFO
        assert msg.endswith("(ignored)")

    def test_schema_match_is_not_logged(self, audit):
        schema.check_schema(_da(), Dims("x"), "p", on_mismatch="ignore")
        assert _records(audit, SCHEMA) == []

    def test_temporal_uninferable_ignore_is_info(self, audit):
        da = xr.DataArray(
            [1.0, 2.0],
            dims="time",
            coords={"time": xr.date_range("2020-01-01", periods=2, freq="D")},
        )
        temporal.check_freq(da, Freq("D"), "p", on_uninferable="ignore")
        [(level, msg)] = _records(audit, TEMPORAL)
        assert level == logging.INFO
        assert "uninferable" in msg


class TestQuietByDefault:
    def test_nothing_reaches_a_warning_level_handler(self, caplog):
        caplog.set_level(logging.WARNING, logger="xarray_annotated")
        units.check_units(_da(), "Pa", "p", convert=True)
        units.apply_output_units(_da(), "Pa", "return")
        assert caplog.records == []

    def test_switched_off_logs_nothing(self, audit):
        with xra.enabled(False):
            units.check_units(_da(), "Pa", "p", convert=True)
            schema.check_schema(_da(), Dims("time"), "p", on_mismatch="ignore")
        assert audit.records == []
