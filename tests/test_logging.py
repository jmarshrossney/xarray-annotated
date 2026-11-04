"""The audit trail: every non-trivial action is logged.

DEBUG marks every conversion, which is always opted into.  A failed check raises, so
it has nothing to log.
"""

import logging
from typing import Annotated

import pint_xarray  # noqa: F401  (registers the .pint accessor)
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Dims, Unit, schema, units

UNITS = "xarray_annotated.units"


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

    def test_conversion_through_enforce_is_debug(self, audit):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]): ...

        f(_da())
        [(level, _)] = _records(audit, UNITS)
        assert level == logging.DEBUG

    def test_quantified_input_conversion_is_logged(self, audit):
        units.check_units(_da().pint.quantify(), Unit("Pa", convert=True), "p")
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
        units.check_units(_da(), Unit("Pa", convert=True), "p", qualname="mod.f")
        [(_, msg)] = _records(audit, UNITS)
        assert msg.startswith("[mod.f] input 'p'")


class TestUnreadableInputs:
    @pytest.mark.parametrize("unit", [None, "x" * 300, "not_a_unit"])
    def test_refused_not_logged(self, audit, unit):
        with pytest.raises(ValueError, match="'units' attribute"):
            units.check_units(_da(unit), "Pa", "p")
        assert _records(audit, UNITS) == []


class TestOutputs:
    @pytest.mark.parametrize("label", ["Pa", "pascal"])
    def test_a_matching_output_is_not_logged(self, audit, label):
        units.apply_output_units(_da(label), "Pa", "return")
        assert _records(audit, UNITS) == []

    @pytest.mark.parametrize("label", [None, "hPa", "not_a_unit", "e" * 300])
    def test_a_mismatched_output_raises_and_is_not_logged(self, audit, label):
        with pytest.raises(ValueError, match="output 'return'"):
            units.apply_output_units(_da(label), "Pa", "return")
        assert _records(audit, UNITS) == []

    def test_over_long_label_is_neither_parsed_nor_echoed(self, monkeypatch):
        label = "e" * 300

        def fail(*_):
            raise AssertionError("over-long label reached the parser")

        monkeypatch.setattr(units._check, "get_registry", fail)
        with pytest.raises(ValueError, match="over-long") as info:
            units.apply_output_units(_da(label), "Pa", "return")
        assert label not in str(info.value)


class TestQuietByDefault:
    def test_nothing_reaches_a_warning_level_handler(self, caplog):
        caplog.set_level(logging.WARNING, logger="xarray_annotated")
        units.check_units(_da(), Unit("Pa", convert=True), "p")
        units.apply_output_units(_da("pascal"), "Pa", "return")
        assert caplog.records == []

    def test_switched_off_logs_nothing(self, audit):
        with xra.enabled(False):
            units.check_units(_da(), Unit("Pa", convert=True), "p")
            schema.check_schema(_da(), Dims("time"), "p")
        assert audit.records == []
