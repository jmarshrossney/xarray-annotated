"""Tests for the package-wide switch."""

import threading
from typing import Annotated

import numpy as np
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Unit, _config
from xarray_annotated.units import UnitError


def _resolve() -> bool:
    return _config.resolve_enabled()


@xra.enforce
def _to_pa(
    p: Annotated[xr.DataArray, Unit("Pa")],
) -> Annotated[xr.DataArray, Unit("Pa")]:
    return p


def _hpa() -> xr.DataArray:
    return xr.DataArray([1.0], attrs={"units": "hPa"})


class TestResolution:
    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv(_config.ENABLED_ENV_VAR, raising=False)
        assert _resolve() is True

    @pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
    def test_env_switches_off(self, monkeypatch, value):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, value)
        assert _resolve() is False

    def test_empty_env_is_unset(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "")
        assert _resolve() is True

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "maybe")
        with pytest.raises(ValueError, match=_config.ENABLED_ENV_VAR):
            _resolve()

    def test_block_restores_on_exit(self):
        with xra.enabled(False):
            assert _resolve() is False
        assert _resolve() is True

    def test_block_restores_on_exception(self):
        with pytest.raises(RuntimeError), xra.enabled(False):
            raise RuntimeError
        assert _resolve() is True

    def test_innermost_block_wins(self):
        with xra.enabled(False):
            with xra.enabled(True):
                assert _resolve() is True
            assert _resolve() is False

    def test_block_beats_env_off(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "false")
        with xra.enabled(True):
            assert _resolve() is True

    def test_block_beats_env_on(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "true")
        with xra.enabled(False):
            assert _resolve() is False

    def test_block_does_not_leak_to_other_threads(self):
        seen = []
        with xra.enabled(False):
            t = threading.Thread(target=lambda: seen.append(_resolve()))
            t.start()
            t.join()
        assert seen == [True]


class TestOnlyABoolIsAccepted:
    @pytest.mark.parametrize("value", ["false", "off", "true", 0, 1, None])
    def test_anything_else_raises(self, value):
        with pytest.raises(TypeError, match="enabled takes a bool"), xra.enabled(value):
            pytest.fail("the block should not run")

    def test_a_refused_value_changes_nothing(self):
        with pytest.raises(TypeError), xra.enabled("false"):  # type: ignore[arg-type]
            pass
        assert _resolve() is True

    def test_a_numpy_bool_is_a_bool(self):
        with xra.enabled(np.False_):
            assert _resolve() is False


class TestOffIsTotal:
    def test_on_checks(self):
        with pytest.raises(UnitError, match="'hPa' differs"):
            _to_pa(_hpa())

    def test_off_skips_checks(self):
        with xra.enabled(False):
            out = _to_pa(_hpa())
        assert float(out.values[0]) == 1.0
        assert out.attrs["units"] == "hPa"

    def test_env_off_skips_checks(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "off")
        assert _to_pa(_hpa()).attrs["units"] == "hPa"

    def test_declarations_still_validated_when_off(self):
        with (
            xra.enabled(False),
            pytest.raises(ValueError, match="not a recognised unit"),
        ):

            @xra.enforce
            def bad(p: Annotated[xr.DataArray, Unit("not_a_unit")]) -> None: ...
