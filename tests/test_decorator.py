"""Tests for the ``declare_units`` signature-driven decorator (plain-pint registry)."""

import typing
import warnings
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import numpy as np
import pint
import pytest
import xarray as xr

from xarray_annotated import units


def _da(values, unit=None):
    """Build a (time, pixel) DataArray, optionally with a units attribute."""
    arr = np.asarray(values, dtype=float)
    time = xr.date_range("2020-01-01", periods=arr.shape[0], freq="7D")
    da = xr.DataArray(
        arr,
        dims=("time", "pixel"),
        coords={"time": time, "pixel": np.arange(arr.shape[1])},
    )
    if unit is not None:
        da.attrs["units"] = unit
    return da


class TestBareDecorator:
    def test_converts_input_and_stamps_output(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> Annotated[xr.DataArray, "Pa"]:
            return p

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_stamps_typeddict_outputs(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            plain: xr.DataArray  # no declared unit

        @units.declare_units
        def f() -> Out:
            return {"gpp": _da([[1.0]]), "plain": _da([[2.0]])}

        out = f()
        assert out["gpp"].attrs["units"] == "g / m**2 / d"
        assert "units" not in out["plain"].attrs

    def test_stamps_dataclass_outputs(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, "g / m**2 / d"]
            plain: xr.DataArray  # no declared unit

        @units.declare_units
        def f() -> Out:
            return Out(gpp=_da([[1.0]]), plain=_da([[2.0]]))

        out = f()
        assert out.gpp.attrs["units"] == "g / m**2 / d"
        assert "units" not in out.plain.attrs

    def test_non_dataarray_args_pass_through(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"], scale: int) -> xr.DataArray:
            return p * scale

        out = f(_da([[1.0]], unit="Pa"), 3)
        np.testing.assert_allclose(out.values, [[3.0]])

    def test_input_passed_by_keyword(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        out = f(p=_da([[10.0]], unit="hPa"))
        np.testing.assert_allclose(out.values, [[1000.0]])

    def test_optional_dataarray_none_is_skipped(self):
        @units.declare_units
        def f(x: Annotated[xr.DataArray | None, "Pa"] = None) -> xr.DataArray:
            return _da([[1.0]], unit="Pa")

        out = f()  # x defaults to None; must not raise
        assert out.attrs["units"] == "Pa"


class TestFailFast:
    def test_bad_declared_unit_raises_at_decoration(self):
        with pytest.raises(ValueError, match="not a recognised"):

            @units.declare_units
            def f(p: Annotated[xr.DataArray, "not_a_unit"]) -> xr.DataArray:
                return p

    def test_dimensional_mismatch_always_raises(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "kg"]) -> xr.DataArray:
            return p

        with pytest.raises(pint.DimensionalityError):
            f(_da([[1.0]], unit="degC"))


class TestPolicyResolution:
    def test_on_missing_kwarg_overrides_global(self):
        @units.declare_units(on_missing="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        # Global on_missing is warn, but the per-decorator "error" must raise.
        with (
            units.policy(on_missing="warn"),
            pytest.raises(ValueError, match="no 'units'"),
        ):
            f(_da([[1.0]]))

    def test_default_resolves_active_policy_per_call(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        da = _da([[1.0]])  # missing units
        with (
            units.policy(on_missing="error"),
            pytest.raises(ValueError, match="no 'units'"),
        ):
            f(da)
        with units.policy(on_missing="warn"), pytest.warns(units.UnitsWarning):
            f(da)

    def test_on_missing_ignore_skips_input_validation(self):
        @units.declare_units(on_missing="ignore")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = f(_da([[1.0]]))  # missing units, but ignore neither raises nor warns
        assert "units" not in out.attrs

    def test_disabled_policy_is_total_noop(self):
        # enabled=False: input is not converted AND output is not stamped.
        @units.declare_units
        def f(
            p: Annotated[xr.DataArray, "Pa"],
        ) -> Annotated[xr.DataArray, "Pa"]:
            return p

        with units.policy(enabled=False):
            out = f(_da([[10.0]], unit="hPa"))
        assert out.attrs["units"] == "hPa"  # not converted, not re-stamped
        np.testing.assert_allclose(out.values, [[10.0]])

    def test_invalid_on_missing_kwarg_raises_at_decoration(self):
        with pytest.raises(ValueError, match="Invalid on_missing"):

            @units.declare_units(on_missing="bogus")  # type: ignore[arg-type]
            def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
                return p


class TestInexact:
    def test_on_inexact_error_forbids_conversion(self):
        @units.declare_units(on_inexact="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with pytest.raises(ValueError, match="on_inexact='error'"):
            f(_da([[10.0]], unit="hPa"))

    def test_on_inexact_error_accepts_equivalent_spelling(self):
        @units.declare_units(on_inexact="error")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        out = f(_da([[10.0]], unit="pascal"))
        np.testing.assert_allclose(out.values, [[10.0]])

    def test_on_inexact_warn_converts_with_warning(self):
        @units.declare_units(on_inexact="warn")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> xr.DataArray:
            return p

        with pytest.warns(units.UnitsWarning, match="value-changing"):
            out = f(_da([[10.0]], unit="hPa"))
        np.testing.assert_allclose(out.values, [[1000.0]])


class TestAnnotationsSurviveWraps:
    """PEP 749 changed functools.wraps to copy __annotate__ instead of
    __annotations__ on Python 3.14, which can lose annotations that were
    injected after the function was compiled.  The decorator must explicitly
    restore fn.__annotations__ on the wrapper."""

    def test_annotations_in_source_survive(self):
        @units.declare_units
        def f(p: Annotated[xr.DataArray, "Pa"]) -> Annotated[xr.DataArray, "Pa"]:
            return p

        hints = typing.get_type_hints(f, include_extras=True)
        assert "return" in hints

    def test_injected_annotations_survive(self):
        src = "def f(p):\n    return p\n"
        ns: dict[str, Any] = {}
        exec(src, ns)
        fn = ns["f"]
        fn.__annotations__["return"] = Annotated[xr.DataArray, "Pa"]

        wrapped = units.declare_units(fn)
        hints = typing.get_type_hints(wrapped, include_extras=True)
        assert "return" in hints


class TestOnOutput:
    """The ``on_output`` axis: whether a returned array's own unit label is
    verified before being overwritten.

    Outputs are *stamped* rather than converted because xarray's ``attrs`` are
    inert under arithmetic: a body converting by scalar multiplication returns an
    array still labelled with its input's unit.  These tests pin the four label
    cases (absent / equal / compatible-but-different / incompatible) across both
    modes, and the limits of what ``"strict"`` can mean.
    """

    @staticmethod
    def _returning(label, on_output=None):
        """A function whose return carries ``label`` while declaring 'Pa'."""

        @units.declare_units(on_output=on_output)
        def f() -> Annotated[xr.DataArray, "Pa"]:
            return _da([[1.0]], unit=label)

        return f

    # -- default behaviour is unchanged ------------------------------------

    def test_default_is_stamp(self):
        assert units.get_policy().on_output == "stamp"

    def test_stamp_does_not_reconvert(self):
        """The regression this asymmetry exists to prevent: a body that has
        already done the arithmetic must not have it applied a second time."""

        @units.declare_units
        def to_pascals(
            p: Annotated[xr.DataArray, "hPa"],
        ) -> Annotated[xr.DataArray, "Pa"]:
            return p * 100.0  # values now Pa; attrs still say hPa

        with xr.set_options(keep_attrs=True):
            out = to_pascals(_da([[1013.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[101300.0]])

    def test_stamp_tolerates_dimension_changing_body(self):
        """A body converting moles -> mass by a scalar factor leaves a
        dimensionally *incompatible* stale label.  Under the default this is
        expected, not an error -- the label simply carries no information."""

        @units.declare_units
        def to_mass(
            flux: Annotated[xr.DataArray, "umol / m**2 / s"],
        ) -> Annotated[xr.DataArray, "g / m**2 / d"]:
            return flux * 1.0377504

        with xr.set_options(keep_attrs=True):
            out = to_mass(_da([[1.0]], unit="umol / m**2 / s"))
        assert out.attrs["units"] == "g / m**2 / d"
        np.testing.assert_allclose(out.values, [[1.0377504]])

    @pytest.mark.parametrize("mode", ["stamp", "strict"])
    def test_absent_label_is_stamped_in_every_mode(self, mode):
        out = self._returning(None, on_output=mode)()
        assert out.attrs["units"] == "Pa"

    @pytest.mark.parametrize("mode", ["stamp", "strict"])
    def test_equal_label_passes_in_every_mode(self, mode):
        # 'pascal' is an equivalent spelling of 'Pa': no value change implied.
        out = self._returning("pascal", on_output=mode)()
        assert out.attrs["units"] == "Pa"

    # -- stamp believes nothing ---------------------------------------------

    @pytest.mark.parametrize("label", ["hPa", "kg"])
    def test_stamp_overwrites_any_label(self, label):
        out = self._returning(label, on_output="stamp")()
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1.0]])  # never rescaled

    # -- strict requires the label to match ---------------------------------

    def test_strict_rejects_compatible_but_different(self):
        with pytest.raises(ValueError, match="on_output='strict'"):
            self._returning("hPa", on_output="strict")()

    def test_strict_rejects_incompatible(self):
        with pytest.raises(pint.DimensionalityError) as excinfo:
            self._returning("kg", on_output="strict")()
        msg = str(excinfo.value)
        assert "kilogram" in msg and "pascal" in msg
        assert "output 'return'" in msg
        assert "on_output='strict'" in msg

    def test_strict_message_names_the_function(self):
        with pytest.raises(pint.DimensionalityError, match=r"\[[\w.<>]*\bf\] output"):
            self._returning("kg", on_output="strict")()

    def test_strict_message_suggests_the_escape_hatch(self):
        """The likeliest cause is a scalar-arithmetic body, so the message must
        point at that rather than implying the values are wrong."""
        with pytest.raises(pint.DimensionalityError, match="scalar arithmetic"):
            self._returning("kg", on_output="strict")()

    def test_strict_accepts_a_body_that_clears_its_label(self):
        """The documented escape hatch for a manual-arithmetic body."""

        @units.declare_units(on_output="strict")
        def f(
            p: Annotated[xr.DataArray, "hPa"],
        ) -> Annotated[xr.DataArray, "Pa"]:
            out = p * 100.0
            out.attrs.pop("units", None)
            return out

        with xr.set_options(keep_attrs=True):
            out = f(_da([[1013.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[101300.0]])

    def test_strict_passes_a_unit_preserving_body(self):
        """Pass-through and subsetting bodies keep a truthful label, which is the
        case ``strict`` is actually for."""

        @units.declare_units(on_output="strict")
        def f(p: Annotated[xr.DataArray, "Pa"]) -> Annotated[xr.DataArray, "Pa"]:
            return p.isel(pixel=slice(0, 1))

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0]])

    # -- unverifiable labels are never fatal -------------------------------

    def test_unparseable_label_is_stamped(self):
        out = self._returning("not_a_unit", on_output="strict")()
        assert out.attrs["units"] == "Pa"

    def test_overlong_label_is_stamped(self):
        out = self._returning("m" * 300, on_output="strict")()
        assert out.attrs["units"] == "Pa"

    # -- container returns --------------------------------------------------

    def test_typeddict_fields_are_checked(self):
        class Out(TypedDict):
            good: Annotated[xr.DataArray, "Pa"]
            bad: Annotated[xr.DataArray, "Pa"]

        @units.declare_units(on_output="strict")
        def f() -> Out:
            return {"good": _da([[1.0]], unit="Pa"), "bad": _da([[2.0]], unit="kg")}

        with pytest.raises(pint.DimensionalityError, match="output 'bad'"):
            f()

    def test_dataclass_fields_are_checked(self):
        @dataclass
        class Out:
            bad: Annotated[xr.DataArray, "Pa"]

        @units.declare_units(on_output="strict")
        def f() -> Out:
            return Out(bad=_da([[2.0]], unit="kg"))

        with pytest.raises(pint.DimensionalityError, match="output 'bad'"):
            f()

    # -- policy plumbing ----------------------------------------------------

    def test_policy_context_manager_applies(self):
        f = self._returning("kg")
        with units.policy(on_output="strict"), pytest.raises(pint.DimensionalityError):
            f()
        # Restored on exit: the same call now merely stamps.
        assert f().attrs["units"] == "Pa"

    def test_set_policy_applies_and_clears(self):
        f = self._returning("kg")
        units.set_policy(on_output="strict")
        with pytest.raises(pint.DimensionalityError):
            f()
        units.set_policy(on_output=None)
        assert f().attrs["units"] == "Pa"

    def test_env_var_wins_over_set_policy(self, monkeypatch):
        monkeypatch.setenv("XARRAY_ANNOTATED_UNITS_ON_OUTPUT", "stamp")
        units.set_policy(on_output="strict")
        assert units.get_policy().on_output == "stamp"

    def test_decorator_argument_wins_over_policy(self):
        f = self._returning("kg", on_output="stamp")
        with units.policy(on_output="strict"):
            assert f().attrs["units"] == "Pa"

    def test_disabled_is_a_total_no_op(self):
        f = self._returning("kg", on_output="strict")
        with units.policy(enabled=False):
            out = f()
        assert out.attrs["units"] == "kg"  # not even stamped

    @pytest.mark.parametrize("bad", ["convert", "check", "STAMP_", "", "warn"])
    def test_invalid_value_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid on_output"):
            units.declare_units(on_output=bad)

    def test_invalid_value_rejected_by_set_policy(self):
        with pytest.raises(ValueError, match="Invalid on_output"):
            units.set_policy(on_output="check")  # type: ignore[arg-type]


class TestApplyOutputUnitsPrimitive:
    """``apply_output_units`` is public; it must behave standalone."""

    def test_stamps_and_returns_same_object(self):
        da = _da([[1.0]])
        out = units.apply_output_units(da, "Pa", "return")
        assert out is da
        assert out.attrs["units"] == "Pa"

    def test_defers_to_policy_when_none(self):
        da = _da([[1.0]], unit="kg")
        with units.policy(on_output="strict"), pytest.raises(pint.DimensionalityError):
            units.apply_output_units(da, "Pa", "return")

    def test_qualname_omitted_when_none(self):
        da = _da([[1.0]], unit="kg")
        with pytest.raises(pint.DimensionalityError) as excinfo:
            units.apply_output_units(da, "Pa", "return", "strict")
        msg = str(excinfo.value)
        assert "output 'return'" in msg
        assert "] output" not in msg  # no empty '[] ' qualname prefix
