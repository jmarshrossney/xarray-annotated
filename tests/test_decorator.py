"""Tests for ``enforce`` over unit declarations (plain-pint registry)."""

import typing
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import numpy as np
import pint
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Unit, units


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
    def test_converts_input_and_checks_output(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0, 2000.0]])

    def test_checks_typeddict_outputs(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, Unit("g / m**2 / d")]
            plain: xr.DataArray  # no declared unit

        @xra.enforce
        def f() -> Out:
            return {"gpp": _da([[1.0]], unit="g / m**2 / d"), "plain": _da([[2.0]])}

        out = f()
        assert out["gpp"].attrs["units"] == "g / m**2 / d"
        assert "units" not in out["plain"].attrs

    def test_checks_dataclass_outputs(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, Unit("g / m**2 / d")]
            plain: xr.DataArray  # no declared unit

        @xra.enforce
        def f() -> Out:
            return Out(gpp=_da([[1.0]], unit="g / m**2 / d"), plain=_da([[2.0]]))

        out = f()
        assert out.gpp.attrs["units"] == "g / m**2 / d"
        assert "units" not in out.plain.attrs

    def test_non_dataarray_args_pass_through(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")], scale: int) -> xr.DataArray:
            return p * scale

        out = f(_da([[1.0]], unit="Pa"), 3)
        np.testing.assert_allclose(out.values, [[3.0]])

    def test_input_passed_by_keyword(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]) -> xr.DataArray:
            return p

        out = f(p=_da([[10.0]], unit="hPa"))
        np.testing.assert_allclose(out.values, [[1000.0]])

    def test_optional_dataarray_none_is_skipped(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray | None, Unit("Pa")] = None) -> xr.DataArray:
            return _da([[1.0]], unit="Pa")

        out = f()  # x defaults to None; must not raise
        assert out.attrs["units"] == "Pa"


class TestDefaults:
    def test_declared_default_is_converted(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)] = _da(
                [[10.0]], unit="hPa"
            ),
        ) -> xr.DataArray:
            return p

        out = f()  # p left at its default; must still be converted
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0]])

    def test_default_object_is_not_mutated(self):
        default = _da([[10.0]], unit="hPa")

        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)] = default,
        ) -> xr.DataArray:
            return p

        f()
        assert default.attrs["units"] == "hPa"
        np.testing.assert_allclose(default.values, [[10.0]])
        np.testing.assert_allclose(f().values, [[1000.0]])  # second call unaffected

    def test_varargs_and_kwargs_survive_apply_defaults(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)], *rest: int, **kw: int
        ) -> xr.DataArray:
            assert rest == (1, 2)
            assert kw == {"z": 3}
            return p

        out = f(_da([[10.0]], unit="hPa"), 1, 2, z=3)
        np.testing.assert_allclose(out.values, [[1000.0]])


class TestFailFast:
    def test_bad_declared_unit_raises_at_decoration(self):
        with pytest.raises(ValueError, match="not a recognised"):

            @xra.enforce
            def f(p: Annotated[xr.DataArray, Unit("not_a_unit")]) -> xr.DataArray:
                return p

    def test_bare_string_declares_nothing(self):
        # Pinned because it is silent: a string is a description, so this input
        # is not checked and this output is not stamped.
        @xra.enforce
        def f(p: Annotated[xr.DataArray, "Pa"]) -> Annotated[xr.DataArray, "Pa"]:
            return p

        out = f(_da([[1.0]], unit="kg"))
        assert out.attrs["units"] == "kg"

    def test_dimensional_mismatch_always_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("kg")]) -> xr.DataArray:
            return p

        with pytest.raises(pint.DimensionalityError):
            f(_da([[1.0]], unit="degC"))


class TestMissingUnits:
    def test_missing_unit_always_raises(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]) -> xr.DataArray:
            return p

        with pytest.raises(ValueError, match=r"no 'units'.*assign_attrs"):
            f(_da([[1.0]]))

    def test_no_enforce_keyword(self):
        with pytest.raises(TypeError, match="units_on_missing"):
            xra.enforce(units_on_missing="ignore")  # type: ignore[call-arg]

    def test_disabled_policy_is_total_noop(self):
        # enabled=False: input is not converted AND output is not stamped.
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        with xra.enabled(False):
            out = f(_da([[10.0]], unit="hPa"))
        assert out.attrs["units"] == "hPa"  # not converted, not re-stamped
        np.testing.assert_allclose(out.values, [[10.0]])


class TestConversionIsOptIn:
    def test_refused_by_default(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]) -> xr.DataArray:
            return p

        with pytest.raises(ValueError, match=r"Unit\('Pa', convert=True\)"):
            f(_da([[10.0]], unit="hPa"))

    def test_equivalent_spelling_needs_no_opt_in(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa")]) -> xr.DataArray:
            return p

        out = f(_da([[10.0]], unit="pascal"))
        np.testing.assert_allclose(out.values, [[10.0]])

    def test_opted_in_converts(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]) -> xr.DataArray:
            return p

        np.testing.assert_allclose(f(_da([[10.0]], unit="hPa")).values, [[1000.0]])

    def test_no_enforce_keyword(self):
        with pytest.raises(TypeError, match="units_on_inexact"):
            xra.enforce(units_on_inexact="convert")  # type: ignore[call-arg]


class TestAnnotationsSurviveWraps:
    """PEP 749 changed functools.wraps to copy __annotate__ instead of
    __annotations__ on Python 3.14, which can lose annotations that were
    injected after the function was compiled.  The decorator must explicitly
    restore fn.__annotations__ on the wrapper."""

    def test_annotations_in_source_survive(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        hints = typing.get_type_hints(f, include_extras=True)
        assert "return" in hints

    def test_injected_annotations_survive(self):
        src = "def f(p):\n    return p\n"
        ns: dict[str, Any] = {}
        exec(src, ns)
        fn = ns["f"]
        fn.__annotations__["return"] = Annotated[xr.DataArray, Unit("Pa")]

        wrapped = xra.enforce(fn)
        hints = typing.get_type_hints(wrapped, include_extras=True)
        assert "return" in hints


class TestOutputs:
    """A returned array must already match its declaration.

    Its author controls what it returns, so a mismatch is a bug: nothing is converted
    or labelled on the author's behalf.  ``attrs`` are inert under arithmetic, so a
    body that changes the unit has to state the new one.
    """

    @staticmethod
    def _returning(label):
        """A function whose return carries ``label`` while declaring 'Pa'."""

        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da([[1.0]], unit=label)

        return f

    # -- the label decides -----------------------------------------------------

    def test_equal_label_is_respelled(self):
        out = self._returning("pascal")()
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1.0]])

    def test_missing_label_raises_with_the_fix(self):
        with pytest.raises(ValueError, match=r"no 'units' attribute.*assign_attrs"):
            self._returning(None)()

    def test_unparseable_label_raises(self):
        with pytest.raises(ValueError, match="unparseable 'units' attribute"):
            self._returning("not_a_unit")()

    def test_overlong_label_raises_without_echoing_it(self):
        with pytest.raises(ValueError, match="over-long") as info:
            self._returning("m" * 10_000)()
        assert "m" * 100 not in str(info.value)

    def test_non_string_label_raises(self):
        with pytest.raises(ValueError, match="unparseable 'units' attribute"):
            units.apply_output_units(_da([[1.0]]).assign_attrs(units=3), "Pa", "r")

    def test_compatible_label_raises_with_the_fix(self):
        with pytest.raises(ValueError, match=r"'hPa' differs.*assign_attrs"):
            self._returning("hPa")()

    def test_incompatible_label_raises_with_the_fix(self):
        with pytest.raises(pint.DimensionalityError, match=r"'return'.*assign_attrs"):
            self._returning("kg")()

    def test_message_names_the_function(self):
        with pytest.raises(ValueError, match=r"\[[\w.<>]*\bf\] output"):
            self._returning("hPa")()

    # -- bodies ------------------------------------------------------------------

    def test_a_forgotten_conversion_raises(self):
        @xra.enforce
        def forgot(
            p: Annotated[xr.DataArray, Unit("hPa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p

        with pytest.raises(ValueError, match="'hPa' differs"):
            forgot(_da([[1.0]], unit="hPa"))

    def test_a_stale_label_raises(self):
        @xra.enforce
        def to_pascals(
            p: Annotated[xr.DataArray, Unit("hPa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p * 100.0

        with (
            xr.set_options(keep_attrs=True),
            pytest.raises(ValueError, match="'hPa' differs"),
        ):
            to_pascals(_da([[1013.0]], unit="hPa"))

    def test_a_body_can_state_a_new_unit(self):
        @xra.enforce
        def to_pascals(
            p: Annotated[xr.DataArray, Unit("hPa")],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return (p * 100.0).assign_attrs(units="Pa")

        out = to_pascals(_da([[1013.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[101300.0]])

    def test_a_unit_preserving_body_passes(self):
        @xra.enforce
        def f(
            p: Annotated[xr.DataArray, Unit("Pa", convert=True)],
        ) -> Annotated[xr.DataArray, Unit("Pa")]:
            return p.isel(pixel=slice(0, 1))

        out = f(_da([[10.0, 20.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[1000.0]])

    # -- convert does not apply -----------------------------------------------

    @pytest.mark.parametrize(
        "make", [lambda: _da([[1.0]], "hPa"), lambda: _quantified([[1.0]], "hPa")]
    )
    def test_convert_does_not_apply_to_an_output(self, make):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa", convert=True)]:
            return make()

        with pytest.raises(ValueError, match="'hPa' differs"):
            f()

    def test_one_marker_can_serve_both_sides(self):
        marker = Unit("Pa", convert=True)

        @xra.enforce
        def f(p: Annotated[xr.DataArray, marker]) -> Annotated[xr.DataArray, marker]:
            return p

        out = f(_da([[1.0]], unit="hPa"))
        assert out.attrs["units"] == "Pa"
        np.testing.assert_allclose(out.values, [[100.0]])

    # -- container returns --------------------------------------------------

    def test_typeddict_fields_are_checked(self):
        class Out(TypedDict):
            good: Annotated[xr.DataArray, Unit("Pa")]
            bad: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return {"good": _da([[1.0]], unit="Pa"), "bad": _da([[2.0]], unit="kg")}

        with pytest.raises(pint.DimensionalityError, match="output 'bad'"):
            f()

    def test_dataclass_fields_are_checked(self):
        @dataclass
        class Out:
            bad: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(bad=_da([[2.0]], unit="kg"))

        with pytest.raises(pint.DimensionalityError, match="output 'bad'"):
            f()

    # -- anything that is not a DataArray passes through, as for an input -------

    @pytest.mark.parametrize("value", [None, np.array([1.0]), 1.0])
    def test_non_dataarray_return_passes_through(self, value):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return value

        assert f() is value

    def test_optional_return_of_none_passes(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")] | None:
            return None

        assert f() is None

    def test_missing_typeddict_key_passes(self):
        class Out(TypedDict, total=False):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return {}

        assert f() == {}

    def test_none_dataclass_field_passes(self):
        @dataclass
        class Out:
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(p=None)  # type: ignore[arg-type]

        assert f().p is None

    def test_undeclared_container_passes(self):
        class Out(TypedDict):
            p: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return None  # type: ignore[return-value]

        assert f() is None

    # -- plumbing ---------------------------------------------------------

    def test_disabled_is_a_total_no_op(self):
        f = self._returning("kg")
        with xra.enabled(False):
            out = f()
        assert out.attrs["units"] == "kg"

    def test_the_on_output_keyword_is_gone(self):
        with pytest.raises(TypeError, match="units_on_output"):
            xra.enforce(units_on_output="stamp")  # type: ignore[call-arg]


class TestApplyOutputUnitsPrimitive:
    """``apply_output_units`` is public; it must behave standalone."""

    def test_respells_a_copy(self):
        da = _da([[1.0]], unit="pascal")
        out = units.apply_output_units(da, "Pa", "return")
        assert out is not da
        assert out.attrs["units"] == "Pa"
        assert da.attrs["units"] == "pascal"

    def test_has_no_setting(self):
        with pytest.raises(TypeError):
            units.apply_output_units(_da([[1.0]]), "Pa", "r", on_output="stamp")  # type: ignore[call-arg]

    def test_qualname_omitted_when_none(self):
        with pytest.raises(ValueError, match="output 'return'") as excinfo:
            units.apply_output_units(_da([[1.0]], unit="hPa"), "Pa", "return")
        assert "] output" not in str(excinfo.value)  # no empty '[] ' qualname prefix


def _quantified(values, unit):
    """A DataArray holding a pint Quantity, as pint-xarray users produce."""
    return _da(values, unit=unit).pint.quantify()


class TestQuantifiedOutputs:
    """A quantified return must already be in its declared unit, and stays quantified."""

    def test_already_declared_unit_is_returned_untouched(self):
        # Same object, and crucially no copy of the buffer: dequantifying purely to
        # write an attrs string would cost one.
        da = _quantified([[1.0]], "Pa")
        out = units.apply_output_units(da, "Pa", "return")
        assert out is da
        assert "units" not in out.attrs

    def test_equivalent_spelling_is_not_a_conversion(self):
        da = _quantified([[1.0]], "pascal")
        assert units.apply_output_units(da, "Pa", "return") is da

    @pytest.mark.parametrize(
        "declared", ["Pa", Unit("Pa"), Unit("Pa", convert=True)], ids=repr
    )
    def test_a_different_unit_raises_with_the_fix(self, declared):
        with pytest.raises(ValueError, match=r"'hPa' differs.*pint\.to\('Pa'\)"):
            units.apply_output_units(_quantified([[1.0]], "hPa"), declared, "return")

    def test_an_incompatible_unit_raises(self):
        with pytest.raises(pint.DimensionalityError, match="'return'"):
            units.apply_output_units(_quantified([[1.0]], "hPa"), "kg", "return")

    def test_a_body_can_convert_it_itself(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _quantified([[1.0]], "hPa").pint.to("Pa")

        np.testing.assert_allclose(f().pint.magnitude, [[100.0]])

    def test_a_wrapper_can_attach_a_unit_and_convert(self):
        @xra.enforce
        def wrap() -> Annotated[xr.DataArray, Unit("Pa")]:
            bare = _da([[1.0]])  # as a library would return it
            return bare.assign_attrs(units="hPa").pint.quantify().pint.to("Pa")

        np.testing.assert_allclose(wrap().pint.magnitude, [[100.0]])

    def test_unit_from_a_foreign_registry_raises(self):
        # Unparseable here, as an unparseable attrs label is.
        foreign = pint.UnitRegistry(force_ndarray_like=True)
        foreign.define("widget = [thing]")
        da = _da([[1.0]]).pint.quantify("widget", unit_registry=foreign)
        with pytest.raises(ValueError, match="unparseable 'units' attribute 'widget'"):
            units.apply_output_units(da, "Pa", "return")

    def test_disabled_policy_leaves_it_alone(self):
        da = _quantified([[1.0]], "hPa")
        with xra.enabled(False):
            assert units.apply_output_units(da, "Pa", "return") is da


class TestQuantifiedThroughDecorator:
    def test_quantified_input_end_to_end(self):
        @xra.enforce
        def f(p: Annotated[xr.DataArray, Unit("Pa", convert=True)]) -> xr.DataArray:
            return p

        out = f(_quantified([[1.0]], "hPa"))
        np.testing.assert_allclose(out.values, [[100.0]])

    def test_respelled_output_is_written_back_for_single_return(self):
        @xra.enforce
        def f() -> Annotated[xr.DataArray, Unit("Pa")]:
            return _da([[1.0]], unit="pascal")

        # Respelling produces a *new* array, so a decorator that discarded
        # apply_output_units' return would silently hand back the old label.
        assert f().attrs["units"] == "Pa"

    def test_respelled_output_is_written_back_for_typeddict(self):
        class Out(TypedDict):
            gpp: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return {"gpp": _da([[1.0]], unit="pascal")}

        assert f()["gpp"].attrs["units"] == "Pa"

    def test_respelled_output_is_written_back_for_dataclass(self):
        @dataclass
        class Out:
            gpp: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(gpp=_da([[1.0]], unit="pascal"))

        assert f().gpp.attrs["units"] == "Pa"

    def test_frozen_dataclass_is_fine_when_no_conversion_is_needed(self):
        # Nothing is written back unless the array actually changed, so the
        # common case does not trip over frozen=True.
        @dataclass(frozen=True)
        class Out:
            gpp: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(gpp=_quantified([[1.0]], "Pa"))

        assert f().gpp.pint.units is not None

    def test_frozen_dataclass_with_a_respelling_is_rebuilt(self):
        @dataclass(frozen=True)
        class Out:
            gpp: Annotated[xr.DataArray, Unit("Pa")]

        @xra.enforce
        def f() -> Out:
            return Out(gpp=_da([[1.0]], unit="pascal"))

        assert f().gpp.attrs["units"] == "Pa"
