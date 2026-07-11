"""Tests for the `xarray_annotated.temporal` frequency-validation domain.

Covers the `Freq` marker and its reader, the spacing/phase model behind
`freq_compatible`, the runtime `check_freq`, the policy, and the `declare_freq`
decorator.  Frequency validation never mutates: the checks assert/raise/warn but
return arrays unchanged.
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_annotated import temporal, units
from xarray_annotated.schema import Dims
from xarray_annotated.temporal import Freq
from xarray_annotated.temporal import _annotations, _config


def _da(times, dim="time", extra_coords=None):
    """A 1-d DataArray over `times`, plus any `extra_coords` on the same dim."""
    da = xr.DataArray(
        np.arange(len(times), dtype="float64"), dims=(dim,), coords={dim: times}
    )
    for name, values in (extra_coords or {}).items():
        da = da.assign_coords({name: (dim, values)})
    return da


# ---------------------------------------------------------------------------
# The Freq marker
# ---------------------------------------------------------------------------


class TestFreqMarker:
    def test_accessors_and_defaults(self):
        f = Freq("7D")
        assert f.freq == "7D"
        assert f.dim is None
        assert f.anchored is None
        assert f.on_mismatch is None
        assert Freq("W", dim="time", anchored=True).dim == "time"
        assert Freq("W", anchored=True).anchored is True

    def test_keeps_the_raw_spelling(self):
        # Never normalised: "W" must not become "W-SUN" (that is where
        # anchoredness would be lost).
        assert Freq("W").freq == "W"

    def test_repr_roundtrips_eval(self):
        markers = [
            Freq("7D"),
            Freq("W-WED", dim="time"),
            Freq("W", anchored=True),
            Freq("ME", anchored=False, on_mismatch="warn"),
            Freq("QE-MAR", dim="t", anchored=True, on_mismatch="ignore"),
        ]
        for marker in markers:
            assert eval(repr(marker)) == marker  # noqa: S307

    def test_equality_includes_all_fields(self):
        assert Freq("7D") == Freq("7D")
        assert Freq("7D") != Freq("W-WED")
        assert Freq("W") != Freq("W", anchored=True)
        assert Freq("W") != Freq("W", dim="time")
        assert Freq("W") != Freq("W", on_mismatch="warn")
        assert Freq("7D") != "7D"  # not equal to a bare string
        assert Freq("7D") != Dims("time")  # nor to another domain's marker

    def test_hashable_in_annotations(self):
        assert hash(Freq("W", anchored=True)) == hash(Freq("W", anchored=True))
        assert {Freq("7D"), Freq("W-WED")}
        assert hash(Annotated[xr.DataArray, Freq("7D")])

    def test_invalid_on_mismatch_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            Freq("7D", on_mismatch="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reading declarations off annotations
# ---------------------------------------------------------------------------


class TestAnnotatedFreq:
    def test_reads_the_marker(self):
        hint = Annotated[xr.DataArray, Freq("7D")]
        assert _annotations.annotated_freq(hint) == Freq("7D")

    def test_coexists_with_other_domains_metadata(self):
        hint = Annotated[xr.DataArray, "degC", Dims("time"), Freq("W-SUN")]
        assert _annotations.annotated_freq(hint) == Freq("W-SUN")

    def test_non_dataarray_yields_none(self):
        assert _annotations.annotated_freq(Annotated[bool, Freq("7D")]) is None

    def test_non_annotated_yields_none(self):
        assert _annotations.annotated_freq(xr.DataArray) is None

    def test_no_bare_string_shorthand(self):
        # A string in the metadata is a unit or a description, never a frequency.
        assert _annotations.annotated_freq(Annotated[xr.DataArray, "7D"]) is None

    def test_optional_dataarray_read(self):
        hint = Annotated[xr.DataArray | None, Freq("D")]
        assert _annotations.annotated_freq(hint) == Freq("D")

    def test_from_signature_inputs_and_single_output(self):
        def node(
            a: Annotated[xr.DataArray, Freq("D")],
            b: int,
        ) -> Annotated[xr.DataArray, Freq("W-SUN")]:
            return a

        inputs, output = _annotations.freq_from_signature(node)
        assert inputs == {"a": Freq("D")}
        assert output == Freq("W-SUN")

    def test_from_signature_typeddict_output(self):
        class Out(TypedDict):
            hi: Annotated[xr.DataArray, Freq("D")]
            lo: Annotated[xr.DataArray, Freq("ME")]

        def node() -> Out: ...

        _, output = _annotations.freq_from_signature(node)
        assert output == {"hi": Freq("D"), "lo": Freq("ME")}

    def test_from_signature_dataclass_output(self):
        @dataclass
        class Out:
            hi: Annotated[xr.DataArray, Freq("D")]
            plain: int

        def node() -> Out: ...

        _, output = _annotations.freq_from_signature(node)
        assert output == {"hi": Freq("D")}

    def test_from_signature_no_output(self):
        def node(a: int) -> int:
            return a

        inputs, output = _annotations.freq_from_signature(node)
        assert inputs == {}
        assert output is None


# ---------------------------------------------------------------------------
# Marker-vs-marker compatibility (no array in hand)
# ---------------------------------------------------------------------------

#: The frequency model, as a table: (a, b, compatible?).  Spacing is always compared;
#: phase only where both declarations determine it.
COMPATIBILITY_TABLE = [
    (Freq("7D"), Freq("W-WED"), True),  # same spacing; "7D" pins no anchor
    (Freq("7D"), Freq("W-SUN"), True),
    (Freq("W-SUN"), Freq("W-WED"), False),  # the resample-phase footgun
    (Freq("W"), Freq("W-WED"), True),  # "W" spelled without an anchor => unanchored
    (Freq("W-SUN", anchored=False), Freq("W-WED"), True),  # comparison suppressed
    (Freq("W", anchored=True), Freq("W-WED"), False),  # forced to pandas' W-SUN
    (Freq("D"), Freq("24h"), True),  # same fixed spacing
    (Freq("D"), Freq("7D"), False),
    (Freq("ME"), Freq("MS"), False),  # convention is always compared
    (Freq("ME"), Freq("ME"), True),
    (Freq("QE"), Freq("3ME"), True),  # same spacing and convention, neither anchors
    (Freq("QE-MAR"), Freq("QE-DEC"), False),  # both anchored, different
    (Freq("QE"), Freq("QE-DEC"), True),  # "QE" unanchored
    (Freq("YE"), Freq("12ME"), True),
    (Freq("ME"), Freq("30D"), False),  # "months" and "fixed" never compare equal
    (Freq("2W-WED"), Freq("14D"), True),  # multiples carried through
    # Beyond the plan's table: the year anchor behaves like the quarter one, and a
    # fixed-spacing offset has no anchor to force even under `anchored=True`.
    (Freq("YE-JUN"), Freq("YE-DEC"), False),
    (Freq("YE"), Freq("YE-DEC"), True),
    (Freq("D", anchored=True), Freq("24h"), True),
]


class TestFreqCompatible:
    @pytest.mark.parametrize(("a", "b", "expected"), COMPATIBILITY_TABLE)
    def test_truth_table(self, a, b, expected):
        assert temporal.freq_compatible(a, b) is expected

    @pytest.mark.parametrize(("a", "b", "expected"), COMPATIBILITY_TABLE)
    def test_symmetry(self, a, b, expected):
        assert temporal.freq_compatible(b, a) is expected

    def test_opaque_offsets_compare_by_string(self):
        # Business days have no spacing we can reduce, so only an exact match passes.
        assert temporal.freq_compatible(Freq("B"), Freq("B"))
        assert not temporal.freq_compatible(Freq("B"), Freq("D"))


# ---------------------------------------------------------------------------
# assert_valid_freq (fail-fast declaration checks)
# ---------------------------------------------------------------------------


class TestAssertValidFreq:
    @pytest.mark.parametrize("freq", ["M", "H", "nonsense", ""])
    def test_bad_offset_string(self, freq):
        with pytest.raises(ValueError, match="ctx: invalid frequency"):
            temporal.assert_valid_freq(Freq(freq), "ctx")

    def test_legacy_alias_message_surfaces(self):
        # pandas' own message is the one that says to spell it "ME".
        with pytest.raises(ValueError, match="ME"):
            temporal.assert_valid_freq(Freq("M"), "ctx")

    def test_empty_dim(self):
        with pytest.raises(ValueError, match="non-empty string"):
            temporal.assert_valid_freq(Freq("D", dim=""), "ctx")

    def test_valid_declarations_pass(self):
        temporal.assert_valid_freq(Freq("7D"), "ctx")
        temporal.assert_valid_freq(Freq("W-WED", dim="time"), "ctx")
        temporal.assert_valid_freq(Freq("QE-MAR", anchored=True), "ctx")


# ---------------------------------------------------------------------------
# check_freq (the runtime leg)
# ---------------------------------------------------------------------------


class TestCheckFreq:
    def test_matching_axis_passes_and_returns_same_object(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert temporal.check_freq(da, Freq("D"), "x") is da

    def test_wrong_spacing_raises(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, Freq("7D"), "x")

    def test_seven_day_axis_infers_as_weekly(self):
        # The headline case: pandas infers a 7D axis starting on a Wednesday as
        # "W-WED", so the declaration decides how strict the comparison is.
        da = _da(pd.date_range("2020-01-01", periods=5, freq="7D"))  # a Wednesday
        assert xr.infer_freq(da["time"]) == "W-WED"
        temporal.check_freq(da, Freq("7D"), "x")
        temporal.check_freq(da, Freq("W-WED"), "x")
        with pytest.raises(temporal.FreqError, match="expected 'W-SUN', got 'W-WED'"):
            temporal.check_freq(da, Freq("W-SUN"), "x")

    def test_too_few_points_is_uninferable(self):
        da = _da(pd.date_range("2020-01-01", periods=2, freq="D"))
        with pytest.warns(temporal.FreqWarning, match="uninferable"):
            assert temporal.check_freq(da, Freq("D"), "x") is da
        with pytest.raises(temporal.FreqError, match="uninferable"):
            temporal.check_freq(da, Freq("D"), "x", on_uninferable="error")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            temporal.check_freq(da, Freq("D"), "x", on_uninferable="ignore")

    def test_irregular_spacing_is_uninferable(self):
        da = _da(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"]))
        assert xr.infer_freq(da["time"]) is None
        with pytest.warns(temporal.FreqWarning, match="uninferable"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_no_datetime_coord_is_a_mismatch(self):
        da = xr.DataArray(np.zeros(3), dims=("time",))
        with pytest.raises(temporal.FreqError, match="no datetime coordinate"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_no_datetime_coord_under_warn(self):
        # The "no axis to check" mismatch is routed through on_mismatch, so `warn`
        # (and `ignore`) really do soften it rather than raising regardless.
        da = xr.DataArray(np.zeros(3), dims=("time",))
        with pytest.warns(temporal.FreqWarning, match="no datetime coordinate"):
            assert temporal.check_freq(da, Freq("D"), "x", on_mismatch="warn") is da

    def test_two_datetime_coords_are_ambiguous(self):
        da = _da(
            pd.date_range("2020-01-01", periods=5, freq="D"),
            extra_coords={"valid_time": pd.date_range("2021-01-01", periods=5)},
        )
        with pytest.raises(temporal.FreqError, match="ambiguous") as excinfo:
            temporal.check_freq(da, Freq("D"), "x")
        assert "time" in str(excinfo.value)
        assert "valid_time" in str(excinfo.value)

    def test_ambiguity_resolved_by_declaring_the_dim(self):
        da = _da(
            pd.date_range("2020-01-01", periods=5, freq="D"),
            extra_coords={"valid_time": pd.date_range("2021-01-01", periods=5)},
        )
        assert temporal.check_freq(da, Freq("D", dim="time"), "x") is da

    def test_declared_dim_absent_is_a_mismatch(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match="not a datetime coordinate"):
            temporal.check_freq(da, Freq("D", dim="nope"), "x")

    def test_cftime_calendar_axis(self):
        # The reason we use xarray's infer_freq rather than pandas': a non-standard
        # calendar has a CFTimeIndex, not a DatetimeIndex.
        times = xr.date_range(
            "2020-01-01", periods=5, freq="D", calendar="noleap", use_cftime=True
        )
        da = _da(times)
        assert isinstance(da.indexes["time"], xr.CFTimeIndex)
        assert temporal.check_freq(da, Freq("D"), "x") is da
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, Freq("ME"), "x")

    def test_marker_override_beats_argument(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.warns(temporal.FreqWarning):
            temporal.check_freq(
                da, Freq("7D", on_mismatch="warn"), "x", on_mismatch="error"
            )

    def test_qualname_prefix_in_message(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match=r"\[myfunc\] 'x'"):
            temporal.check_freq(da, Freq("7D"), "x", qualname="myfunc")

    def test_list_of_markers(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, [Freq("D"), Freq("ME")], "x")

    def test_disabled_is_noop(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with temporal.policy(enabled=False):
            assert temporal.check_freq(da, Freq("ME"), "x") is da  # would raise

    def test_invalid_severity_arguments(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            temporal.check_freq(da, Freq("D"), "x", on_mismatch="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid on_uninferable"):
            temporal.check_freq(da, Freq("D"), "x", on_uninferable="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_defaults(self):
        with temporal.policy(enabled=None, on_mismatch=None, on_uninferable=None):
            pol = temporal.get_policy()
            assert pol.enabled is True
            assert pol.on_mismatch == "error"
            assert pol.on_uninferable == "warn"

    def test_set_and_restore(self):
        with temporal.policy(on_mismatch="warn", on_uninferable="error"):
            pol = temporal.get_policy()
            assert pol.on_mismatch == "warn"
            assert pol.on_uninferable == "error"
        pol = temporal.get_policy()
        assert pol.on_mismatch == "error"
        assert pol.on_uninferable == "warn"

    def test_env_overrides_process(self, monkeypatch):
        with temporal.policy(on_mismatch="ignore", on_uninferable="ignore"):
            monkeypatch.setenv(_config.ON_MISMATCH_ENV_VAR, "warn")
            monkeypatch.setenv(_config.ON_UNINFERABLE_ENV_VAR, "error")
            pol = temporal.get_policy()
            assert pol.on_mismatch == "warn"
            assert pol.on_uninferable == "error"

    def test_invalid_on_mismatch_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ON_MISMATCH_ENV_VAR, "bogus")
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            temporal.get_policy()

    def test_invalid_on_uninferable_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ON_UNINFERABLE_ENV_VAR, "bogus")
        with pytest.raises(ValueError, match="Invalid on_uninferable"):
            temporal.get_policy()

    def test_set_policy_partial_update(self):
        temporal.set_policy(on_mismatch="warn")
        pol = temporal.get_policy()
        assert pol.on_mismatch == "warn"
        assert pol.on_uninferable == "warn"  # untouched axis keeps its default

    def test_none_clears_an_override(self):
        temporal.set_policy(on_uninferable="error")
        assert temporal.get_policy().on_uninferable == "error"
        temporal.set_policy(on_uninferable=None)
        assert temporal.get_policy().on_uninferable == "warn"

    def test_context_manager_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with temporal.policy(on_mismatch="ignore"):
                raise RuntimeError("boom")
        assert temporal.get_policy().on_mismatch == "error"

    def test_invalid_on_mismatch_raises(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            temporal.set_policy(on_mismatch="bogus")  # type: ignore[arg-type]

    def test_invalid_on_uninferable_raises(self):
        with pytest.raises(ValueError, match="Invalid on_uninferable"):
            temporal.set_policy(on_uninferable="bogus")  # type: ignore[arg-type]

    def test_invalid_enabled_env_raises(self, monkeypatch):
        monkeypatch.setenv(_config.ENABLED_ENV_VAR, "maybe")
        with pytest.raises(ValueError, match=_config.ENABLED_ENV_VAR):
            temporal.get_policy()

    def test_enabled_switch_is_shared_with_units(self):
        # The master switch is package-wide: toggling it here is seen by units.
        with temporal.policy(enabled=False):
            assert units.get_policy().enabled is False

    def test_policy_axis_types_are_public(self):
        assert temporal.OnMismatch is _config.OnMismatch
        assert temporal.OnUninferable is _config.OnUninferable


# ---------------------------------------------------------------------------
# declare_freq decorator
# ---------------------------------------------------------------------------


class TestDeclareFreq:
    def test_validates_input_and_passes_through(self):
        @temporal.declare_freq
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(da) is da
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_keyword_and_non_dataarray_args(self):
        @temporal.declare_freq
        def f(flag: bool, x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(True, x=da) is da

    def test_optional_none_input_skipped(self):
        @temporal.declare_freq
        def f(
            x: Annotated[xr.DataArray | None, Freq("D")] = None,
        ) -> xr.DataArray | None:
            return x

        assert f(None) is None

    def test_output_single_validated(self):
        @temporal.declare_freq
        def f(n: int) -> Annotated[xr.DataArray, Freq("W-SUN")]:
            # a Wednesday-anchored weekly axis: right spacing, wrong phase
            return _da(pd.date_range("2020-01-01", periods=5, freq="7D"))

        with pytest.raises(temporal.FreqError, match="expected 'W-SUN', got 'W-WED'"):
            f(1)

    def test_output_typeddict_validated(self):
        class Out(TypedDict):
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @temporal.declare_freq
        def f() -> Out:
            return {"weekly": _da(pd.date_range("2020-01-01", periods=5, freq="D"))}

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_output_dataclass_validated(self):
        @dataclass
        class Out:
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @temporal.declare_freq
        def f() -> Out:
            return Out(weekly=_da(pd.date_range("2020-01-01", periods=5, freq="D")))

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_decorator_kwargs(self):
        @temporal.declare_freq(on_mismatch="warn")
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        with pytest.warns(temporal.FreqWarning, match="frequency mismatch"):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_decorator_on_uninferable_kwarg(self):
        @temporal.declare_freq(on_uninferable="error")
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        with pytest.raises(temporal.FreqError, match="uninferable"):
            f(_da(pd.date_range("2020-01-01", periods=2, freq="D")))

    def test_marker_override_beats_decorator(self):
        @temporal.declare_freq(on_mismatch="error")
        def f(
            x: Annotated[xr.DataArray, Freq("D", on_mismatch="warn")],
        ) -> xr.DataArray:
            return x

        with pytest.warns(temporal.FreqWarning):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_wraps_metadata_preserved(self):
        @temporal.declare_freq
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            """Docstring."""
            return x

        assert f.__name__ == "f"
        assert f.__doc__ == "Docstring."

    def test_disabled_is_total_noop(self):
        @temporal.declare_freq
        def f(x: Annotated[xr.DataArray, Freq("ME")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with temporal.policy(enabled=False):
            assert f(da) is da

    def test_invalid_decorator_kwargs(self):
        with pytest.raises(ValueError, match="Invalid on_mismatch"):

            @temporal.declare_freq(on_mismatch="bogus")  # type: ignore[arg-type]
            def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
                return x

        with pytest.raises(ValueError, match="Invalid on_uninferable"):

            @temporal.declare_freq(on_uninferable="bogus")  # type: ignore[arg-type]
            def g(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_input_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @temporal.declare_freq
            def f(x: Annotated[xr.DataArray, Freq("M")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_output_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @temporal.declare_freq
            def f(n: int) -> Annotated[xr.DataArray, Freq("nonsense")]:
                return _da(pd.date_range("2020-01-01", periods=5, freq="D"))
