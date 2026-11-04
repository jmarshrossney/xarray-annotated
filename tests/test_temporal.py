"""Tests for the `xarray_annotated.temporal` frequency-validation domain.

Covers the `Freq` marker and its reader, the spacing/phase model behind
`freq_compatible`, the runtime `check_freq`, the policy, and `enforce` over `Freq`
markers.  Frequency validation never mutates: the checks assert/raise/warn but
return arrays unchanged.
"""

import warnings
from dataclasses import dataclass
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

import xarray_annotated as xra
from xarray_annotated import temporal
from xarray_annotated.schema import Dims
from xarray_annotated.temporal import Freq, _config


def _two_time_dims():
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    inits = pd.date_range("2021-01-01", periods=2, freq="D")
    return xr.DataArray(
        np.zeros((5, 2)), dims=("time", "init"), coords={"time": times, "init": inits}
    )


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
            assert eval(repr(marker)) == marker

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
    (Freq("QE-MAR"), Freq("QE-DEC"), True),  # the same four quarter ends
    (Freq("QS-JAN"), Freq("QS-APR"), True),  # the same four quarter starts
    (Freq("QE-MAR"), Freq("QE-NOV"), False),  # different quarter ends
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

    def test_uninferable_has_its_own_category(self):
        da = _da(pd.date_range("2020-01-01", periods=2, freq="D"))
        with pytest.warns(temporal.UninferableFreqWarning):
            temporal.check_freq(da, Freq("D"), "x")
        assert issubclass(temporal.UninferableFreqWarning, temporal.FreqWarning)

    def test_filtering_uninferable_keeps_mismatch_warnings(self):
        short = _da(pd.date_range("2020-01-01", periods=2, freq="D"))
        weekly = _da(pd.date_range("2020-01-05", periods=4, freq="W-SUN"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", temporal.UninferableFreqWarning)
            temporal.check_freq(short, Freq("D"), "x")
            temporal.check_freq(weekly, Freq("D"), "x", on_mismatch="warn")
        assert [type(w.message) for w in caught] == [temporal.FreqWarning]

    def test_irregular_spacing_is_uninferable(self):
        da = _da(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"]))
        assert xr.infer_freq(da["time"]) is None
        with pytest.warns(temporal.FreqWarning, match="uninferable"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_no_datetime_coord_is_a_mismatch(self):
        da = xr.DataArray(np.zeros(3), dims=("time",))
        with pytest.raises(temporal.FreqError, match="no indexed datetime coordinate"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_no_datetime_coord_under_warn(self):
        # The "no axis to check" mismatch is routed through on_mismatch, so `warn`
        # (and `ignore`) really do soften it rather than raising regardless.
        da = xr.DataArray(np.zeros(3), dims=("time",))
        with pytest.warns(temporal.FreqWarning, match="no indexed datetime coordinate"):
            assert temporal.check_freq(da, Freq("D"), "x", on_mismatch="warn") is da

    def test_two_indexed_datetime_coords_are_ambiguous(self):
        with pytest.raises(temporal.FreqError, match="ambiguous") as excinfo:
            temporal.check_freq(_two_time_dims(), Freq("D"), "x")
        assert "time" in str(excinfo.value)
        assert "init" in str(excinfo.value)

    def test_ambiguity_resolved_by_declaring_the_dim(self):
        da = _two_time_dims()
        assert temporal.check_freq(da, Freq("D", dim="time"), "x") is da

    def test_declared_dim_absent_is_a_mismatch(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match="not an indexed datetime"):
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
        with xra.enabled(False):
            assert temporal.check_freq(da, Freq("ME"), "x") is da  # would raise

    def test_invalid_severity_arguments(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(ValueError, match="Invalid on_mismatch"):
            temporal.check_freq(da, Freq("D"), "x", on_mismatch="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid on_uninferable"):
            temporal.check_freq(da, Freq("D"), "x", on_uninferable="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Axis defaults
# ---------------------------------------------------------------------------


class TestAxes:
    def test_defaults(self):
        assert _config.DEFAULT_ON_MISMATCH == "error"
        assert _config.DEFAULT_ON_UNINFERABLE == "warn"

    def test_axis_types_are_public(self):
        assert temporal.OnMismatch is _config.OnMismatch
        assert temporal.OnUninferable is _config.OnUninferable


# ---------------------------------------------------------------------------
# enforce over Freq markers
# ---------------------------------------------------------------------------


class TestDeclareFreq:
    def test_validates_input_and_passes_through(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(da) is da
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_keyword_and_non_dataarray_args(self):
        @xra.enforce
        def f(flag: bool, x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(True, x=da) is da

    def test_optional_none_input_skipped(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray | None, Freq("D")] = None,
        ) -> xr.DataArray | None:
            return x

        assert f(None) is None

    def test_declared_default_is_validated(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Freq("D")] = _da(
                pd.date_range("2020-01-01", periods=5, freq="7D")
            ),
        ) -> xr.DataArray:
            return x

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()  # x left at its non-conforming default

    def test_declared_default_honours_on_mismatch(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Freq("D", on_mismatch="warn")] = _da(
                pd.date_range("2020-01-01", periods=5, freq="7D")
            ),
        ) -> xr.DataArray:
            return x

        with pytest.warns(temporal.FreqWarning, match="frequency mismatch"):
            f()

    def test_varargs_and_kwargs_survive_apply_defaults(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Freq("D")], *rest: int, **kw: int
        ) -> xr.DataArray:
            assert rest == (1, 2)
            assert kw == {"z": 3}
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(da, 1, 2, z=3) is da

    def test_output_single_validated(self):
        @xra.enforce
        def f(n: int) -> Annotated[xr.DataArray, Freq("W-SUN")]:
            # a Wednesday-anchored weekly axis: right spacing, wrong phase
            return _da(pd.date_range("2020-01-01", periods=5, freq="7D"))

        with pytest.raises(temporal.FreqError, match="expected 'W-SUN', got 'W-WED'"):
            f(1)

    def test_output_typeddict_validated(self):
        class Out(TypedDict):
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @xra.enforce
        def f() -> Out:
            return {"weekly": _da(pd.date_range("2020-01-01", periods=5, freq="D"))}

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_output_dataclass_validated(self):
        @dataclass
        class Out:
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @xra.enforce
        def f() -> Out:
            return Out(weekly=_da(pd.date_range("2020-01-01", periods=5, freq="D")))

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_marker_on_mismatch_softens(self):
        @xra.enforce
        def f(
            x: Annotated[xr.DataArray, Freq("D", on_mismatch="warn")],
        ) -> xr.DataArray:
            return x

        with pytest.warns(temporal.FreqWarning, match="frequency mismatch"):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_decorator_on_uninferable_kwarg(self):
        @xra.enforce(temporal_on_uninferable="error")
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        with pytest.raises(temporal.FreqError, match="uninferable"):
            f(_da(pd.date_range("2020-01-01", periods=2, freq="D")))

    def test_wraps_metadata_preserved(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            """Docstring."""
            return x

        assert f.__name__ == "f"
        assert f.__doc__ == "Docstring."

    def test_disabled_is_total_noop(self):
        @xra.enforce
        def f(x: Annotated[xr.DataArray, Freq("ME")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with xra.enabled(False):
            assert f(da) is da

    def test_invalid_decorator_kwargs(self):
        with pytest.raises(ValueError, match="Invalid temporal_on_uninferable"):

            @xra.enforce(temporal_on_uninferable="bogus")  # type: ignore[arg-type]
            def g(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_input_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @xra.enforce
            def f(x: Annotated[xr.DataArray, Freq("M")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_output_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @xra.enforce
            def f(n: int) -> Annotated[xr.DataArray, Freq("nonsense")]:
                return _da(pd.date_range("2020-01-01", periods=5, freq="D"))


class TestAnnotationsSurviveWraps:
    def test_injected_annotations_survive(self):
        import typing

        src = "def f(p):\n    return p\n"
        ns: dict[str, typing.Any] = {}
        exec(src, ns)
        fn = ns["f"]
        fn.__annotations__["return"] = Annotated[xr.DataArray, Freq("D")]

        wrapped = xra.enforce(fn)
        hints = typing.get_type_hints(wrapped, include_extras=True)
        assert "return" in hints


class TestQuarterlyAxisAgainstItsOwnDeclaration:
    @pytest.mark.parametrize(
        ("freq", "start"),
        [
            ("QE-MAR", "2020-03-31"),
            ("QE-JUN", "2020-03-31"),
            ("QE-DEC", "2020-03-31"),
            ("QS-APR", "2020-04-01"),
            ("2QE-MAR", "2020-03-31"),
        ],
    )
    def test_passes(self, freq, start):
        times = pd.date_range(start, periods=8, freq=freq)
        temporal.check_freq(_da(times), Freq(freq), "x")

    def test_a_different_set_of_quarter_ends_fails(self):
        times = pd.date_range("2020-03-31", periods=8, freq="QE-MAR")
        with pytest.raises(temporal.FreqError):
            temporal.check_freq(_da(times), Freq("QE-NOV"), "x")


_TIMES = pd.date_range("2020-01-01", periods=5, freq="D")


class _NoPandasIndex(xr.Index):
    """A custom index, which `da.indexes` cannot cast to pandas."""

    @classmethod
    def from_variables(cls, variables, *, options):
        return cls()


class _CountingBackend(BackendArray):
    """A lazily read array that counts how often the backend is read."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads = 0

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._read
        )

    def _read(self, key):
        self.reads += 1
        return self.values[key]


def _lazy_variable(dim, values):
    backend = _CountingBackend(values)
    data = indexing.LazilyIndexedArray(backend)
    return xr.Variable((dim,), data), backend  # pyright: ignore[reportArgumentType]


def _with_lazy_coord(da, name, variable):
    """Attach an unindexed coord, keeping its backend.

    The `DataArray` constructor deep-copies a coord's backend, so a counter on the
    original would never move.
    """
    return da.assign_coords(xr.Coordinates({name: variable}, indexes={}))


def _custom_index_axis():
    time, backend = _lazy_variable("time", _TIMES.values)
    da = _with_lazy_coord(xr.DataArray(np.zeros(5), dims="time"), "time", time)
    return da.set_xindex("time", _NoPandasIndex), backend


class TestOnlyAnIndexedCoordIsATimeAxis:
    def test_scalar_datetime_coord_is_ignored(self):
        da = _da(_TIMES).assign_coords(ref=np.datetime64("2000-01-01"))
        assert temporal.check_freq(da, Freq("D"), "x") is da

    def test_2d_datetime_coord_is_ignored(self):
        valid = np.tile(_TIMES.values[:, None], (1, 2))
        da = xr.DataArray(
            np.zeros((5, 2)),
            dims=("time", "step"),
            coords={"time": _TIMES, "valid": (("time", "step"), valid)},
        )
        assert temporal.check_freq(da, Freq("D"), "x") is da

    def test_unindexed_aux_coord_on_the_same_dim_is_ignored(self):
        da = _da(_TIMES, extra_coords={"valid": _TIMES + pd.Timedelta(days=1)})
        assert temporal.check_freq(da, Freq("D"), "x") is da

    def test_wrong_frequency_is_still_caught_beside_a_scalar(self):
        da = _da(_TIMES).assign_coords(ref=np.datetime64("2000-01-01"))
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, Freq("ME"), "x")

    @pytest.mark.parametrize("dim", [None, "valid"])
    def test_unindexed_axis_is_a_mismatch(self, dim):
        da = xr.DataArray(np.zeros(5), dims="s", coords={"valid": ("s", _TIMES)})
        with pytest.raises(temporal.FreqError, match="indexed datetime"):
            temporal.check_freq(da, Freq("D", dim=dim), "x")

    def test_unindexed_cftime_is_treated_as_unindexed_datetime64(self):
        times = xr.date_range(
            "2020-01-01", periods=5, freq="D", calendar="noleap", use_cftime=True
        )
        da = _da(times).drop_indexes("time")
        with pytest.raises(temporal.FreqError, match="no indexed datetime coordinate"):
            temporal.check_freq(da, Freq("D"), "x")

    @pytest.mark.parametrize("dim", [None, "time"])
    def test_custom_index_on_another_coord(self, dim):
        da = xr.DataArray(
            np.zeros((5, 2)),
            dims=("time", "s"),
            coords={"time": _TIMES, "lat": ("s", [1.0, 2.0]), "lon": ("s", [3.0, 4.0])},
        ).set_xindex(("lat", "lon"), _NoPandasIndex)
        assert temporal.check_freq(da, Freq("D", dim=dim), "x") is da

    def test_a_multiindex_level_counts(self):
        da = xr.DataArray(
            np.zeros(5),
            dims="s",
            coords={"time": ("s", _TIMES), "n": ("s", np.arange(5))},
        ).set_index(s=["time", "n"])
        assert temporal.check_freq(da, Freq("D", dim="time"), "x") is da


class TestNothingIsReadFromDisk:
    def test_lazy_data_and_aux_coord_with_an_indexed_axis(self):
        data, data_backend = _lazy_variable("time", np.zeros(5))
        aux, aux_backend = _lazy_variable("time", _TIMES.values)
        da = xr.DataArray(data, coords={"time": _TIMES})
        da = _with_lazy_coord(da, "valid", aux)
        assert temporal.check_freq(da, Freq("D"), "x") is da
        assert data_backend.reads == 0
        assert aux_backend.reads == 0
        da.compute()
        assert data_backend.reads > 0
        assert aux_backend.reads > 0

    def test_lazy_unindexed_axis_is_refused_unread(self):
        aux, aux_backend = _lazy_variable("s", _TIMES.values)
        da = _with_lazy_coord(xr.DataArray(np.zeros(5), dims="s"), "valid", aux)
        with pytest.raises(temporal.FreqError, match="no indexed datetime coordinate"):
            temporal.check_freq(da, Freq("D"), "x")
        assert aux_backend.reads == 0
        da.compute()
        assert aux_backend.reads > 0

    @pytest.mark.parametrize("dim", [None, "time"])
    def test_custom_index_on_the_axis_is_refused_unread(self, dim):
        da, backend = _custom_index_axis()
        with pytest.raises(temporal.FreqError, match="indexed datetime"):
            temporal.check_freq(da, Freq("D", dim=dim), "x")
        assert backend.reads == 0
        da.compute()
        assert backend.reads > 0

    def test_replacing_a_custom_index_reads_once_then_the_check_reads_nothing(self):
        da, backend = _custom_index_axis()
        indexed = da.drop_indexes("time").set_xindex("time")
        reads = backend.reads
        assert reads > 0
        assert temporal.check_freq(indexed, Freq("D"), "x") is indexed
        assert backend.reads == reads
