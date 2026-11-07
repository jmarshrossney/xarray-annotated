"""Tests for the `xarray_annotated.temporal` frequency-validation domain.

Covers the `Freq` marker and its reader, the spacing/phase model behind
`freq_compatible`, the runtime `check_freq`, the policy, and `check_annotations` over `Freq`
markers.  Frequency validation never mutates: the checks assert/raise/warn but
return arrays unchanged.
"""

import itertools
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
from xarray_annotated.temporal import Freq


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
        assert Freq("W", dim="time").dim == "time"

    def test_keeps_the_raw_spelling(self):
        # Never normalised: "W" must not become "W-SUN" (that is where
        # anchoredness would be lost).
        assert Freq("W").freq == "W"

    def test_repr_roundtrips_eval(self):
        markers = [
            Freq("7D"),
            Freq("W-WED", dim="time"),
            Freq("W"),
            Freq("ME"),
            Freq("QE-MAR", dim="t"),
            Freq("D", gaps_ok=True),
            Freq("3month", dim="t", gaps_ok=True),
        ]
        for marker in markers:
            assert eval(repr(marker)) == marker

    def test_gaps_ok_defaults_to_false_and_is_keyword_only(self):
        assert Freq("D").gaps_ok is False
        assert Freq("D", gaps_ok=True).gaps_ok is True
        assert repr(Freq("D")) == "Freq('D')"
        with pytest.raises(TypeError):
            Freq("D", "time")  # type: ignore[misc]

    def test_any_day_spelling_is_kept(self):
        assert Freq("3month").freq == "3month"

    def test_equality_includes_all_fields(self):
        assert Freq("7D") == Freq("7D")
        assert Freq("7D") != Freq("W-WED")
        assert Freq("W") != Freq("W", dim="time")
        assert Freq("W") != Freq("W", dim="time")
        assert Freq("D") != Freq("D", gaps_ok=True)
        assert hash(Freq("D", gaps_ok=True)) == hash(Freq("D", gaps_ok=True))
        assert Freq("7D") != "7D"  # not equal to a bare string
        assert Freq("7D") != Dims("time")  # nor to another domain's marker

    def test_hashable_in_annotations(self):
        assert hash(Freq("W", dim="t")) == hash(Freq("W", dim="t"))
        assert {Freq("7D"), Freq("W-WED")}
        assert hash(Annotated[xr.DataArray, Freq("7D")])


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
    # The year anchor behaves like the quarter one.
    (Freq("YE-JUN"), Freq("YE-DEC"), False),
    (Freq("YE"), Freq("YE-DEC"), True),
    # An any-day spelling has a month spacing and no phase.
    (Freq("month"), Freq("ME"), True),
    (Freq("month"), Freq("MS"), True),
    (Freq("month"), Freq("month"), True),
    (Freq("month"), Freq("QE"), False),
    (Freq("3month"), Freq("QE"), True),
    (Freq("12month"), Freq("year"), True),
    (Freq("year"), Freq("YE"), True),
    (Freq("year"), Freq("YE-JUN"), True),
    (Freq("2year"), Freq("YE"), False),
    (Freq("month"), Freq("30D"), False),
    # `gaps_ok` is not compared; spacing and phase still are.
    (Freq("D", gaps_ok=True), Freq("D"), True),
    (Freq("D", gaps_ok=True), Freq("2D"), False),
    (Freq("ME", gaps_ok=True), Freq("MS"), False),
    (Freq("month", gaps_ok=True), Freq("ME"), True),
]


def test_the_spelling_alone_says_whether_an_anchor_matters():
    with pytest.raises(TypeError, match="anchored"):
        Freq("W", anchored=True)  # type: ignore[call-arg]


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
        temporal.assert_valid_freq(Freq("QE-MAR"), "ctx")

    @pytest.mark.parametrize("freq", ["month", "3month", "year", "3year"])
    def test_any_day_spellings_pass(self, freq):
        temporal.assert_valid_freq(Freq(freq), "ctx")
        temporal.assert_valid_freq(Freq(freq, gaps_ok=True), "ctx")

    @pytest.mark.parametrize(
        "freq", ["0month", "0year", "months", "Month", "1.5month", "month-JAN"]
    )
    def test_malformed_any_day_spellings(self, freq):
        with pytest.raises(ValueError, match="ctx: invalid frequency"):
            temporal.assert_valid_freq(Freq(freq), "ctx")

    @pytest.mark.parametrize("gaps_ok", [1, "yes", None])
    def test_gaps_ok_must_be_a_bool(self, gaps_ok):
        with pytest.raises(ValueError, match="gaps_ok must be a bool"):
            temporal.assert_valid_freq(Freq("D", gaps_ok=gaps_ok), "ctx")

    @pytest.mark.parametrize("freq", ["B", "-1D", "-1ME"])
    def test_gaps_ok_needs_a_positive_step(self, freq):
        with pytest.raises(ValueError, match="gaps_ok needs a positive step"):
            temporal.assert_valid_freq(Freq(freq, gaps_ok=True), "ctx")
        temporal.assert_valid_freq(Freq(freq), "ctx")


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

    def test_no_datetime_coord_is_a_mismatch(self):
        da = xr.DataArray(np.zeros(3), dims=("time",))
        with pytest.raises(temporal.FreqError, match="no indexed datetime coordinate"):
            temporal.check_freq(da, Freq("D"), "x")

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

    def test_qualname_prefix_in_message(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(temporal.FreqError, match=r"\[myfunc\] 'x'"):
            temporal.check_freq(da, Freq("7D"), "x", qualname="myfunc")

    def test_disabled_is_noop(self):
        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with xra.enabled(False):
            assert temporal.check_freq(da, Freq("ME"), "x") is da  # would raise


# ---------------------------------------------------------------------------
# check_annotations over Freq markers
# ---------------------------------------------------------------------------


class TestDeclareFreq:
    def test_validates_input_and_passes_through(self):
        @xra.check_annotations
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(da) is da
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f(_da(pd.date_range("2020-01-01", periods=5, freq="7D")))

    def test_keyword_and_non_dataarray_args(self):
        @xra.check_annotations
        def f(flag: bool, x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(True, x=da) is da

    def test_optional_none_input_skipped(self):
        @xra.check_annotations
        def f(
            x: Annotated[xr.DataArray, Freq("D")] | None = None,
        ) -> xr.DataArray | None:
            return x

        assert f(None) is None

    def test_declared_default_is_validated(self):
        @xra.check_annotations
        def f(
            x: Annotated[xr.DataArray, Freq("D")] = _da(
                pd.date_range("2020-01-01", periods=5, freq="7D")
            ),
        ) -> xr.DataArray:
            return x

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()  # x left at its non-conforming default

    def test_varargs_and_kwargs_survive_apply_defaults(self):
        @xra.check_annotations
        def f(
            x: Annotated[xr.DataArray, Freq("D")], *rest: int, **kw: int
        ) -> xr.DataArray:
            assert rest == (1, 2)
            assert kw == {"z": 3}
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        assert f(da, 1, 2, z=3) is da

    def test_output_single_validated(self):
        @xra.check_annotations
        def f(n: int) -> Annotated[xr.DataArray, Freq("W-SUN")]:
            # a Wednesday-anchored weekly axis: right spacing, wrong phase
            return _da(pd.date_range("2020-01-01", periods=5, freq="7D"))

        with pytest.raises(temporal.FreqError, match="expected 'W-SUN', got 'W-WED'"):
            f(1)

    def test_output_typeddict_validated(self):
        class Out(TypedDict):
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @xra.check_annotations
        def f() -> Out:
            return {"weekly": _da(pd.date_range("2020-01-01", periods=5, freq="D"))}

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_output_dataclass_validated(self):
        @dataclass
        class Out:
            weekly: Annotated[xr.DataArray, Freq("W-SUN")]

        @xra.check_annotations
        def f() -> Out:
            return Out(weekly=_da(pd.date_range("2020-01-01", periods=5, freq="D")))

        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            f()

    def test_wraps_metadata_preserved(self):
        @xra.check_annotations
        def f(x: Annotated[xr.DataArray, Freq("D")]) -> xr.DataArray:
            """Docstring."""
            return x

        assert f.__name__ == "f"
        assert f.__doc__ == "Docstring."

    def test_disabled_is_total_noop(self):
        @xra.check_annotations
        def f(x: Annotated[xr.DataArray, Freq("ME")]) -> xr.DataArray:
            return x

        da = _da(pd.date_range("2020-01-01", periods=5, freq="D"))
        with xra.enabled(False):
            assert f(da) is da

    def test_fail_fast_bad_input_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @xra.check_annotations
            def f(x: Annotated[xr.DataArray, Freq("M")]) -> xr.DataArray:
                return x

    def test_fail_fast_bad_output_declaration(self):
        with pytest.raises(ValueError, match="invalid frequency"):

            @xra.check_annotations
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

        wrapped = xra.check_annotations(fn)
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


def _track():
    return xr.DataArray(np.zeros(5), dims="obs", coords={"time": ("obs", _TIMES)})


class TestErrorsNameTheFix:
    """The message says what is wrong with the coordinate, and its remedy works."""

    @pytest.mark.parametrize("dim", [None, "time"])
    @pytest.mark.parametrize(
        ("make", "reason", "fix"),
        [
            (_track, r"'time' has no index.*\.set_xindex\('time'\)", "set_xindex"),
            (
                lambda: _custom_index_axis()[0],
                r"'time' has a _NoPandasIndex, not a pandas index"
                r".*\.drop_indexes\('time'\)\.set_xindex\('time'\)",
                "replace",
            ),
        ],
        ids=["unindexed", "custom"],
    )
    def test_the_named_fix_passes(self, make, reason, fix, dim):
        da = make()
        with pytest.raises(temporal.FreqError, match=reason):
            temporal.check_freq(da, Freq("D", dim=dim), "x")
        if fix == "replace":
            da = da.drop_indexes("time")
        fixed = da.set_xindex("time")
        assert temporal.check_freq(fixed, Freq("D", dim=dim), "x") is fixed

    @pytest.mark.parametrize("dim", [None, "time"])
    def test_scalar_names_expand_dims_which_keeps_a_time_dimension(self, dim):
        da = _da(_TIMES).isel(time=0)
        with pytest.raises(temporal.FreqError, match=r"\.expand_dims\('time'\)"):
            temporal.check_freq(da, Freq("D", dim=dim), "x")
        fixed = da.expand_dims("time")
        assert temporal.check_freq(fixed, Freq("D", dim=dim), "x") is fixed

    def test_2d_coord_is_named_as_such(self):
        valid = np.tile(_TIMES.values[:, None], (1, 2))
        da = xr.DataArray(
            np.zeros((5, 2)), dims=("a", "b"), coords={"valid": (("a", "b"), valid)}
        )
        with pytest.raises(temporal.FreqError, match="'valid' is 2-D"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_every_coordinate_is_named(self):
        da = _track().assign_coords(valid=("obs", _TIMES))
        with pytest.raises(temporal.FreqError) as exc:
            temporal.check_freq(da, Freq("D"), "x")
        assert "'time' has no index" in str(exc.value)
        assert "'valid' has no index" in str(exc.value)

    def test_unindexed_cftime_gets_the_generic_wording(self):
        times = xr.date_range(
            "2020-01-01", periods=5, freq="D", calendar="noleap", use_cftime=True
        )
        with pytest.raises(temporal.FreqError, match="unindexed cftime"):
            temporal.check_freq(_da(times).drop_indexes("time"), Freq("D"), "x")

    def test_no_time_at_all_gets_the_generic_wording(self):
        da = xr.DataArray(np.zeros(5), dims="x")
        with pytest.raises(temporal.FreqError, match="undecoded times"):
            temporal.check_freq(da, Freq("D"), "x")

    def test_naming_an_absent_coordinate_lists_the_candidates(self):
        with pytest.raises(temporal.FreqError, match=r"coords: \['time'\]"):
            temporal.check_freq(_da(_TIMES), Freq("D", dim="nope"), "x")

    def test_naming_a_non_datetime_coordinate_lists_the_candidates(self):
        da = _da(_TIMES, extra_coords={"n": np.arange(5)})
        with pytest.raises(temporal.FreqError, match=r"coords: \['time'\]"):
            temporal.check_freq(da, Freq("D", dim="n"), "x")

    def test_building_the_message_reads_nothing(self):
        aux, backend = _lazy_variable("s", _TIMES.values)
        da = _with_lazy_coord(xr.DataArray(np.zeros(5), dims="s"), "valid", aux)
        with pytest.raises(temporal.FreqError, match="'valid' has no index"):
            temporal.check_freq(da, Freq("D"), "x")
        assert backend.reads == 0
        da.compute()
        assert backend.reads > 0


class TestAnAuxCoordCanBeTheAxis:
    def _both(self):
        spaced = pd.date_range("2020-01-01", periods=5, freq="2D")
        return _da(_TIMES, extra_coords={"valid": spaced}).set_xindex("valid")

    def test_indexing_it_makes_the_axis_ambiguous(self):
        with pytest.raises(temporal.FreqError, match="ambiguous"):
            temporal.check_freq(self._both(), Freq("D"), "x")

    def test_naming_it_checks_it(self):
        da = self._both()
        assert temporal.check_freq(da, Freq("2D", dim="valid"), "x") is da
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, Freq("D", dim="valid"), "x")

    def test_naming_the_other_one_checks_that(self):
        da = self._both()
        assert temporal.check_freq(da, Freq("D", dim="time"), "x") is da


def _cftime(start, periods, freq, calendar="360_day"):
    return xr.date_range(
        start, periods=periods, freq=freq, calendar=calendar, use_cftime=True
    )


_DAILY = pd.date_range("2020-01-01", periods=30, freq="D")
_MONTH_ENDS = pd.date_range("2020-01-31", periods=14, freq="ME")


class TestEveryAxisIsTested:
    """An axis either conforms or raises: nothing is left untested."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_no_spacing_to_contradict(self, n):
        da = _da(_DAILY[:n])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert temporal.check_freq(da, Freq("D"), "x") is da

    @pytest.mark.parametrize(
        ("declared", "index"),
        [
            ("D", _DAILY.delete(10)),
            ("D", pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"])),
            ("D", pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03"])),
            ("ME", _MONTH_ENDS.delete(5)),
            ("ME", _MONTH_ENDS - pd.Timedelta(days=1)),
            ("ME", _MONTH_ENDS + pd.to_timedelta(np.arange(14) % 2, unit="h")),
            ("ME", _cftime("2000-01-01", 14, "ME").delete(5)),
        ],
    )
    def test_irregular_spacing_is_a_mismatch(self, declared, index):
        with pytest.raises(temporal.FreqError, match="got irregular spacing"):
            temporal.check_freq(_da(index), Freq(declared), "x")

    @pytest.mark.parametrize(
        "time",
        [
            _MONTH_ENDS + pd.Timedelta(hours=12),
            _MONTH_ENDS + pd.Timedelta(hours=23, minutes=59, seconds=59),
            _MONTH_ENDS.as_unit("s"),
            _cftime("2000-01-01", 14, "ME"),
            _cftime("2000-01-01", 14, "ME", "noleap"),
        ],
    )
    def test_month_ends_conform_at_any_time_of_day_or_calendar(self, time):
        temporal.check_freq(_da(time), Freq("ME"), "x")


class TestTwoPoints:
    @pytest.mark.parametrize(
        ("declared", "first", "second"),
        [
            ("D", "2020-01-01", "2020-01-02"),
            ("24h", "2020-01-01", "2020-01-02"),
            ("7D", "2020-01-01", "2020-01-08"),
            ("W", "2020-01-01", "2020-01-08"),
            ("W-WED", "2020-01-01", "2020-01-08"),
            ("ME", "2020-01-31", "2020-02-29"),
            ("ME", "2020-12-31", "2021-01-31"),
            ("ME", "2020-01-31 12:00", "2020-02-29 12:00"),
            ("MS", "2020-01-01", "2020-02-01"),
            ("QE", "2020-03-31", "2020-06-30"),
            ("QE-DEC", "2020-03-31", "2020-06-30"),
            ("QS-FEB", "2020-02-01", "2020-05-01"),
            ("YE", "2020-12-31", "2021-12-31"),
            ("YE-JUN", "2020-06-30", "2021-06-30"),
            ("B", "2020-01-03", "2020-01-20"),  # no reduction to compare
        ],
    )
    def test_conforming(self, declared, first, second):
        da = _da(pd.to_datetime([first, second]))
        assert temporal.check_freq(da, Freq(declared), "x") is da

    @pytest.mark.parametrize(
        ("declared", "first", "second"),
        [
            ("D", "2020-01-01", "2020-01-06"),
            ("D", "2020-01-01 00:00", "2020-01-01 01:00"),
            ("D", "2020-01-02", "2020-01-01"),
            ("D", "2020-01-01", "2020-01-01"),
            ("W-SUN", "2020-01-01", "2020-01-08"),
            ("ME", "2020-01-31", "2020-03-31"),
            ("ME", "2020-01-30", "2020-02-29"),
            ("ME", "2020-01-31", "2020-02-28"),
            ("ME", "2020-01-31 00:00", "2020-02-29 12:00"),
            ("ME", "2020-02-29", "2020-01-31"),
            ("MS", "2020-01-31", "2020-02-29"),
            ("QE-JAN", "2020-03-31", "2020-06-30"),
            ("YE-JUN", "2020-12-31", "2021-12-31"),
        ],
    )
    def test_contradicting(self, declared, first, second):
        da = _da(pd.to_datetime([first, second]))
        with pytest.raises(temporal.FreqError, match=r"frequency mismatch.* then "):
            temporal.check_freq(da, Freq(declared), "x")

    @pytest.mark.parametrize("calendar", ["360_day", "noleap", "standard"])
    def test_cftime(self, calendar):
        time = _cftime("2000-01-01", 2, "ME", calendar)
        temporal.check_freq(_da(time), Freq("ME"), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(time), Freq("MS"), "x")

    def test_cftime_weekday_anchor_on_a_longer_axis(self):
        # `infer_freq` says "7D" on cftime, where pandas names the weekday.
        monday = _cftime("2024-01-01", 5, "7D", "standard")
        temporal.check_freq(_da(monday), Freq("W-MON"), "x")
        temporal.check_freq(_da(monday), Freq("W"), "x")
        with pytest.raises(temporal.FreqError, match="on another day"):
            temporal.check_freq(_da(monday), Freq("W-SUN"), "x")

    def test_cftime_weekday_anchor(self):
        monday = _cftime("2024-01-01", 2, "7D", "standard")
        temporal.check_freq(_da(monday), Freq("W-MON"), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(monday), Freq("W-SUN"), "x")


CALENDARS = ["standard", "noleap", "360_day"]

_TWO_JANUARIES = pd.DatetimeIndex(
    pd.date_range("2025-01-01", "2025-01-31").append(
        pd.date_range("2026-01-01", "2026-01-31")
    )
)


def _gappy_cftime(calendar):
    """Two runs of daily values a year apart, joined as a user would."""
    runs = [_da(_cftime(f"{year}-01-01", 30, "D", calendar)) for year in (2025, 2026)]
    return xr.concat(runs, "time")


def _midpoints(start, periods, freq, calendar="standard"):
    """Timestamps at the middle of each period, as a climate model stamps its means."""
    edges = _cftime(start, periods + 1, freq, calendar)
    return xr.CFTimeIndex([a + (b - a) / 2 for a, b in itertools.pairwise(edges)])


_GappyDaily = Annotated[xr.DataArray, Freq("D", gaps_ok=True)]


class TestGapsOk:
    """`gaps_ok` declares the sampling period without a gap-free axis (#29)."""

    def test_the_declaration_it_replaces_still_raises(self):
        with pytest.raises(temporal.FreqError, match="got irregular spacing"):
            temporal.check_freq(_da(_TWO_JANUARIES), Freq("D"), "x")

    @pytest.mark.parametrize(
        ("declared", "index"),
        [
            ("D", _TWO_JANUARIES),
            ("D", _TWO_JANUARIES.as_unit("s")),
            ("D", _TWO_JANUARIES + pd.Timedelta(hours=12)),
            ("D", _DAILY),  # a regular axis is a gappy axis with no gaps
            ("24h", _TWO_JANUARIES),
            ("30min", pd.date_range("2020", periods=4, freq="30min").delete(2)),
            ("W-WED", pd.date_range("2020-01-01", periods=9, freq="7D").delete(4)),
            ("7D", pd.date_range("2020-01-01", periods=9, freq="7D").delete(4)),
            ("ME", _MONTH_ENDS.delete(5)),
            ("MS", pd.date_range("2020-01-01", periods=14, freq="MS").delete(5)),
            ("QE-DEC", pd.date_range("2020-03-31", periods=9, freq="QE").delete(4)),
            ("YE-JUN", pd.date_range("2020-06-30", periods=9, freq="YE-JUN").delete(4)),
        ],
    )
    def test_conforming(self, declared, index):
        da = _da(index)
        assert temporal.check_freq(da, Freq(declared, gaps_ok=True), "x") is da

    @pytest.mark.parametrize(
        ("declared", "index", "got"),
        [
            ("D", pd.date_range("2025", periods=9, freq="30min"), " then "),
            ("D", pd.date_range("2025", periods=9, freq="2D"), "no step that short"),
            ("D", pd.date_range("2025", periods=9, freq="7D"), "no step that short"),
            ("D", pd.date_range("2025", periods=9, freq="MS"), "no step that short"),
            ("D", _TWO_JANUARIES.append(_TWO_JANUARIES[-1:]), " then "),  # duplicate
            ("D", _TWO_JANUARIES[::-1], " then "),
            ("D", pd.to_datetime(["2020-01-01", "2020-01-02", "NaT"]), " then "),
            (
                "D",
                _TWO_JANUARIES.append(pd.to_datetime(["2026-02-01 12:00"])),
                " then ",
            ),
            ("W-SUN", pd.date_range("2020-01-01", periods=9, freq="7D"), " then "),
            ("ME", _MONTH_ENDS.delete(5) - pd.Timedelta(days=1), " then "),
            ("ME", pd.date_range("2020-01-01", periods=9, freq="MS"), " then "),
            (
                "ME",
                _MONTH_ENDS.delete(5) + pd.to_timedelta(np.arange(13) % 2, unit="h"),
                " then ",
            ),
            ("ME", _MONTH_ENDS[::3], "no step that short"),
            ("QE-JAN", pd.date_range("2020-03-31", periods=9, freq="QE"), " then "),
            ("YE-JUN", pd.date_range("2020-12-31", periods=9, freq="YE"), " then "),
        ],
    )
    def test_contradicting(self, declared, index, got):
        with pytest.raises(temporal.FreqError, match=f"frequency mismatch.*{got}"):
            temporal.check_freq(_da(index), Freq(declared, gaps_ok=True), "x")

    @pytest.mark.parametrize("n", [0, 1])
    def test_no_spacing_to_contradict(self, n):
        da = _da(_DAILY[:n])
        assert temporal.check_freq(da, Freq("D", gaps_ok=True), "x") is da

    def test_two_points_have_one_step_which_is_the_smallest(self):
        close = pd.to_datetime(["2020-01-01", "2020-01-02"])
        apart = pd.to_datetime(["2020-01-01", "2020-01-03"])
        temporal.check_freq(_da(close), Freq("D", gaps_ok=True), "x")
        with pytest.raises(temporal.FreqError, match="no step that short"):
            temporal.check_freq(_da(apart), Freq("D", gaps_ok=True), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_cftime(self, calendar):
        da = _gappy_cftime(calendar)
        assert isinstance(da.indexes["time"], xr.CFTimeIndex)
        assert temporal.check_freq(da, Freq("D", gaps_ok=True), "x") is da
        with pytest.raises(temporal.FreqError, match="got irregular spacing"):
            temporal.check_freq(da, Freq("D"), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(da, Freq("2D", gaps_ok=True), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_cftime_calendar_spacing(self, calendar):
        time = _cftime("2000-01-01", 14, "ME", calendar).delete(5)
        temporal.check_freq(_da(time), Freq("ME", gaps_ok=True), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(time), Freq("MS", gaps_ok=True), "x")

    def test_cftime_weekday_anchor(self):
        mondays = _cftime("2024-01-01", 9, "7D", "standard").delete(4)
        temporal.check_freq(_da(mondays), Freq("W-MON", gaps_ok=True), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(mondays), Freq("W-SUN", gaps_ok=True), "x")

    def test_a_multiindex_level_counts(self):
        keep = np.arange(5) != 2
        da = xr.DataArray(
            np.zeros(4),
            dims="s",
            coords={"time": ("s", _TIMES[keep]), "n": ("s", np.arange(4))},
        ).set_index(s=["time", "n"])
        declared = Freq("D", dim="time", gaps_ok=True)
        assert temporal.check_freq(da, declared, "x") is da

    @pytest.mark.parametrize("freq", ["B", "-1D"])
    def test_called_directly_it_refuses_what_decoration_refuses(self, freq):
        with pytest.raises(ValueError, match="gaps_ok needs a positive step"):
            temporal.check_freq(_da(_DAILY), Freq(freq, gaps_ok=True), "x")

    def test_disabled_is_noop(self):
        da = _da(pd.date_range("2025", periods=9, freq="7D"))
        with xra.enabled(False):
            assert temporal.check_freq(da, Freq("D", gaps_ok=True), "x") is da

    def test_the_reported_function_through_check_annotations(self):
        @xra.check_annotations
        def mask_snow_days(vwc: _GappyDaily, snow: _GappyDaily) -> _GappyDaily:
            return vwc.where(snow != 1)

        vwc = _da(_TWO_JANUARIES)
        assert mask_snow_days(vwc, xr.zeros_like(vwc)).identical(vwc)
        half_hourly = _da(pd.date_range("2025", periods=48, freq="30min"))
        with pytest.raises(temporal.FreqError, match=r"\[.*mask_snow_days\] 'vwc'"):
            mask_snow_days(half_hourly, xr.zeros_like(half_hourly))

    def test_output_is_checked(self):
        @xra.check_annotations
        def f(da: xr.DataArray) -> Annotated[xr.DataArray, Freq("D", gaps_ok=True)]:
            return da

        f(_da(_TWO_JANUARIES))
        with pytest.raises(temporal.FreqError, match="no step that short"):
            f(_da(pd.date_range("2025", periods=9, freq="7D")))

    def test_decoration_refuses_a_step_it_could_not_compare(self):
        with pytest.raises(ValueError, match="gaps_ok needs a positive step"):

            @xra.check_annotations
            def f(x: Annotated[xr.DataArray, Freq("B", gaps_ok=True)]) -> None: ...


class TestAnyDay:
    """`"month"` and `"year"` fit a calendar axis on no month boundary (#28)."""

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_month_midpoints(self, calendar):
        time = _midpoints("2000-01-01", 26, "MS", calendar)
        da = _da(time)
        assert temporal.check_freq(da, Freq("month"), "x") is da

    @pytest.mark.parametrize(
        "index",
        [
            pd.date_range("2020-01-15", periods=14, freq="MS") + pd.Timedelta(days=14),
            _MONTH_ENDS,
            pd.date_range("2020-01-01", periods=14, freq="MS"),
            pd.to_datetime(
                ["2020-01-16 00:00", "2020-02-15 12:00", "2020-03-16 00:00"]
            ),
            pd.to_datetime(["2020-01-31", "2020-02-01"]),
        ],
    )
    def test_any_day_of_each_month(self, index):
        temporal.check_freq(_da(index), Freq("month"), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_midpoints_still_raise_on_a_convention(self, calendar):
        time = _midpoints("2000-01-01", 26, "MS", calendar)
        for declared in ("MS", "ME"):
            with pytest.raises(temporal.FreqError, match="frequency mismatch"):
                temporal.check_freq(_da(time), Freq(declared), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    @pytest.mark.parametrize(
        "change",
        [lambda t: t.delete(5), lambda t: t.insert(5, t[5]), lambda t: t[::-1]],
        ids=["missing", "duplicated", "reversed"],
    )
    def test_a_month_out_of_place_raises(self, calendar, change):
        time = change(_midpoints("2000-01-01", 26, "MS", calendar))
        with pytest.raises(temporal.FreqError, match=r"frequency mismatch.* then "):
            temporal.check_freq(_da(time), Freq("month"), "x")

    def test_a_daily_axis_is_not_monthly(self):
        with pytest.raises(temporal.FreqError, match=r"frequency mismatch.* then "):
            temporal.check_freq(_da(_DAILY), Freq("month"), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_seasonal_midpoints_are_three_months(self, calendar):
        time = _midpoints("2000-12-01", 9, "QS-DEC", calendar)
        temporal.check_freq(_da(time), Freq("3month"), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(time), Freq("month"), "x")

    @pytest.mark.parametrize("calendar", CALENDARS)
    def test_annual_midpoints(self, calendar):
        time = _midpoints("2000-01-01", 9, "YS", calendar)
        temporal.check_freq(_da(time), Freq("year"), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(time), Freq("2year"), "x")

    def test_fifteenth_of_january_each_year(self):
        index = pd.to_datetime(["2020-01-15", "2021-01-15", "2022-01-15"])
        temporal.check_freq(_da(index), Freq("year"), "x")

    def test_a_year_that_drifts_a_month_raises(self):
        index = pd.to_datetime(["2020-01-15", "2021-02-15", "2022-02-15"])
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(index), Freq("year"), "x")

    @pytest.mark.parametrize("n", [0, 1])
    def test_no_spacing_to_contradict(self, n):
        temporal.check_freq(_da(_DAILY[:n]), Freq("month"), "x")

    def test_two_points(self):
        temporal.check_freq(
            _da(pd.to_datetime(["2020-01-16", "2020-02-15"])), Freq("month"), "x"
        )
        with pytest.raises(temporal.FreqError, match=" then "):
            temporal.check_freq(
                _da(pd.to_datetime(["2020-01-16", "2020-03-16"])), Freq("month"), "x"
            )

    def test_gaps_ok_allows_a_hole_in_months(self):
        index = pd.to_datetime(
            ["2020-12-16", "2021-01-16", "2021-02-15", "2021-12-16", "2022-01-16"]
        )
        temporal.check_freq(_da(index), Freq("month", gaps_ok=True), "x")
        with pytest.raises(temporal.FreqError, match="frequency mismatch"):
            temporal.check_freq(_da(index), Freq("month"), "x")

    def test_januaries_alone_are_yearly_not_gappy_months(self):
        index = pd.to_datetime(["2020-01-16", "2021-01-16", "2022-01-16"])
        temporal.check_freq(_da(index), Freq("year"), "x")
        with pytest.raises(temporal.FreqError, match="no step that short"):
            temporal.check_freq(_da(index), Freq("month", gaps_ok=True), "x")

    def test_through_check_annotations(self):
        @xra.check_annotations
        def f(
            x: Annotated[xr.DataArray, Freq("month")],
        ) -> Annotated[xr.DataArray, Freq("month")]:
            return x

        da = _da(_midpoints("2000-01-01", 26, "MS", "noleap"))
        assert f(da) is da
        with pytest.raises(temporal.FreqError, match="expected 'month'"):
            f(_da(_DAILY))

    def test_decoration_refuses_a_zero_multiple(self):
        with pytest.raises(ValueError, match="invalid frequency '0month'"):

            @xra.check_annotations
            def f(x: Annotated[xr.DataArray, Freq("0month")]) -> None: ...

    def test_legacy_alias_still_names_its_replacement(self):
        with pytest.raises(ValueError, match="ME"):
            temporal.assert_valid_freq(Freq("M"), "ctx")


class TestSteppingReadsNothing:
    @pytest.mark.parametrize(
        ("declared", "index"),
        [
            (Freq("D", gaps_ok=True), _TIMES.delete(2)),
            (Freq("ME", gaps_ok=True), _MONTH_ENDS[:5].delete(2)),
            (Freq("ME"), _MONTH_ENDS[:2]),
            (Freq("month"), _MONTH_ENDS[:4]),
        ],
        ids=["fixed", "calendar", "two-points", "any-day"],
    )
    def test_lazy_data_and_aux_coord_stay_unread(self, declared, index):
        data, data_backend = _lazy_variable("time", np.zeros(len(index)))
        aux, aux_backend = _lazy_variable("time", index.values)
        da = xr.DataArray(data, coords={"time": index})
        da = _with_lazy_coord(da, "valid", aux)
        assert temporal.check_freq(da, declared, "x") is da
        assert data_backend.reads == 0
        assert aux_backend.reads == 0
        da.compute()
        assert data_backend.reads > 0
        assert aux_backend.reads > 0


class TestAnyDayProperty:
    @pytest.mark.parametrize("freq", ["month", "3month", "year", "2year", "0month"])
    def test_calendar_steps_on_any_day(self, freq):
        assert Freq(freq).any_day is True

    @pytest.mark.parametrize("freq", ["ME", "MS", "D", "YE", "months", "bogus", 7])
    def test_everything_else(self, freq):
        assert Freq(freq).any_day is False

    def test_is_not_part_of_the_marker_state(self):
        assert Freq("month") == Freq("month")
        assert repr(Freq("month")) == "Freq('month')"


class TestFreqCompatibleOnDim:
    def test_different_dims_never_conflict(self):
        a, b = Freq("D", dim="a"), Freq("7D", dim="b")
        assert temporal.freq_compatible(a, b)
        assert temporal.freq_compatible(b, a)

    def test_the_same_dim_is_compared(self):
        assert not temporal.freq_compatible(Freq("D", dim="a"), Freq("7D", dim="a"))

    @pytest.mark.parametrize("dim", ["a", None])
    def test_a_dim_on_one_side_is_compared(self, dim):
        a, b = Freq("D", dim=dim), Freq("7D")
        assert not temporal.freq_compatible(a, b)

    def test_an_unparseable_frequency_still_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            temporal.freq_compatible(Freq("bogus", dim="a"), Freq("D", dim="b"))


class TestMalformedFreqInCheckFreq:
    @pytest.mark.parametrize(
        "marker",
        [
            Freq("bogus"),
            Freq("M"),
            Freq("0month"),
            Freq("D", dim=""),
            Freq("B", gaps_ok=True),
        ],
        ids=repr,
    )
    @pytest.mark.parametrize(
        "da",
        [
            xr.DataArray(
                [1.0, 2.0],
                dims="time",
                coords={"time": pd.date_range("2000-01-01", periods=2, freq="D")},
            ),
            xr.DataArray([1.0, 2.0], dims="x"),
        ],
        ids=["time-axis", "no-time-axis"],
    )
    def test_raises_value_error_whatever_the_array(self, marker, da):
        with pytest.raises(ValueError, match="'p'"):
            temporal.check_freq(da, marker, "p")

    def test_switched_off_checks_nothing(self):
        da = xr.DataArray([1.0], dims="x")
        with xra.enabled(False):
            assert temporal.check_freq(da, Freq("bogus"), "p") is da
