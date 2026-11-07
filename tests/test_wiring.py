"""`wiring_conflicts`, which compares declarations along a pipeline's graph."""

from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, TypedDict

import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Dims, Freq, Unit

type Pa = Annotated[xr.DataArray, Unit("Pa")]
type HPa = Annotated[xr.DataArray, Unit("hPa")]


def _hpa_to_pa(p: HPa) -> Pa: ...


def _wants_pa(x: Pa) -> None: ...


def _wants_hpa(x: HPa) -> None: ...


def _undeclared(x: xr.DataArray) -> xr.DataArray: ...


class Stats(TypedDict):
    mean: Pa
    count: xr.DataArray
    spread: NotRequired[HPa]


@dataclass
class StatsRecord:
    mean: Pa
    count: xr.DataArray


def _stats(x: HPa) -> Stats: ...


def _stats_record(x: HPa) -> StatsRecord | None: ...


class TestEdges:
    def test_agreeing_edge(self):
        graph = {"a": (_hpa_to_pa, {"p": "p"}), "b": (_wants_pa, {"x": "a"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_conflicting_edge(self):
        graph = {"a": (_hpa_to_pa, {"p": "p"}), "b": (_wants_hpa, {"x": "a"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == [
            "a -> b(x): unit: Unit('Pa') and Unit('hPa') are different units"
        ]

    def test_reports_every_facet(self):
        def produce() -> Annotated[xr.DataArray, Unit("Pa"), Dims("x")]: ...

        def consume(x: Annotated[xr.DataArray, Unit("K"), Dims("y")]) -> None: ...

        conflicts = xra.wiring_conflicts(
            {"a": (produce, {}), "b": (consume, {"x": "a"})}
        )
        assert [c.split(": ")[1] for c in conflicts] == ["unit", "dims"]

    def test_undeclared_node_breaks_the_chain(self):
        graph = {
            "a": (_hpa_to_pa, {"p": "p"}),
            "b": (_undeclared, {"x": "a"}),
            "c": (_wants_hpa, {"x": "b"}),
        }
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_unwired_parameter_is_not_compared(self):
        def windowed(x: Pa, window: int) -> None: ...

        graph = {"a": (_hpa_to_pa, {"p": "p"}), "b": (windowed, {"x": "a"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_enforced_functions(self):
        a, b = xra.check_annotations(_hpa_to_pa), xra.check_annotations(_wants_hpa)
        graph = {"a": (a, {"p": "p"}), "b": (b, {"x": "a"})}
        assert len(xra.wiring_conflicts(graph, inputs=["p"])) == 1

    def test_ignores_the_switch(self):
        graph = {"a": (_hpa_to_pa, {"p": "p"}), "b": (_wants_hpa, {"x": "a"})}
        with xra.enabled(False):
            assert len(xra.wiring_conflicts(graph, inputs=["p"])) == 1


class TestSeveralConsumers:
    def test_consumers_of_an_input_are_compared(self):
        graph = {"a": (_wants_pa, {"x": "p"}), "b": (_wants_hpa, {"x": "p"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == [
            "a(x) and b(x), both from p: unit: Unit('Pa') and Unit('hPa') are "
            "different units"
        ]

    def test_consumers_of_an_undeclared_node_are_compared(self):
        graph = {
            "u": (_undeclared, {"x": "p"}),
            "a": (_wants_pa, {"x": "u"}),
            "b": (_wants_hpa, {"x": "u"}),
        }
        assert len(xra.wiring_conflicts(graph, inputs=["p"])) == 1

    def test_all_pairs_not_each_against_the_first(self):
        def loose(x: Annotated[xr.DataArray, Freq("month")]) -> None: ...

        def month_end(x: Annotated[xr.DataArray, Freq("ME")]) -> None: ...

        def month_start(x: Annotated[xr.DataArray, Freq("MS")]) -> None: ...

        graph = {
            "a": (loose, {"x": "p"}),
            "b": (month_end, {"x": "p"}),
            "c": (month_start, {"x": "p"}),
        }
        assert [c.split(",")[0] for c in xra.wiring_conflicts(graph, inputs=["p"])] == [
            "b(x) and c(x)"
        ]

    def test_loose_producer_does_not_hide_a_consumer_pair(self):
        def loose() -> Annotated[xr.DataArray, Freq("month")]: ...

        def month_end(x: Annotated[xr.DataArray, Freq("ME")]) -> None: ...

        def month_start(x: Annotated[xr.DataArray, Freq("MS")]) -> None: ...

        graph = {
            "p": (loose, {}),
            "b": (month_end, {"x": "p"}),
            "c": (month_start, {"x": "p"}),
        }
        assert [c.split(",")[0] for c in xra.wiring_conflicts(graph)] == [
            "b(x) and c(x)"
        ]

    def test_pair_not_repeated_after_a_producer_conflict(self):
        graph = {
            "a": (_hpa_to_pa, {"p": "p"}),
            "b": (_wants_hpa, {"x": "a"}),
            "c": (_wants_hpa, {"x": "a"}),
            "d": (_wants_pa, {"x": "a"}),
        }
        assert [c.split(":")[0] for c in xra.wiring_conflicts(graph, inputs=["p"])] == [
            "a -> b(x)",
            "a -> c(x)",
        ]


class TestRecords:
    @pytest.mark.parametrize("producer", [_stats, _stats_record])
    def test_field_is_its_own_producer(self, producer):
        graph = {
            "s": (producer, {"x": "p"}),
            "ok": (_wants_pa, {"x": ("s", "mean")}),
            "bad": (_wants_hpa, {"x": ("s", "mean")}),
        }
        assert xra.wiring_conflicts(graph, inputs=["p"]) == [
            "s['mean'] -> bad(x): unit: Unit('Pa') and Unit('hPa') are different units"
        ]

    def test_undeclared_field(self):
        graph = {"s": (_stats, {"x": "p"}), "a": (_wants_hpa, {"x": ("s", "count")})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_not_required_field(self):
        graph = {"s": (_stats, {"x": "p"}), "a": (_wants_pa, {"x": ("s", "spread")})}
        assert len(xra.wiring_conflicts(graph, inputs=["p"])) == 1

    def test_whole_record_to_a_declared_array(self):
        graph = {"s": (_stats, {"x": "p"}), "a": (_wants_pa, {"x": "s"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == [
            "s -> a(x): s is Stats, and a(x) declares a DataArray"
        ]

    def test_whole_record_to_a_union_that_allows_it(self):
        def either(x: Pa | Stats) -> None: ...

        graph = {"s": (_stats, {"x": "p"}), "a": (either, {"x": "s"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_whole_record_to_an_undeclared_parameter(self):
        def anything(x: Stats) -> None: ...

        graph = {"s": (_stats, {"x": "p"}), "a": (anything, {"x": "s"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []


class TestMalformedGraph:
    @pytest.mark.parametrize(
        ("graph", "match"),
        [
            ({"a": (_wants_pa, {"x": "nowhere"})}, "neither a node nor an input"),
            ({"a": (_wants_pa, {"y": "p"})}, "has no parameter 'y'"),
            ({"a": (_wants_pa, {"x": ("nowhere", "f")})}, "'nowhere' is not a node"),
            (
                {"a": (_hpa_to_pa, {"p": "p"}), "b": (_wants_pa, {"x": ("a", "f")})},
                "returns no TypedDict or dataclass",
            ),
            (
                {"s": (_stats, {"x": "p"}), "b": (_wants_pa, {"x": ("s", "nope")})},
                "returns no field 'nope'",
            ),
        ],
    )
    def test_raises(self, graph, match):
        with pytest.raises(ValueError, match=match):
            xra.wiring_conflicts(graph, inputs=["p"])

    def test_name_is_both_node_and_input(self):
        with pytest.raises(ValueError, match="both a node and an input"):
            xra.wiring_conflicts({"p": (_wants_pa, {})}, inputs=["p"])

    def test_positional_only_parameter_cannot_be_wired(self):
        def positional(x: Pa, /) -> None: ...

        with pytest.raises(ValueError, match="has no parameter 'x'"):
            xra.wiring_conflicts({"a": (positional, {"x": "p"})}, inputs=["p"])

    def test_var_keyword_accepts_any_name(self):
        def kwargs(**kw: xr.DataArray) -> None: ...

        assert xra.wiring_conflicts({"a": (kwargs, {"x": "p"})}, inputs=["p"]) == []

    def test_malformed_declaration(self):
        def bad() -> Annotated[xr.DataArray, Unit("not_a_unit")]: ...

        graph = {"a": (bad, {}), "b": (_wants_pa, {"x": "a"})}
        with pytest.raises(ValueError, match="not_a_unit"):
            xra.wiring_conflicts(graph)


class Counts(TypedDict):
    n: int


def _counts(x: HPa) -> Counts: ...


class TestNonArrayValues:
    @pytest.mark.parametrize(
        ("returned", "shown"),
        [(float, "float"), (xr.Dataset, "Dataset"), (None, "None"), (Stats, "Stats")],
    )
    def test_reported(self, returned, shown):
        def produce(): ...

        produce.__annotations__ = {"return": returned}

        graph = {"n": (produce, {}), "a": (_wants_pa, {"x": "n"})}
        assert xra.wiring_conflicts(graph) == [
            f"n -> a(x): n is {shown}, and a(x) declares a DataArray"
        ]

    def test_record_field(self):
        graph = {"s": (_counts, {"x": "p"}), "a": (_wants_pa, {"x": ("s", "n")})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == [
            "s['n'] -> a(x): s['n'] is int, and a(x) declares a DataArray"
        ]

    def test_union_member_is_named(self):
        def produce() -> float | None: ...

        graph = {"n": (produce, {}), "a": (_wants_pa, {"x": "n"})}
        assert xra.wiring_conflicts(graph) == [
            "n -> a(x): n is float | None, and a(x) declares a DataArray"
        ]

    @pytest.mark.parametrize(
        ("returned", "consumer"),
        [
            (float, Pa | float),
            (None, Pa | None),
            (float | None, Pa | None),
            (float, xr.DataArray),
            (float, int),
        ],
    )
    def test_allowed_by_the_consumer(self, returned, consumer):
        def produce(): ...

        produce.__annotations__ = {"return": returned}

        def consume(x): ...

        consume.__annotations__ = {"x": consumer}

        graph = {"n": (produce, {}), "a": (consume, {"x": "n"})}
        assert xra.wiring_conflicts(graph) == []

    @pytest.mark.parametrize("returned", [Any, object, Pa | float, xr.DataArray])
    def test_hint_allows_an_array(self, returned):
        def produce(): ...

        produce.__annotations__ = {"return": returned}

        graph = {"n": (produce, {}), "a": (_wants_pa, {"x": "n"})}
        assert xra.wiring_conflicts(graph) == []

    def test_no_return_hint(self):
        def produce(): ...

        graph = {"n": (produce, {}), "a": (_wants_pa, {"x": "n"})}
        assert xra.wiring_conflicts(graph) == []


class TestRequiredParameters:
    def test_declared_and_unwired_raises(self):
        def two(x: Pa, y: Pa) -> None: ...

        with pytest.raises(ValueError, match="parameter 'y' is declared and has no"):
            xra.wiring_conflicts({"a": (two, {"x": "p"})}, inputs=["p"])

    def test_declared_with_a_default(self):
        def mask_optional(x: Pa, mask: Pa | None = None) -> None: ...

        graph = {"a": (mask_optional, {"x": "p"})}
        assert xra.wiring_conflicts(graph, inputs=["p"]) == []

    def test_undeclared_and_unwired(self):
        def windowed(x: Pa, window: int, raw: xr.DataArray) -> None: ...

        assert xra.wiring_conflicts({"a": (windowed, {"x": "p"})}, inputs=["p"]) == []
