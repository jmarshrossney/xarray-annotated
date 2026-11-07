"""Comparing the declarations along every edge of a pipeline's graph, with no data.

`declaration_conflicts` compares two declarations.  `wiring_conflicts` decides which
pairs to compare, given a graph written out as a mapping, so a plain Python pipeline
can be checked without a graph tool and without calling any function.

Like `Declared`, this is cross-domain, so it lives at the package root.
"""

import dataclasses
import inspect
import itertools
import types
from collections.abc import Callable, Collection, Mapping
from typing import Any, get_type_hints, is_typeddict

from ._annotations import _record_return
from ._contract import declaration_conflicts
from ._reader import Declared, declarations_from_signature

__all__ = ["wiring_conflicts"]

type Upstream = str | tuple[str, str]
type Node = tuple[Callable[..., Any], Mapping[str, Upstream]]


def _record_fields(func: Callable[..., Any]) -> set[str] | None:
    """Return the fields of the record `func` returns, or `None` if it returns none."""
    record, _ = _record_return(get_type_hints(func, include_extras=True).get("return"))
    if record is None:
        return None
    if is_typeddict(record):
        return set(get_type_hints(record))
    return {f.name for f in dataclasses.fields(record)}


def _param_names(func: Callable[..., Any]) -> set[str] | None:
    """Return the names `func` accepts as keywords, or `None` if it takes `**kwargs`."""
    params = inspect.signature(func).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params if p.kind is not p.POSITIONAL_ONLY}


def _show(upstream: Upstream) -> str:
    """Name an upstream value in a message."""
    if isinstance(upstream, tuple):
        return f"{upstream[0]}[{upstream[1]!r}]"
    return upstream


def _facet(conflict: str) -> str:
    """The facet a `declaration_conflicts` message is about, such as `"unit"`."""
    return conflict.split(":", 1)[0]


def _accepts_other(declared: Declared) -> bool:
    """Whether a parameter's union allows something besides a `DataArray` and `None`."""
    return any(other is not types.NoneType for other in declared.others)


def wiring_conflicts(
    graph: Mapping[str, Node], inputs: Collection[str] = ()
) -> list[str]:
    """Return every conflict between declarations along a pipeline's graph.

    Each node names a function and where each of its parameters comes from: another
    node, a field of the `TypedDict` or dataclass another node returns, or one of the
    pipeline's `inputs`.  No function is called and no array is read, so this works on
    any pipeline, however its bodies compute.

    The declarations for one value, that of its producer and those of all its
    consumers, are compared in all pairs with `declaration_conflicts`, since
    compatibility is not transitive.  Two consumers of an undeclared value, or of an
    input, are still compared with each other, as no array could satisfy both if they
    conflict.  Two consumers are not reported for a facet on which either of them
    already conflicts with the producer, as that message would add nothing.

    A node that declares nothing breaks the chain, as whether it preserves a unit or a
    structure is a decision about its body.  A whole record wired to a declared
    `DataArray` parameter is reported, unless that parameter's union allows some other
    type.

    The graph is checked against the functions as it is read, so a renamed parameter,
    field or node raises rather than leaving an edge unchecked.

    Like `declaration_conflicts`, this reports a fact about declarations, so it ignores
    `enabled`.

    Args:
        graph: Maps each node's name to `(func, wiring)`, where `wiring` maps a
            parameter of `func` to its source: a node's or an input's name, or a
            `(node, field)` pair for one field of a node's record.  Parameters not
            wired, such as those taking a constant, are not compared.
        inputs: The names of the values that enter the pipeline from outside it.

    Returns:
        One message per conflicting facet on each pair, in graph order, or an empty
        list if there are none.

    Raises:
        ValueError: If a source names neither a node nor an input, a field the node's
            record does not have, or a field of a node that returns no record, if a
            parameter is not one its function accepts, if a name is both a node and
            an input, or if a declaration is malformed.
        TypeError: If a function has no readable type hints.

    Examples:
        >>> from typing import Annotated
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> def to_pa(
        ...     p: Annotated[xr.DataArray, xra.Unit("hPa")],
        ... ) -> Annotated[xr.DataArray, xra.Unit("Pa")]: ...
        >>> def height(p: Annotated[xr.DataArray, xra.Unit("hPa")]) -> xr.DataArray: ...
        >>> graph = {"p_pa": (to_pa, {"p": "p"}), "z": (height, {"p": "p_pa"})}
        >>> xra.wiring_conflicts(graph, inputs=["p"])
        ["p_pa -> z(p): unit: Unit('Pa') and Unit('hPa') are different units"]
    """
    inputs = set(inputs)
    if both := sorted(inputs & graph.keys()):
        raise ValueError(f"{both} name both a node and an input")
    fields = {name: _record_fields(func) for name, (func, _) in graph.items()}
    read = {
        name: declarations_from_signature(func) for name, (func, _) in graph.items()
    }

    uses: dict[Upstream, list[tuple[str, Declared]]] = {}
    for name, (func, wiring) in graph.items():
        consumed, _ = read[name]
        accepted = _param_names(func)
        for param, upstream in wiring.items():
            where = f"node {name!r}, parameter {param!r}"
            if accepted is not None and param not in accepted:
                raise ValueError(
                    f"{where}: {func.__qualname__} has no parameter {param!r}"
                )
            if isinstance(upstream, tuple):
                node, field = upstream
                if node not in graph:
                    raise ValueError(f"{where}: {node!r} is not a node")
                if (known := fields[node]) is None:
                    raise ValueError(
                        f"{where}: node {node!r} returns no TypedDict or dataclass, "
                        f"so has no field {field!r}"
                    )
                if field not in known:
                    raise ValueError(
                        f"{where}: node {node!r} returns no field {field!r}"
                    )
            elif upstream not in graph and upstream not in inputs:
                raise ValueError(
                    f"{where}: {upstream!r} is neither a node nor an input"
                )
            if param in consumed:
                label = f"{name}({param})"
                uses.setdefault(upstream, []).append((label, consumed[param]))

    conflicts = []
    for upstream, consumers in uses.items():
        source = _show(upstream)
        if isinstance(upstream, tuple):
            record = read[upstream[0]][1]
            produced = record.get(upstream[1]) if isinstance(record, dict) else None
        else:
            produced = read[upstream][1] if upstream in read else None
        if isinstance(produced, dict):
            conflicts += [
                f"{source} -> {label}: {source} returns a record, and {label} "
                f"declares a DataArray"
                for label, declared in consumers
                if not _accepts_other(declared)
            ]
            continue
        reported: dict[str, set[str]] = {}
        if produced is not None:
            for label, declared in consumers:
                found = declaration_conflicts(produced, declared)
                reported[label] = {_facet(c) for c in found}
                conflicts += [f"{source} -> {label}: {c}" for c in found]
        for (a, da), (b, db) in itertools.combinations(consumers, 2):
            seen = reported.get(a, set()) | reported.get(b, set())
            conflicts += [
                f"{a} and {b}, both from {source}: {c}"
                for c in declaration_conflicts(da, db)
                if _facet(c) not in seen
            ]
    return conflicts
