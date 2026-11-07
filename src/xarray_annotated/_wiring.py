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
from typing import (
    Annotated,
    Any,
    TypeAliasType,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import xarray as xr

from ._annotations import _is_dataarray_type, _record_return
from ._contract import declaration_conflicts
from ._reader import _QUALIFIERS, Declared, declarations_from_signature

__all__ = ["wiring_conflicts"]

type Upstream = str | tuple[str, str]
type Node = tuple[Callable[..., Any], Mapping[str, Upstream]]

_UNKNOWN = object()

_NAMED = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def _record_fields(return_hint: Any) -> dict[str, Any] | None:
    """Map each field of a return hint's record to its hint, or `None` if none."""
    record, _ = _record_return(return_hint)
    if record is None:
        return None
    hints = get_type_hints(record, include_extras=True)
    if is_typeddict(record):
        return hints
    return {f.name: hints.get(f.name, _UNKNOWN) for f in dataclasses.fields(record)}


def _members(hint: Any, path: frozenset[Any] = frozenset()) -> tuple[Any, ...]:
    """Flatten a hint into the types a value of it can have.

    Reads through `type` aliases, `Annotated`, `TypedDict` qualifiers and unions.  An
    alias that refers to itself flattens to `_UNKNOWN`.
    """
    if isinstance(hint, TypeAliasType):
        return (
            _members(hint.__value__, path | {hint}) if hint not in path else (_UNKNOWN,)
        )
    origin = get_origin(hint)
    if origin is Annotated or origin in _QUALIFIERS:
        return _members(get_args(hint)[0], path)
    if origin in (Union, types.UnionType):
        return tuple(m for arg in get_args(hint) for m in _members(arg, path))
    return (hint,)


def _non_array_types(hint: Any) -> tuple[Any, ...] | None:
    """The types a hint allows, or `None` if it could be a `DataArray` or is unknown."""
    members = _members(hint)
    for member in members:
        if (
            member is _UNKNOWN
            or member is Any
            or member is object
            or isinstance(member, TypeVar)
            or _is_dataarray_type(member)
            or (isinstance(member, type) and issubclass(member, xr.DataArray))
        ):
            return None
    return members


def _refuses(declared: Declared, kinds: tuple[Any, ...]) -> bool:
    """Whether `enforce` would refuse a value of every one of these types."""
    return all(
        types.NoneType not in declared.others
        if kind is types.NoneType
        else not _accepts_other(declared)
        for kind in kinds
    )


def _type_name(kind: Any) -> str:
    """Name a type in a message."""
    return "None" if kind is types.NoneType else getattr(kind, "__name__", repr(kind))


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
    structure is a decision about its body.

    A value whose hint rules out a `DataArray`, such as a node returning `float` or a
    record, or a record field hinted `int`, is reported where it is wired to a
    declared parameter that `enforce` would refuse it for.  A missing or `Any` hint
    says nothing, so is not reported, and nor is a value wired to an undeclared
    parameter, as checking those types is a type checker's job.

    The graph is checked against the functions as it is read, so a renamed parameter,
    field or node raises rather than leaving an edge unchecked, as does a declared
    parameter with no default that is not wired.  Other parameters may be left out of
    the wiring, as for one taking a constant.

    Like `declaration_conflicts`, this reports a fact about declarations, so it ignores
    `enabled`.

    Args:
        graph: Maps each node's name to `(func, wiring)`, where `wiring` maps a
            parameter of `func` to its source: a node's or an input's name, or a
            `(node, field)` pair for one field of a node's record.
        inputs: The names of the values that enter the pipeline from outside it.

    Returns:
        One message per conflicting facet on each pair, in graph order, or an empty
        list if there are none.

    Raises:
        ValueError: If a source names neither a node nor an input, a field the node's
            record does not have, or a field of a node that returns no record, if a
            parameter is not one its function accepts, if a declared parameter with no
            default is not wired, if a name is both a node and an input, or if a
            declaration is malformed.
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
    returns = {
        name: get_type_hints(func, include_extras=True).get("return", _UNKNOWN)
        for name, (func, _) in graph.items()
    }
    fields = {name: _record_fields(hint) for name, hint in returns.items()}
    read = {
        name: declarations_from_signature(func) for name, (func, _) in graph.items()
    }

    uses: dict[Upstream, list[tuple[str, Declared]]] = {}
    for name, (func, wiring) in graph.items():
        consumed, _ = read[name]
        params = inspect.signature(func).parameters
        takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for param, upstream in wiring.items():
            where = f"node {name!r}, parameter {param!r}"
            p = params.get(param)
            if not takes_any and (p is None or p.kind not in _NAMED):
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
        for param in sorted(consumed.keys() - wiring.keys()):
            p = params.get(param)
            if p is not None and p.default is p.empty and p.kind in _NAMED:
                raise ValueError(
                    f"node {name!r}: parameter {param!r} is declared and has no "
                    f"default, but is not wired. Wire it to a node or an input."
                )

    conflicts = []
    for upstream, consumers in uses.items():
        source = _show(upstream)
        if isinstance(upstream, tuple):
            node, field = upstream
            record = read[node][1]
            produced = record.get(field) if isinstance(record, dict) else None
            hint = (fields[node] or {}).get(field, _UNKNOWN)
        else:
            produced = read[upstream][1] if upstream in read else None
            hint = returns.get(upstream, _UNKNOWN)
        if (kinds := _non_array_types(hint)) is not None:
            shown = " | ".join(map(_type_name, kinds))
            conflicts += [
                f"{source} -> {label}: {source} is {shown}, and {label} declares a "
                f"DataArray"
                for label, declared in consumers
                if _refuses(declared, kinds)
            ]
            continue
        reported: dict[str, set[str]] = {}
        if isinstance(produced, Declared):
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
