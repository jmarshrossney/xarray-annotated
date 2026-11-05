"""Whole-declaration checks, for `enforce` and for tools built on declarations.

A `Declared` is compared here as one value, against an array or another `Declared`,
so a tool such as a pipeline's pre-flight applies the rules `enforce` applies rather
than a copy of them.  Walking a graph, and deciding which declarations to compare, is
the tool's job.

Like `Declared`, this is cross-domain, so it lives at the package root.
"""

import types
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import xarray as xr

from ._config import resolve_enabled
from ._errors import ContractError
from ._reader import Declared
from .schema._check import (
    _check_valid_schema,
    assert_valid_schema,
    dims_compatible,
    dtype_compatible,
)
from .temporal._check import _check_valid_freq, assert_valid_freq, freq_compatible
from .units._check import (
    _check_input_unit,
    assert_valid_unit,
    units_compatible,
    units_equal,
)

__all__ = [
    "assert_valid_declaration",
    "check_declared",
    "declaration_conflicts",
    "declaration_violations",
]


def assert_valid_declaration(declared: Declared, context: str) -> None:
    """Raise `ValueError` if any part of a declaration is malformed.

    This is what `enforce` runs at decoration, for a tool that wants to fail as early
    on a declaration it built or read, with no array in hand.

    Args:
        declared: The declaration to validate.
        context: Names the offending site in the error message, such as
            `"node 'x'"`.

    Raises:
        ValueError: If a marker is malformed, such as an unrecognised unit, dtype or
            frequency.

    Examples:
        >>> import xarray_annotated as xra
        >>> xra.assert_valid_declaration(xra.Declared(unit=xra.Unit("Pa")), "node 'x'")
        >>> xra.assert_valid_declaration(xra.Declared(dims=xra.Dims("x", "x")), "n")
        Traceback (most recent call last):
        ...
        ValueError: n: ...
    """
    if declared.unit is not None:
        assert_valid_unit(declared.unit, context)
    for marker in declared.schema:
        assert_valid_schema(marker, context)
    if declared.freq is not None:
        assert_valid_freq(declared.freq, context)


def _refuse_other_value(value: Any, decl: Declared, where: str) -> None:
    """Raise unless the hint allows this value, which is not a `DataArray`.

    Raises:
        TypeError: If the hint names nothing that could be this value.
    """
    if value is None:
        if types.NoneType not in decl.others:
            raise TypeError(
                f"{where}: declared a DataArray, got None. If that is allowed, declare "
                f"it with `| None`."
            )
    elif all(other is types.NoneType for other in decl.others):
        raise TypeError(
            f"{where}: declared a DataArray, got {type(value).__name__}. Checks apply "
            f"only to a DataArray, so this value could not be checked."
        )


def _input_label(name: str, qualname: str | None) -> str:
    """Name an input in a message."""
    return f"[{qualname}] input {name!r}" if qualname else f"input {name!r}"


def _facet_checks(
    value: xr.DataArray, declared: Declared, name: str, qualname: str | None
) -> Iterator[Callable[[], Any]]:
    """Yield one check per declared marker: units, then structure, then frequency."""
    if declared.unit is not None:
        yield partial(_check_input_unit, value, declared.unit, name, qualname)
    for marker in declared.schema:
        yield partial(_check_valid_schema, value, marker, name, qualname)
    if declared.freq is not None:
        yield partial(_check_valid_freq, value, declared.freq, name, qualname)


def _check_valid_declared(
    value: Any, declared: Declared, name: str, qualname: str | None
) -> None:
    """`check_declared` on a validated declaration, whatever `enabled` says."""
    if not isinstance(value, xr.DataArray):
        _refuse_other_value(value, declared, _input_label(name, qualname))
        return
    for check in _facet_checks(value, declared, name, qualname):
        check()


def check_declared(
    value: Any, declared: Declared, name: str, qualname: str | None = None
) -> Any:
    """Check an input against everything declared for it; return it unchanged.

    This is the check `enforce` makes on each declared input, so a tool calling it
    before a run gets the verdict the run would.  The declaration is validated first,
    as `enforce` does at decoration.  Units are checked next, then structure, then
    frequency, and the first failure raises.  Only metadata and indexes are read, so a
    lazily opened array stays unread.

    A value that is not a `DataArray` raises unless `declared.others` allows it, as
    `| None` allows `None`.  When xarray-annotated is switched off (`xra.enabled(False)`
    or `XARRAY_ANNOTATED_ENABLED=false`) this is a total no-op.

    Args:
        value: The value to check.
        declared: Its declaration, as read by `declaration_from_hint` or
            `declarations_from_signature`.
        name: The name to report it under.
        qualname: The qualified name of the function it is for, if any.

    Returns:
        `value`, unchanged.

    Raises:
        TypeError: If `value` is not a `DataArray` and `declared` does not allow it.
        ValueError: If the declaration is malformed.
        UnitError: If the unit differs, or is missing or unparseable.
        SchemaError: If the structure does not match.
        FreqError: If the frequency does not match.

    Examples:
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> declared = xra.Declared(unit=xra.Unit("Pa"), dims=xra.Dims("x"))
        >>> p = xr.DataArray([1.0], dims="x", attrs={"units": "Pa"})
        >>> xra.check_declared(p, declared, "p") is p
        True
    """
    if not resolve_enabled():
        return value
    assert_valid_declaration(declared, _input_label(name, qualname))
    _check_valid_declared(value, declared, name, qualname)
    return value


def declaration_violations(
    value: Any, declared: Declared, name: str, qualname: str | None = None
) -> list[str]:
    """Return every way an input fails its declaration, without raising for it.

    This is the array counterpart of `declaration_conflicts`, for a tool reporting all
    that is wrong at once.  It applies the checks `check_declared` applies, and differs
    only in collecting one message per failing marker rather than raising at the first.

    To check one array against several consumers, call this once per consumer and
    de-duplicate by message.  `Declared` equality ignores `others`, so `| None` on one
    consumer would otherwise excuse `None` for another.

    Args:
        value: The value to check.
        declared: Its declaration, as read by `declaration_from_hint` or
            `declarations_from_signature`.
        name: The name to report it under.
        qualname: The qualified name of the function it is for, if any.

    Returns:
        One message per failure, each naming `name`, or an empty list if there are none
        or xarray-annotated is switched off.

    Raises:
        ValueError: If the declaration is malformed.

    Examples:
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> declared = xra.Declared(unit=xra.Unit("Pa"), dims=xra.Dims("y"))
        >>> p = xr.DataArray([1.0], dims="x", attrs={"units": "Pa"})
        >>> len(xra.declaration_violations(p, declared, "p"))
        1
        >>> xra.declaration_violations(p.rename(x="y"), declared, "p")
        []
    """
    if not resolve_enabled():
        return []
    assert_valid_declaration(declared, _input_label(name, qualname))
    if not isinstance(value, xr.DataArray):
        try:
            _refuse_other_value(value, declared, _input_label(name, qualname))
        except TypeError as exc:
            return [str(exc)]
        return []
    violations = []
    for check in _facet_checks(value, declared, name, qualname):
        try:
            check()
        except ContractError as exc:
            violations.append(str(exc))
    return violations


def declaration_conflicts(a: Declared, b: Declared) -> list[str]:
    """Return how two declarations provably cannot describe the same array.

    This is for checking a producer against a consumer with no array in hand, such as
    on an edge of a pipeline's graph.  A facet declared on only one side cannot
    conflict, so a partly annotated pipeline raises no false alarms, and neither can
    `Coords`, since each declaration only requires names to be present.

    Units conflict unless they are equal: `Pa` and `hPa` are compatible, but
    `check_declared` and `enforce` refuse one for the other.

    Compatibility of two markers is not transitive, as a loose `Dims`, `Dtype` or `Freq`
    is compatible with stricter ones that conflict with each other, as `Freq("month")`
    is with `Freq("ME")` and `Freq("MS")`, so compare all pairs of the declarations for
    one array, not each against the first.

    Nor does compatibility prove an edge safe.  `Freq("D", gaps_ok=True)` is compatible
    with `Freq("D")`, which still raises on an axis with holes when the data arrives.

    This reports a fact about the declarations rather than checking anything, so it
    ignores `enabled`, like `units_compatible`.  A tool that should stay quiet when
    switched off can ask `resolve_enabled`.

    Args:
        a: A declaration.
        b: Another declaration.

    Returns:
        One message per conflicting facet, or an empty list if there are none.

    Raises:
        ValueError: If either declaration is malformed.

    Examples:
        >>> import xarray_annotated as xra
        >>> a = xra.Declared(unit=xra.Unit("Pa"), dims=xra.Dims("x"))
        >>> xra.declaration_conflicts(a, xra.Declared(unit=xra.Unit("pascal")))
        []
        >>> xra.declaration_conflicts(a, xra.Declared(unit=xra.Unit("hPa")))
        ["unit: Unit('Pa') and Unit('hPa') are different units"]
    """
    assert_valid_declaration(a, "declaration_conflicts")
    assert_valid_declaration(b, "declaration_conflicts")
    conflicts = []
    if a.unit is not None and b.unit is not None:
        if not units_compatible(a.unit, b.unit):
            conflicts.append(
                f"unit: {a.unit!r} and {b.unit!r} are dimensionally incompatible"
            )
        elif not units_equal(a.unit, b.unit):
            conflicts.append(f"unit: {a.unit!r} and {b.unit!r} are different units")
    if (
        a.dims is not None
        and b.dims is not None
        and not dims_compatible(a.dims, b.dims)
    ):
        conflicts.append(f"dims: {a.dims!r} and {b.dims!r} are incompatible")
    if (
        a.dtype is not None
        and b.dtype is not None
        and not dtype_compatible(a.dtype, b.dtype)
    ):
        conflicts.append(f"dtype: {a.dtype!r} and {b.dtype!r} are incompatible")
    if (
        a.freq is not None
        and b.freq is not None
        and not freq_compatible(a.freq, b.freq)
    ):
        conflicts.append(f"freq: {a.freq!r} and {b.freq!r} are incompatible")
    return conflicts
