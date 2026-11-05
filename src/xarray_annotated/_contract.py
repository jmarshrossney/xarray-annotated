"""Whole-declaration checks, for `enforce` and for tools built on declarations.

A `Declared` is compared here as one value, against an array or another `Declared`,
so a tool such as a pipeline's pre-flight applies the rules `enforce` applies rather
than a copy of them.  Walking a graph, and deciding which declarations to compare, is
the tool's job.

Like `Declared`, this is cross-domain, so it lives at the package root.
"""

import types
from typing import Any

import xarray as xr

from ._config import resolve_enabled
from ._reader import Declared
from .schema._check import (
    assert_valid_schema,
    check_schema,
    dims_compatible,
    dtype_compatible,
)
from .temporal._check import assert_valid_freq, check_freq, freq_compatible
from .units._check import (
    assert_valid_unit,
    check_units,
    units_compatible,
    units_equal,
)

__all__ = ["check_declared", "declaration_conflicts"]


def _assert_valid(declared: Declared, context: str) -> None:
    """Fail fast on a malformed declaration."""
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


def check_declared(
    value: Any, declared: Declared, name: str, qualname: str | None = None
) -> Any:
    """Check an input against everything declared for it; return it unchanged.

    This is the check `enforce` makes on each declared input, so a tool calling it
    before a run gets the verdict the run would.  Units are checked first, then
    structure, then frequency, and the first failure raises.  Only metadata and
    indexes are read, so a lazily opened array stays unread.

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
        ValueError: If the unit differs, or is missing or unparseable.
        pint.DimensionalityError: If the unit has a different dimensionality.
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
    if not isinstance(value, xr.DataArray):
        where = f"input {name!r}"
        _refuse_other_value(
            value, declared, f"[{qualname}] {where}" if qualname else where
        )
        return value
    if declared.unit is not None:
        check_units(value, declared.unit, name, qualname=qualname)
    if markers := declared.schema:
        check_schema(value, markers, name, qualname=qualname)
    if declared.freq is not None:
        check_freq(value, declared.freq, name, qualname=qualname)
    return value


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
    _assert_valid(a, "declaration_conflicts")
    _assert_valid(b, "declaration_conflicts")
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
