"""Whole-declaration checks, for `check_annotations` and for tools built on them.

A `Declared` is compared here as one value, against an array or another `Declared`, so a
tool such as a pipeline's pre-flight applies the rules `check_annotations` applies
rather than a copy of them.  Walking a graph, and deciding which declarations to
compare, is the tool's job.

Like `Declared`, this is cross-domain, so it lives at the package root.
"""

import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import xarray as xr

from ._config import resolve_enabled
from ._errors import ContractError, Kind, Role, _label
from ._reader import Declared
from .schema._annotations import Coords, Dims, Dtype
from .schema._check import (
    _check_valid_schema,
    assert_valid_schema,
    dims_compatible,
    dtype_compatible,
)
from .temporal._annotations import Freq
from .temporal._check import _check_valid_freq, assert_valid_freq, freq_compatible
from .units._annotations import Unit
from .units._check import (
    _check_input_unit,
    assert_valid_unit,
    units_compatible,
    units_equal,
)

__all__ = [
    "Conflict",
    "Violation",
    "assert_valid_declaration",
    "check_declared",
    "declaration_conflicts",
    "declaration_violations",
]

type Marker = Unit | Dims | Coords | Dtype | Freq


def _marker_dict(marker: Marker | None) -> dict[str, Any] | None:
    """A marker as JSON-ready data, one key per thing it declares."""
    match marker:
        case None:
            return None
        case Unit():
            return {"unit": marker.unit}
        case Dims():
            return {"names": list(marker.names), "ordered": marker.ordered}
        case Coords():
            return {"names": list(marker.names)}
        case Dtype():
            return {"dtype": marker.dtype}
        case Freq():
            return {"freq": marker.freq, "dim": marker.dim, "gaps_ok": marker.gaps_ok}


@dataclass(frozen=True)
class Violation:
    """One way an input fails its declaration, as `declaration_violations` reports it.

    The fields say what `message` says, for a tool or an agent that should not parse
    prose, and `to_dict` gives them as JSON-ready data.  `str()` gives `message`, so a
    record interpolates as the message did.  `check_annotations` raises a
    `ContractError` with the same fields.

    `facet` is the `Declared` slot that failed, and `kind` says why.  Both are public
    API: a `kind` is added only for a new way to fail, and never renamed.

    | `facet` | `kind` | Meaning |
    | --- | --- | --- |
    | `"unit"` | `"unit_missing"` | The array has no unit |
    | | `"unit_too_long"` | Its unit is too long to parse safely |
    | | `"unit_unparseable"` | Its unit is not one the registry knows |
    | | `"unit_different"` | Its unit converts to the declared one, but is not it |
    | | `"unit_incompatible"` | Its unit cannot convert to the declared one |
    | `"dims"` | `"dims_mismatch"` | It has other dims |
    | | `"dims_order"` | It has the declared dims, in another order |
    | `"coords"` | `"coords_missing"` | It lacks a declared coord |
    | `"dtype"` | `"dtype_kind"` | Its dtype is another kind, such as int for float |
    | | `"dtype_width"` | Its dtype is the declared kind, at another width |
    | `"freq"` | `"freq_no_axis"` | It has no time axis |
    | | `"freq_ambiguous_axis"` | It has several, and the declaration names none |
    | | `"freq_irregular"` | Its time axis has no regular frequency |
    | | `"freq_mismatch"` | Its time axis steps at another frequency |
    | | `"freq_phase"` | It steps as declared, on another weekday or day |
    | `None` | `"not_array"` | The value is not a `DataArray`, and the hint refuses it |

    A `ContractError` from `check_annotations` can also have `kind` `"record_class"`,
    for a returned record of the wrong class, or `"record_field_missing"`.

    `found` is what the array has for the facet, read from metadata and indexes only:

    | `facet` | `found` |
    | --- | --- |
    | `"unit"` | its unit label, or `None` if it has none or it is too long to read |
    | `"dims"` | its dims, in order |
    | `"coords"` | the names of all its coords, sorted |
    | `"dtype"` | its dtype, such as `"float32"` |
    | `"freq"` | the frequency inferred for its time axis, or `None` if none was |
    | `None` | the name of the value's type, such as `"NoneType"` |

    Attributes:
        name: The name the input was checked under.
        qualname: The qualified name of the function it is for, if one was given.
        role: `"input"`, as only inputs are checked before a run.
        facet: The `Declared` slot that failed, or `None` for a value that is not a
            `DataArray`.
        kind: Why it failed, as above.
        declared: The marker that failed, or `None` when `facet` is.
        found: What the array has for that facet, as above.
        hint: What to do about it, as it appears in `message`, or `None` if the
            message suggests nothing.
        message: The message `check_declared` would raise with.
    """

    name: str
    qualname: str | None
    role: Role
    facet: str | None
    kind: Kind | None
    declared: Marker | None
    found: str | tuple[str, ...] | None
    hint: str | None
    message: str

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as JSON-ready data, with each marker as a dict.

        Examples:
            >>> import json
            >>> import xarray as xr
            >>> import xarray_annotated as xra
            >>> p = xr.DataArray([1.0], dims="x", attrs={"units": "Pa"})
            >>> declared = xra.Declared(dims=xra.Dims("y"))
            >>> [violation] = xra.declaration_violations(p, declared, "p")
            >>> record = violation.to_dict()
            >>> record["kind"], record["declared"], record["found"]
            ('dims_mismatch', {'names': ['y'], 'ordered': False}, ['x'])
            >>> _ = json.dumps(record)
        """
        found = list(self.found) if isinstance(self.found, tuple) else self.found
        return {
            "name": self.name,
            "qualname": self.qualname,
            "role": self.role,
            "facet": self.facet,
            "kind": self.kind,
            "declared": _marker_dict(self.declared),
            "found": found,
            "hint": self.hint,
            "message": self.message,
        }


@dataclass(frozen=True)
class Conflict:
    """One facet on which two declarations conflict, as `declaration_conflicts` says.

    `a` and `b` are the two markers in the order the declarations were given, and
    `facet` is their `Declared` slot: `"unit"`, `"dims"`, `"dtype"` or `"freq"`, as
    `Coords` cannot conflict.  `kind` takes its values from those of `Violation`:

    | `kind` | Meaning |
    | --- | --- |
    | `"unit_different"` | The units convert, but are not equal |
    | `"unit_incompatible"` | The units cannot convert |
    | `"dims_mismatch"` | The dims are different sets |
    | `"dims_order"` | Both pin an order, and the orders differ |
    | `"dtype_kind"` | The dtypes are different kinds |
    | `"dtype_width"` | Both name a width of one kind, and the widths differ |
    | `"freq_mismatch"` | The frequencies cannot describe one axis |

    `to_dict` gives the fields as JSON-ready data, and `str()` gives `message`, so a
    record interpolates as the message did.

    Attributes:
        facet: The `Declared` slot the two disagree on.
        kind: Why, as above.
        a: The marker from the first declaration.
        b: The marker from the second.
        message: The conflict in words, starting with `facet`.
    """

    facet: str
    kind: Kind
    a: Marker
    b: Marker
    message: str

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as JSON-ready data, with each marker as a dict.

        Examples:
            >>> import xarray_annotated as xra
            >>> pa = xra.Declared(unit=xra.Unit("Pa"))
            >>> hpa = xra.Declared(unit=xra.Unit("hPa"))
            >>> [conflict] = xra.declaration_conflicts(pa, hpa)
            >>> record = conflict.to_dict()
            >>> record["kind"], record["a"], record["b"]
            ('unit_different', {'unit': 'Pa'}, {'unit': 'hPa'})
        """
        return {
            "facet": self.facet,
            "kind": self.kind,
            "a": _marker_dict(self.a),
            "b": _marker_dict(self.b),
            "message": self.message,
        }


def assert_valid_declaration(declared: Declared, context: str) -> None:
    """Raise `ValueError` if any part of a declaration is malformed.

    This is what `check_annotations` runs at decoration, for a tool that wants to fail
    as early on a declaration it built or read, with no array in hand.

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


def _refuse_other_value(
    value: Any, decl: Declared, name: str, qualname: str | None, role: Role
) -> None:
    """Raise unless the hint allows this value, which is not a `DataArray`.

    Raises:
        ContractError: If the hint names nothing that could be this value.
    """
    where = _label(name, qualname, role)
    site: dict[str, Any] = {"name": name, "qualname": qualname, "role": role}
    if value is None:
        if types.NoneType not in decl.others:
            hint = "If that is allowed, declare it with `| None`."
            raise ContractError(
                f"{where}: declared a DataArray, got None. {hint}",
                **site,
                kind="not_array",
                found="NoneType",
                hint=hint,
            )
    elif all(other is types.NoneType for other in decl.others):
        raise ContractError(
            f"{where}: declared a DataArray, got {type(value).__name__}. Checks apply "
            f"only to a DataArray, so this value could not be checked.",
            **site,
            kind="not_array",
            found=type(value).__name__,
        )


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
        _refuse_other_value(value, declared, name, qualname, "input")
        return
    for check in _facet_checks(value, declared, name, qualname):
        check()


def check_declared(
    value: Any, declared: Declared, name: str, qualname: str | None = None
) -> Any:
    """Check an input against everything declared for it; return it unchanged.

    This is the check `check_annotations` makes on each declared input, so a tool
    calling it before a run gets the verdict the run would.  The declaration is
    validated first, as `check_annotations` does at decoration.  Units are checked next,
    then structure, then frequency, and the first failure raises.  Only metadata and
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
        ContractError: If `value` is not a `DataArray` and `declared` does not allow it.
        ValueError: If the declaration is malformed, or `XARRAY_ANNOTATED_ENABLED` holds
            an unrecognised value.
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
    assert_valid_declaration(declared, _label(name, qualname, "input"))
    _check_valid_declared(value, declared, name, qualname)
    return value


def _violation(exc: ContractError, name: str, qualname: str | None) -> Violation:
    return Violation(
        name=name,
        qualname=qualname,
        role="input",
        facet=exc.facet,
        kind=exc.kind,
        declared=exc.declared,
        found=exc.found,
        hint=exc.hint,
        message=str(exc),
    )


def declaration_violations(
    value: Any, declared: Declared, name: str, qualname: str | None = None
) -> list[Violation]:
    """Return every way an input fails its declaration, without raising for it.

    This is the array counterpart of `declaration_conflicts`, for a tool reporting all
    that is wrong at once.  It applies the checks `check_declared` applies, and differs
    only in collecting one record per failing marker rather than raising at the first.
    Each record's `message` is what `check_declared` would raise with.

    To check one array against several consumers, call this once per consumer and
    de-duplicate, by record or by message.  `Declared` equality ignores `others`, so
    `| None` on one consumer would otherwise excuse `None` for another.

    Args:
        value: The value to check.
        declared: Its declaration, as read by `declaration_from_hint` or
            `declarations_from_signature`.
        name: The name to report it under.
        qualname: The qualified name of the function it is for, if any.

    Returns:
        One `Violation` per failure, units first, or an empty list if there are none or
        xarray-annotated is switched off.

    Raises:
        ValueError: If the declaration is malformed, or `XARRAY_ANNOTATED_ENABLED` holds
            an unrecognised value.

    Examples:
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> declared = xra.Declared(unit=xra.Unit("Pa"), dims=xra.Dims("y"))
        >>> p = xr.DataArray([1.0], dims="x", attrs={"units": "Pa"})
        >>> [violation] = xra.declaration_violations(p, declared, "p")
        >>> violation.facet, violation.declared, violation.found
        ('dims', Dims('y'), ('x',))
        >>> print(violation)
        'p' dims mismatch: expected ('y',) in any order, got ('x',)
        >>> xra.declaration_violations(p.rename(x="y"), declared, "p")
        []
    """
    if not resolve_enabled():
        return []
    assert_valid_declaration(declared, _label(name, qualname, "input"))
    if not isinstance(value, xr.DataArray):
        try:
            _refuse_other_value(value, declared, name, qualname, "input")
        except ContractError as exc:
            return [_violation(exc, name, qualname)]
        return []
    violations = []
    for check in _facet_checks(value, declared, name, qualname):
        try:
            check()
        except ContractError as exc:
            violations.append(_violation(exc, name, qualname))
    return violations


def declaration_conflicts(a: Declared, b: Declared) -> list[Conflict]:
    """Return how two declarations provably cannot describe the same array.

    This is for checking a producer against a consumer with no array in hand, such as
    on an edge of a pipeline's graph.  A facet declared on only one side cannot
    conflict, so a partly annotated pipeline raises no false alarms, and neither can
    `Coords`, since each declaration only requires names to be present.

    Units conflict unless they are equal: `Pa` and `hPa` are compatible, but
    `check_declared` and `check_annotations` refuse one for the other.

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
        One `Conflict` per conflicting facet, in the order `unit`, `dims`, `dtype`,
        `freq`, or an empty list if there are none.

    Raises:
        ValueError: If either declaration is malformed.

    Examples:
        >>> import xarray_annotated as xra
        >>> a = xra.Declared(unit=xra.Unit("Pa"), dims=xra.Dims("x"))
        >>> xra.declaration_conflicts(a, xra.Declared(unit=xra.Unit("pascal")))
        []
        >>> hpa = xra.Declared(unit=xra.Unit("hPa"))
        >>> [conflict] = xra.declaration_conflicts(a, hpa)
        >>> conflict.facet, conflict.a, conflict.b
        ('unit', Unit('Pa'), Unit('hPa'))
        >>> print(conflict)
        unit: Unit('Pa') and Unit('hPa') are different units
    """
    assert_valid_declaration(a, "declaration_conflicts")
    assert_valid_declaration(b, "declaration_conflicts")
    conflicts = []

    def conflict(facet: str, kind: Kind, x: Marker, y: Marker, why: str) -> None:
        message = f"{facet}: {x!r} and {y!r} {why}"
        conflicts.append(Conflict(facet, kind, x, y, message))

    if a.unit is not None and b.unit is not None:
        if not units_compatible(a.unit, b.unit):
            why = "are dimensionally incompatible"
            conflict("unit", "unit_incompatible", a.unit, b.unit, why)
        elif not units_equal(a.unit, b.unit):
            conflict("unit", "unit_different", a.unit, b.unit, "are different units")
    if (
        a.dims is not None
        and b.dims is not None
        and not dims_compatible(a.dims, b.dims)
    ):
        same_set = set(a.dims.names) == set(b.dims.names)
        kind = "dims_order" if same_set else "dims_mismatch"
        conflict("dims", kind, a.dims, b.dims, "are incompatible")
    if (
        a.dtype is not None
        and b.dtype is not None
        and not dtype_compatible(a.dtype, b.dtype)
    ):
        same_kind = np.dtype(a.dtype.dtype).kind == np.dtype(b.dtype.dtype).kind
        kind = "dtype_width" if same_kind else "dtype_kind"
        conflict("dtype", kind, a.dtype, b.dtype, "are incompatible")
    if (
        a.freq is not None
        and b.freq is not None
        and not freq_compatible(a.freq, b.freq)
    ):
        conflict("freq", "freq_mismatch", a.freq, b.freq, "are incompatible")
    return conflicts
