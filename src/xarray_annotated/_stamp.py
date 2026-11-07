"""`stamp_output_units`: state each declared output unit on a returned array.

An input unit declaration is checked against the label the caller's data carries.
An output one is stated by the developer, since nothing outside a body can see what
it did.  This decorator states it: it fills each missing output label from the
declaration, and replaces a present one only where `overwrite` names it.

It is an operation, like `units.convert`, so `enabled(False)` does not affect it.
Comparing labels is left to `check_annotations`, which sits above it.  See
`docs/adr/0004-output-units-are-stated-and-stamping.md`.
"""

import copy
import functools
import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any, get_type_hints, is_typeddict, overload

import numpy as np
import xarray as xr

from ._annotations import _record_return
from ._check_annotations import _CHECKED
from ._reader import Declared, declarations_from_signature
from .units._check import _quantified_units, _stamped, assert_valid_unit

__all__ = ["stamp_output_units"]

_log = logging.getLogger("xarray_annotated.units")


def _stamp_one(value: Any, unit: str, overwrite: bool, where: str) -> Any:
    """Label one returned value `unit`, if it is a plain array that needs it."""
    if not isinstance(value, xr.DataArray) or _quantified_units(value) is not None:
        return value
    have = value.attrs.get("units")
    if have is None:
        return _stamped(value, unit)
    if overwrite and have != unit:
        _log.debug("overwriting the unit of %s: %r -> %r", where, have, unit)
        return _stamped(value, unit)
    return value


def _overwritten_fields(
    overwrite: bool | np.bool_ | Iterable[str],
    units: dict[str, str],
    record: Any,
    where: str,
) -> frozenset[str]:
    """Resolve `overwrite` to the fields it names, refusing what it cannot apply to.

    Raises:
        ValueError: If `overwrite` is neither a bool nor an iterable of names, is a
            bare string, names a field that is not one of the record's, or names one
            that declares no unit.
    """
    if isinstance(overwrite, bool | np.bool_):
        return frozenset(units) if overwrite else frozenset()
    if isinstance(overwrite, str):
        raise ValueError(
            f"{where}: overwrite={overwrite!r} is a bare string. Pass a tuple of field "
            f"names, as overwrite=({overwrite!r},), or True for every output."
        )
    if not isinstance(overwrite, Iterable):
        raise ValueError(
            f"{where}: overwrite={overwrite!r} is neither a bool nor field names. "
            f"Pass True for every output, or a tuple of field names."
        )
    names = frozenset(overwrite)
    if names and record is None:
        raise ValueError(
            f"{where}: overwrite names fields {sorted(names)!r}, but the return is "
            f"one DataArray. Pass overwrite=True."
        )
    fields = set(get_type_hints(record, include_extras=True)) if record else set()
    if unknown := sorted(names - fields):
        raise ValueError(
            f"{where}: overwrite names {unknown!r}, which are not fields of "
            f"{record.__name__}."
        )
    if unlabelled := sorted(names - set(units)):
        raise ValueError(
            f"{where}: overwrite names {unlabelled!r}, which declare no Unit to "
            f"overwrite with."
        )
    return names


def _stamp_record(
    result: Any, cls: Any, units: dict[str, str], overwrite: frozenset[str], where: str
) -> Any:
    """Stamp the declared fields of a returned record, on a shallow copy if any."""
    is_dict = is_typeddict(cls)
    if not isinstance(result, dict if is_dict else cls):
        return result
    stamped = {}
    for name, unit in units.items():
        if is_dict and name not in result:
            continue
        value = result[name] if is_dict else getattr(result, name, None)
        new = _stamp_one(value, unit, name in overwrite, f"{where} output {name!r}")
        if new is not value:
            stamped[name] = new
    if not stamped:
        return result
    if is_dict:
        return {**result, **stamped}
    # A copy skips `__init__` and `__post_init__`, and works on a frozen class.
    record = copy.copy(result)
    for name, value in stamped.items():
        object.__setattr__(record, name, value)
    return record


def _declared_units(output: Any) -> dict[str, str] | str | None:
    """The declared output unit, or one per record field that declares one."""
    if isinstance(output, Declared):
        return None if output.unit is None else output.unit.unit
    if isinstance(output, dict):
        return {n: d.unit.unit for n, d in output.items() if d.unit is not None}
    return None


def _decorate(
    func: Callable[..., Any], overwrite: bool | np.bool_ | Iterable[str]
) -> Any:
    qualname = getattr(func, "__qualname__", repr(func))
    where = f"stamp_output_units on {qualname}"
    if getattr(func, _CHECKED, False):
        raise ValueError(
            f"{where}: it is applied above check_annotations, so it would run after "
            f"the checks. Place @stamp_output_units below @check_annotations."
        )
    _, output = declarations_from_signature(func)
    units = _declared_units(output)
    if not units:
        raise ValueError(
            f"{where}: the return declares no Unit, so there is nothing to stamp."
        )
    hints = get_type_hints(func, include_extras=True)
    record, record_optional = _record_return(hints.get("return"))
    named = units if isinstance(units, dict) else {"return": units}
    for name, unit in named.items():
        assert_valid_unit(unit, f"[{qualname}] output {name!r}")
    fields = _overwritten_fields(overwrite, named, record, where)

    def _stamp(result: Any) -> Any:
        if isinstance(units, str):
            return _stamp_one(result, units, bool(fields), f"[{qualname}] output")
        if result is None and record_optional:
            return result
        return _stamp_record(result, record, units, fields, f"[{qualname}]")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _stamp(func(*args, **kwargs))

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _stamp(await func(*args, **kwargs))

    chosen = async_wrapper if inspect.iscoroutinefunction(func) else wrapper
    chosen.__annotations__ = dict(func.__annotations__)
    return chosen


@overload
def stamp_output_units(func: Callable[..., Any], /) -> Callable[..., Any]: ...


@overload
def stamp_output_units(
    *, overwrite: bool | np.bool_ | Iterable[str] = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def stamp_output_units(
    func: Callable[..., Any] | None = None,
    /,
    *,
    overwrite: bool | np.bool_ | Iterable[str] = False,
) -> Any:
    """State each declared output unit on the array a function returns.

    For a body that builds its outputs from bare numbers, such as a wrapper around a
    library working on NumPy arrays, where the declaration is the only statement of
    the unit there can be.  Place it below `check_annotations`, which then checks
    that every output label matches its declaration:

        @xra.check_annotations
        @xra.stamp_output_units
        def f(...) -> Annotated[xr.DataArray, Unit("mm / day")]: ...

    Leave it off a body that computes with pint and dequantifies, so that a unit pint
    lost reaches `check_annotations` as a missing label and raises.

    On each call, for the return or each field of a returned `TypedDict` or
    dataclass that declares a `Unit`:

    * a plain array with no `units` attribute is labelled with the declared unit;
    * a plain array with one keeps it, unless `overwrite` covers it, so a stale or
      copied label still reaches `check_annotations`;
    * a quantified array is left alone, as its unit comes from pint's arithmetic;
    * anything else, such as `None`, is left for `check_annotations` to judge.

    A stamped array is a shallow copy sharing its data, and a record with a stamped
    field is a shallow copy too, made without running `__init__` or `__post_init__`.
    Values are never read, so a lazily opened array stays lazy.

    This is not a check, so `enabled(False)` does not affect it: switching off
    removes checks, and the labels later code reads are still stated.

    Args:
        func: The function to decorate, when applied bare as
            `@xra.stamp_output_units`.
        overwrite: Which present labels to replace with the declared unit: `True`
            for every output, or a tuple or list of the record fields to replace.
            Each replacement logs at DEBUG on the `xarray_annotated.units` logger.

    Returns:
        The wrapped function, or a decorator producing it.

    Raises:
        ValueError: At decoration, if the return declares no `Unit`, a declared unit
            is malformed, the function is already wrapped by `check_annotations`, or
            `overwrite` is a bare string, names fields on a single-array return, or
            names a field that is not one of the returned record's or declares no
            `Unit`.

    Example:
        >>> from typing import Annotated
        >>> import numpy as np
        >>> import xarray as xr
        >>> import xarray_annotated as xra
        >>> from xarray_annotated import Unit
        >>> @xra.check_annotations
        ... @xra.stamp_output_units
        ... def depth(n: int) -> Annotated[xr.DataArray, Unit("mm")]:
        ...     return xr.DataArray(np.zeros(n), dims="x")
        >>> depth(2).attrs["units"]
        'mm'
    """
    if func is None:
        return lambda f: _decorate(f, overwrite)
    return _decorate(func, overwrite)
