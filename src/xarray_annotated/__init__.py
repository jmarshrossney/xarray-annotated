"""xarray-annotated: validate DataArray properties declared via ``typing.Annotated``.

Annotation is the unifying technology: a declared property is read off an
``Annotated[DataArray, ...]`` hint and enforced at runtime.  Each kind of
property is a domain subpackage, imported explicitly:

* ``xarray_annotated.units`` — physical units (pint / CF).
* ``xarray_annotated.schema`` — structural properties (dims, coords, dtype);
  validate-only (never mutates).
* ``xarray_annotated.temporal`` — the frequency (and phase) of a time axis;
  validate-only.

One decorator, ``enforce``, applies every facet declared on a signature, units first:

    import xarray_annotated as xra
    from xarray_annotated import Dims, Freq, Unit

    @xra.enforce
    def f(p: Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]): ...

The top level is deliberately thin — the domains' checker names stay in their
subpackages.  The exceptions are the markers (``Coords``, ``Dims``, ``Dtype``,
``Freq``, ``Unit``), which are unique across domains and are combined freely in one
hint, and the domain-*agnostic*
names: ``enforce``; the ``enabled`` switch; the shared
declaration *writer* ``annotate`` and its inverse *readers*
``declaration_from_hint`` and ``declarations_from_signature`` (which read every facet
declared on a hint or a signature into one uniform ``Declared`` value); and, for tools
built on those, ``assert_valid_declaration``, ``check_declared``,
``declaration_conflicts``, ``declaration_violations``, ``resolve_enabled`` and
the ``ContractError`` base of every failed check.
"""

from importlib.metadata import version as _version

from . import schema, temporal, units
from ._config import enabled, resolve_enabled
from ._contract import (
    assert_valid_declaration,
    check_declared,
    declaration_conflicts,
    declaration_violations,
)
from ._enforce import enforce
from ._errors import ContractError
from ._reader import Declared, declaration_from_hint, declarations_from_signature
from ._writer import annotate
from .schema import Coords, Dims, Dtype
from .temporal import Freq
from .units import Unit

__version__ = _version("xarray-annotated")

__all__ = [
    "ContractError",
    "Coords",
    "Declared",
    "Dims",
    "Dtype",
    "Freq",
    "Unit",
    "__version__",
    "annotate",
    "assert_valid_declaration",
    "check_declared",
    "declaration_conflicts",
    "declaration_from_hint",
    "declaration_violations",
    "declarations_from_signature",
    "enabled",
    "enforce",
    "resolve_enabled",
    "schema",
    "temporal",
    "units",
]
