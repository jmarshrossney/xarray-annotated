"""xarray-annotated: validate xarray DataArray properties declared via ``typing.Annotated``.

Annotation is the unifying technology: a declared property is read off an
``Annotated[DataArray, ...]`` hint and validated at runtime.  Each kind of
property is a domain subpackage, imported explicitly:

* ``xarray_annotated.units`` — physical units (pint / CF), the mature domain.
* ``xarray_annotated.schema`` — structural properties (dims, coords, dtype);
  validate-only (never mutates).
* ``xarray_annotated.temporal`` — the frequency (and phase) of a time axis;
  validate-only.

The top level is deliberately thin — nothing domain-specific is re-exported
here, so the domains never collide in a shared namespace::

    from xarray_annotated import schema, temporal, units
    from xarray_annotated.units import declare_units, check_units
    from xarray_annotated.schema import declare_schema, Dims, Dtype
    from xarray_annotated.temporal import declare_freq, Freq

The only names surfaced at the top level are domain-*agnostic* helpers that
belong to no single domain: the shared declaration *writer* ``annotate`` and its
inverse *reader* ``declarations_from_signature`` (which reads every facet declared
on a signature into one uniform ``Declared`` value), plus the ``Annotated``
introspection kernel (``unwrap_annotated``, ``walk_signature``).
``walk_signature`` is the shared driver behind every domain's reader, so a
third-party facet author can use it to build their own ``*_from_signature``
reader.  No domain-specific name is re-exported here.
"""

from . import schema, temporal, units
from ._annotations import unwrap_annotated, walk_signature
from ._reader import Declared, declarations_from_signature
from ._writer import annotate

__all__ = [
    "Declared",
    "annotate",
    "declarations_from_signature",
    "schema",
    "temporal",
    "units",
    "unwrap_annotated",
    "walk_signature",
]
