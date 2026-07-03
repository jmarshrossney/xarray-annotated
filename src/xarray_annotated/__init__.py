"""xarray-annotated: validate xarray DataArray properties declared via ``typing.Annotated``.

Annotation is the unifying technology: a declared property is read off an
``Annotated[DataArray, ...]`` hint and validated at runtime.  Each kind of
property is a domain subpackage, imported explicitly:

* ``xarray_annotated.units`` — physical units (pint / CF), the mature domain.
* ``xarray_annotated.schema`` — structural properties (dims, coords, dtype);
  validate-only (never mutates).

The top level is deliberately thin — nothing domain-specific is re-exported
here, so the domains never collide in a shared namespace::

    from xarray_annotated import schema, units
    from xarray_annotated.units import declare_units, check_units
    from xarray_annotated.schema import declare_schema, Dims, Dtype

The only names surfaced at the top level are domain-*agnostic* helpers that
belong to no single domain: ``annotate`` (the shared declaration writer, inverse
of each domain's ``*_from_signature`` reader) and the ``Annotated`` introspection
kernel (``unwrap_annotated``).  No domain-specific name is re-exported here.
"""

from . import schema, units
from ._writer import annotate

__all__ = ["annotate", "schema", "units"]
