"""xarray-annotated: validate xarray DataArray properties declared via ``typing.Annotated``.

Annotation is the unifying technology: a declared property is read off an
``Annotated[DataArray, ...]`` hint and validated at runtime.  Each kind of
property is a domain subpackage, imported explicitly:

* ``xarray_annotated.units`` — physical units (pint / CF), the mature domain.
* ``xarray_annotated.schema`` — structural properties (dims, coords, dtype).
  Currently a stub reserving the API shape.

The top level is deliberately thin — nothing domain-specific is re-exported
here, so the domains never collide in a shared namespace::

    from xarray_annotated import units
    from xarray_annotated.units import declare_units, check_units
"""

from . import units

__all__ = ["units"]
