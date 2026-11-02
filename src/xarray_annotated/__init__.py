"""xarray-annotated: validate DataArray properties declared via ``typing.Annotated``.

Annotation is the unifying technology: a declared property is read off an
``Annotated[DataArray, ...]`` hint and enforced at runtime.  Each kind of
property is a domain subpackage, imported explicitly:

* ``xarray_annotated.units`` — physical units (pint / CF), the mature domain.
* ``xarray_annotated.schema`` — structural properties (dims, coords, dtype);
  validate-only (never mutates).
* ``xarray_annotated.temporal`` — the frequency (and phase) of a time axis;
  validate-only.

One decorator, ``enforce``, applies every facet declared on a signature, in the
one correct order (units are converted before structure and frequency are
checked)::

    import xarray_annotated as xra
    from xarray_annotated.schema import Dims
    from xarray_annotated.temporal import Freq
    from xarray_annotated.units import Unit

    @xra.enforce
    def f(p: Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]): ...

The top level is deliberately thin — nothing domain-specific is re-exported
here, so the domains never collide in a shared namespace.  The only names
surfaced at the top level are domain-*agnostic*: ``enforce``; the shared
declaration *writer* ``annotate`` and its inverse *reader*
``declarations_from_signature`` (which reads every facet declared on a signature
into one uniform ``Declared`` value); and the ``Annotated`` introspection kernel
(``unwrap_annotated``, ``walk_signature``).  ``walk_signature`` is the shared
driver behind every domain's reader, so a third-party facet author can use it to
build their own ``*_from_signature`` reader.
"""

from importlib.metadata import version as _version

from . import schema, temporal, units
from ._annotations import unwrap_annotated, walk_signature
from ._enforce import enforce
from ._reader import Declared, declarations_from_signature
from ._writer import annotate

__version__ = _version("xarray-annotated")

__all__ = [
    "Declared",
    "__version__",
    "annotate",
    "declarations_from_signature",
    "enforce",
    "schema",
    "temporal",
    "units",
    "unwrap_annotated",
    "walk_signature",
]
