"""Structural-property domain for xarray-annotated (stub).

Declares and (eventually) validates a DataArray's *structural* properties — the
ones every DataArray possesses regardless of physical units: its dimensions
(``Dims``), coordinates (``Coords``), and dtype (``Dtype``).  This is the
structural counterpart to ``xarray_annotated.units``.

Currently a stub: the ``Dims`` / ``Coords`` / ``Dtype`` markers are defined so the
API shape is fixed, but ``check_schema`` raises ``NotImplementedError``.
"""

from ._annotations import Coords, Dims, Dtype
from ._check import check_schema

__all__ = ["Coords", "Dims", "Dtype", "check_schema"]
