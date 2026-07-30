"""Tests for the shape of the public API surface.

Each domain declares its public names in ``__all__``, and the API reference is
generated from those modules, so a name that is documented-but-unexported is
invisible to users while still looking public in the source. These tests guard
that boundary in both directions: everything promised is importable, and nothing
internal leaks.
"""

import importlib

import pytest

import xarray_annotated as xa

DOMAINS = ("units", "schema", "temporal")
MODULES = ("xarray_annotated", *(f"xarray_annotated.{d}" for d in DOMAINS))


@pytest.mark.parametrize("modname", MODULES)
class TestAllIsHonest:
    def test_every_exported_name_resolves(self, modname):
        mod = importlib.import_module(modname)
        missing = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert not missing, f"{modname}.__all__ names nothing: {missing}"

    def test_no_private_names_exported(self, modname):
        mod = importlib.import_module(modname)
        private = [name for name in mod.__all__ if name.startswith("_")]
        assert not private, f"{modname} exports private names: {private}"

    def test_all_is_sorted_and_unique(self, modname):
        mod = importlib.import_module(modname)
        assert list(mod.__all__) == sorted(set(mod.__all__))


class TestReadersAreExportedConsistently:
    """Every domain has a signature reader *and* a single-hint reader; a facet
    author imitating one domain must find the same surface in the others."""

    @pytest.mark.parametrize(
        ("domain", "name"),
        [
            ("units", "units_from_signature"),
            ("schema", "schema_from_signature"),
            ("temporal", "freq_from_signature"),
        ],
    )
    def test_signature_readers(self, domain, name):
        assert name in importlib.import_module(f"xarray_annotated.{domain}").__all__

    @pytest.mark.parametrize(
        ("domain", "name"),
        [
            ("units", "annotated_unit"),
            ("schema", "annotated_schema"),
            ("temporal", "annotated_freq"),
        ],
    )
    def test_single_hint_readers(self, domain, name):
        assert name in importlib.import_module(f"xarray_annotated.{domain}").__all__

    def test_single_hint_readers_are_usable_from_the_domain(self):
        """The point of exporting them: a caller need not reach into a private
        module to read one annotation."""
        from typing import Annotated

        import xarray as xr

        from xarray_annotated.schema import Dims, annotated_schema
        from xarray_annotated.temporal import Freq, annotated_freq
        from xarray_annotated.units import Unit, annotated_unit

        hint = Annotated[xr.DataArray, Unit("Pa"), Dims("time"), Freq("D")]
        assert annotated_unit(hint) == "Pa"
        assert annotated_schema(hint) == [Dims("time")]
        assert annotated_freq(hint) == Freq("D")


class TestInternalHelpersStayInternal:
    """``check_dims`` / ``check_coords`` / ``check_dtype`` return ``(ok, detail)``
    tuples and apply no policy: they are message-building plumbing behind
    ``check_schema``, not user API, and are named accordingly."""

    @pytest.mark.parametrize("name", ["check_dims", "check_coords", "check_dtype"])
    def test_not_reachable_from_the_domain(self, name):
        assert not hasattr(xa.schema, name)

    @pytest.mark.parametrize("name", ["_check_dims", "_check_coords", "_check_dtype"])
    def test_still_available_privately(self, name):
        from xarray_annotated.schema import _check

        assert callable(getattr(_check, name))

    def test_public_entry_point_is_check_schema(self):
        assert "check_schema" in xa.schema.__all__
