"""Tests for the shape of the public API surface.

Each domain declares its public names in ``__all__``, and the API reference is
generated from those modules, so a name that is documented-but-unexported is
invisible to users while still looking public in the source. These tests guard
that boundary in both directions: everything promised is importable, and nothing
internal leaks.
"""

import importlib

import pytest

import xarray_annotated as xra

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
        private = [
            name
            for name in mod.__all__
            if name.startswith("_") and not name.startswith("__")
        ]
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
        assert not hasattr(xra.schema, name)

    @pytest.mark.parametrize("name", ["_check_dims", "_check_coords", "_check_dtype"])
    def test_still_available_privately(self, name):
        from xarray_annotated.schema import _check

        assert callable(getattr(_check, name))

    def test_public_entry_point_is_check_schema(self):
        assert "check_schema" in xra.schema.__all__


class TestMarkersAtRoot:
    """Markers are the one domain name re-exported from the root: they are
    unique across domains and mixed in one hint. Policy names are not, since
    they collide (``Policy``, ``set_policy``, ...)."""

    @pytest.mark.parametrize(
        ("domain", "name"),
        [
            ("schema", "Coords"),
            ("schema", "Dims"),
            ("schema", "Dtype"),
            ("temporal", "Freq"),
            ("units", "Unit"),
        ],
    )
    def test_root_marker_is_the_domain_marker(self, domain, name):
        assert name in xra.__all__
        domain_mod = importlib.import_module(f"xarray_annotated.{domain}")
        assert getattr(xra, name) is getattr(domain_mod, name)

    def test_only_markers_are_re_exported_from_domains(self):
        domain_names = {
            n
            for d in DOMAINS
            for n in importlib.import_module(f"xarray_annotated.{d}").__all__
        }
        leaked = domain_names & set(xra.__all__)
        assert leaked == {"Coords", "Dims", "Dtype", "Freq", "Unit"}


def test_integrations_are_never_imported():
    """The jaxtyping/beartype support must stay structural, not a dependency.

    ``_is_dataarray_type`` duck-types on an ``array_type`` attribute precisely so the
    package never imports jaxtyping. A stray import would turn an optional composition
    into a hard requirement, which no existing test would notice — the dev environment
    has both installed. Checked in a subprocess so this test's own imports don't mask it.
    """
    import subprocess
    import sys

    code = (
        "import sys; import xarray_annotated, xarray_annotated.schema, "
        "xarray_annotated.temporal, xarray_annotated.units; "
        "print([m for m in ('jaxtyping', 'beartype') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"imported an integration: {out.stdout}"


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert xra.__version__ == expected
