"""Shared test fixtures.

pint's registry is **process-global**, and ``use_cf_units()`` changes it via a
side effect of importing ``cf_xarray.units`` (which calls
``pint.set_application_registry``). To keep the test suite order-independent —
so the plain-pint tests never silently inherit a CF registry left behind by
``test_cf.py`` — snapshot the full registry state before each test and restore
it afterwards.

Restoring the pint **application** registry is sufficient to also restore
pint-xarray: its ``default_registry`` is the application-registry proxy, so it
follows whatever the application registry currently is.
"""

import pint
import pytest

from xarray_annotated import _config as _shared_config
from xarray_annotated.schema import _config as _schema_config
from xarray_annotated.temporal import _config as _temporal_config
from xarray_annotated.units import _config as _units_config
from xarray_annotated.units import _registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the process-global pint registry around every test."""
    saved_ureg = _registry._UREG
    saved_using_cf = _registry._using_cf
    # The *concrete* registry, not the ApplicationRegistry proxy: the proxy is a
    # live pointer, so snapshotting it would capture nothing to restore.
    saved_app = pint.get_application_registry().get()
    try:
        yield
    finally:
        _registry._UREG = saved_ureg
        _registry._using_cf = saved_using_cf
        pint.set_application_registry(saved_app)


@pytest.fixture(autouse=True)
def _isolate_policy():
    """Snapshot and restore the process-global policy overrides around every test.

    The ``enabled`` master switch (shared, package-wide) and each domain's
    behavioural overrides are module globals; a test that sets one without a
    ``policy(...)`` context manager would otherwise leak into every later test
    across both domains. Restoring them here keeps the suite order-independent.
    """
    saved = (
        _shared_config._process_enabled,
        _units_config._process_on_missing,
        _units_config._process_on_inexact,
        _schema_config._process_on_mismatch,
        _temporal_config._process_on_mismatch,
        _temporal_config._process_on_uninferable,
    )
    try:
        yield
    finally:
        (
            _shared_config._process_enabled,
            _units_config._process_on_missing,
            _units_config._process_on_inexact,
            _schema_config._process_on_mismatch,
            _temporal_config._process_on_mismatch,
            _temporal_config._process_on_uninferable,
        ) = saved
