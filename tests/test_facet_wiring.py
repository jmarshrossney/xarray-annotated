"""Every facet must reach each place that handles markers one kind at a time.

Those places are written out by hand, and missing one raises nothing: the facet simply
goes unchecked there.  A new `Declared` slot fails `test_every_slot_is_listed` until it
joins `FACETS`, and every other test here then holds it to the same wiring.
"""

import dataclasses
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_annotated as xra
from xarray_annotated import Coords, Declared, Dims, Dtype, Freq, Unit
from xarray_annotated.schema import SchemaError
from xarray_annotated.temporal import FreqError
from xarray_annotated.units import UnitError


class Facet(NamedTuple):
    fits: Any  # holds on `_array()`
    misfit: Any  # fails on `_array()`, raising `error`
    malformed: Any
    error: type[xra.ContractError]
    conflicting: Any  # provably cannot describe the same array as `fits`


FACETS = {
    "unit": Facet(Unit("Pa"), Unit("K"), Unit("not_a_unit"), UnitError, Unit("K")),
    "dims": Facet(
        Dims("time"), Dims("x"), Dims("time", "time"), SchemaError, Dims("x")
    ),
    "dtype": Facet(
        Dtype("float"), Dtype("int"), Dtype("not_a_dtype"), SchemaError, Dtype("int")
    ),
    # Each `Coords` only requires names to be present, so two can never conflict.
    "coords": Facet(
        Coords("time"), Coords("lat"), Coords("time", "time"), SchemaError, None
    ),
    "freq": Facet(Freq("h"), Freq("D"), Freq("not_a_freq"), FreqError, Freq("D")),
}

parametrize_facets = pytest.mark.parametrize("slot", FACETS)


def _array():
    time = pd.date_range("2000-01-01", periods=4, freq="h")
    return xr.DataArray(
        np.ones(4), dims="time", coords={"time": time}, attrs={"units": "Pa"}
    )


def test_every_slot_is_listed():
    slots = {field.name for field in dataclasses.fields(Declared)}
    assert slots - {"others"} == set(FACETS)


@parametrize_facets
def test_marker_is_exported_from_the_root(slot):
    marker = FACETS[slot].fits
    assert getattr(xra, type(marker).__name__) is type(marker)


@parametrize_facets
def test_annotate_and_the_reader_round_trip(slot):
    marker = FACETS[slot].fits
    hint = xra.annotate(**{slot: marker})
    assert xra.declaration_from_hint(hint) == Declared(**{slot: marker})


@parametrize_facets
def test_malformed_marker_is_refused(slot):
    declared = Declared(**{slot: FACETS[slot].malformed})
    with pytest.raises(ValueError, match="node"):
        xra.assert_valid_declaration(declared, "node")


@parametrize_facets
def test_input_is_checked(slot):
    facet = FACETS[slot]
    assert (
        xra.declaration_violations(_array(), Declared(**{slot: facet.fits}), "p") == []
    )
    with pytest.raises(facet.error):
        xra.check_declared(_array(), Declared(**{slot: facet.misfit}), "p")


@parametrize_facets
def test_output_is_checked(slot):
    facet = FACETS[slot]

    def returning(marker):
        def func():
            return _array()

        func.__annotations__["return"] = xra.annotate(**{slot: marker})
        return xra.enforce(func)

    returning(facet.fits)()
    with pytest.raises(facet.error):
        returning(facet.misfit)()


@parametrize_facets
def test_stray_marker_is_refused(slot):
    def func(p):
        pass

    hint = xra.annotate(**{slot: FACETS[slot].fits})
    func.__annotations__["p"] = tuple[hint, int]
    with pytest.raises(ValueError, match="cannot apply"):
        xra.enforce(func)


@parametrize_facets
def test_conflicts_are_reported(slot):
    facet = FACETS[slot]
    if facet.conflicting is None:
        pytest.skip(f"{slot} cannot conflict")
    conflicts = xra.declaration_conflicts(
        Declared(**{slot: facet.fits}), Declared(**{slot: facet.conflicting})
    )
    assert len(conflicts) == 1
    assert conflicts[0].startswith(f"{slot}:")
