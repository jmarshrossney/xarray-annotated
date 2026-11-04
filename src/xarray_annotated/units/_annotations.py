"""The `Unit` marker declaring a DataArray's unit.

Pure `typing` metadata with no dependency on the active registry or configuration:
the unit string is validated later, by the checking layer.
"""

__all__ = ["Unit"]


class Unit:
    """Typed marker declaring the expected unit of a DataArray.

    Inside ``Annotated`` it is self-identifying and independent of metadata
    ordering, so it composes cleanly with schema markers, descriptions and other
    ``Annotated``-based tooling:

        Annotated[xr.DataArray, Unit("degC"), SomeOtherMarker(...)]

    The unit string is *not* validated here — parsing is deferred to
    `assert_valid_unit` at decoration time.

    Equivalent spellings (`"pascal"` for `"Pa"`) pass, since the values do not change.
    Any other unit raises, including a compatible one (`"hPa"` where `"Pa"` is declared)
    and a missing one.  Compare spellings by meaning with `units_equal`.

    Args:
        unit: The declared unit string.
    """

    _unit: str
    __slots__ = ("_unit",)

    def __init__(self, unit: str) -> None:
        self._unit = unit

    @property
    def unit(self) -> str:
        """The declared unit string."""
        return self._unit

    def __repr__(self) -> str:
        return f"Unit({self._unit!r})"

    def __str__(self) -> str:
        # The bare unit string, for readable interpolation (e.g. f"expected {m}").
        return self._unit

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._unit == other._unit
        return NotImplemented

    def __hash__(self) -> int:
        # Defining __eq__ drops the default __hash__; restore it so that an
        # Annotated[...] hint carrying a Unit marker stays hashable (typing and
        # downstream tools cache/hash annotations).
        return hash(self._unit)
