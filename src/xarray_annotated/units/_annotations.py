"""The `Unit` marker declaring a DataArray's unit.

Pure `typing` metadata with no dependency on the active registry or configuration:
the unit string is validated later, by the checking layer.
"""

from collections.abc import Callable

from ._config import (
    OnOutput,
    _validate_convert,
    _validate_on_output,
)

__all__ = ["Unit"]


def _optional[T: str](validate: Callable[[str], T], value: T | None) -> T | None:
    """Validate an optional override at construction; `None` means no override."""
    return None if value is None else validate(value)


class Unit:
    """Typed marker declaring the expected unit of a DataArray.

    Inside ``Annotated`` it is self-identifying and independent of metadata
    ordering, so it composes cleanly with schema markers, descriptions and other
    ``Annotated``-based tooling:

        Annotated[xr.DataArray, Unit("degC"), SomeOtherMarker(...)]

    The unit string is *not* validated here — parsing is deferred to
    `assert_valid_unit` at decoration time.

    **Conversion is opt-in, per declaration.**  An input in a different but
    compatible unit (`"hPa"` where `"Pa"` is declared) raises unless the
    declaration says `convert=True`.  Whether a conversion is *correct* is a
    property of the quantity, not of the function or the process: kelvin input
    to a Celsius parameter should convert, but an annual total converted into a
    daily mean is silently scaled by 365.25:

        tair: Annotated[xr.DataArray, Unit("degC", convert=True)]
        total: Annotated[xr.DataArray, Unit("g m-2 yr-1")]  # refuses g m-2 d-1

    Equivalent spellings (`"pascal"` for `"Pa"`) never need `convert=True`,
    since the values do not change.

    `on_output` can be set for this declaration too, and beats the decorator
    keyword, which beats the default; switching the package off
    (`enabled(False)`) still disables everything.  `convert` applies where the
    marker declares an input and `on_output` where it declares an output; a
    setting that does not apply where the marker sits is ignored, so one marker
    can be reused on both sides.

    An input with no parseable unit always raises: the declaration cannot be
    checked, so attach the unit where the data enters.

    Equality and hashing include the overrides, as for every other marker:
    `Unit("Pa", convert=True) != Unit("Pa")`.  To compare only the units,
    compare `.unit`, or use `units_equal` to compare spellings by meaning.

    Args:
        unit: The declared unit string.
        convert: Convert a compatible input into `unit` instead of raising.
        on_output: Override the on-output axis for this declaration.

    Raises:
        TypeError: If `convert` is not a bool.
        ValueError: If `on_output` is not a valid value.
    """

    _unit: str
    _convert: bool
    _on_output: OnOutput | None
    __slots__ = ("_convert", "_on_output", "_unit")

    def __init__(
        self,
        unit: str,
        *,
        convert: bool = False,
        on_output: OnOutput | None = None,
    ) -> None:
        self._unit = unit
        self._convert = _validate_convert(convert)
        self._on_output = _optional(_validate_on_output, on_output)

    @property
    def unit(self) -> str:
        """The declared unit string."""
        return self._unit

    @property
    def convert(self) -> bool:
        """Whether this declaration converts a compatible input."""
        return self._convert

    @property
    def on_output(self) -> OnOutput | None:
        """This declaration's on-output override, or `None`."""
        return self._on_output

    def _key(self) -> tuple[str, bool, str | None]:
        return (self._unit, self._convert, self._on_output)

    def __repr__(self) -> str:
        parts = [repr(self._unit)]
        if self._convert:
            parts.append("convert=True")
        if self._on_output is not None:
            parts.append(f"on_output={self._on_output!r}")
        return f"Unit({', '.join(parts)})"

    def __str__(self) -> str:
        # The bare unit string, for readable interpolation (e.g. f"expected {m}").
        return self._unit

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Unit):
            return self._key() == other._key()
        return NotImplemented

    def __hash__(self) -> int:
        # Defining __eq__ drops the default __hash__; restore it so that an
        # Annotated[...] hint carrying a Unit marker stays hashable (typing and
        # downstream tools cache/hash annotations).
        return hash(self._key())
