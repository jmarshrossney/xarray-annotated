"""The base class of every failed check, in a module the domains can import."""

from typing import Any


class ContractError(Exception):
    """Raised when data fails its declaration.

    The domains subclass this as `UnitError`, `SchemaError` and `FreqError`, so a tool
    can catch every failed check with `except ContractError`.  It is raised directly for
    a value that is not a `DataArray` and for a returned record that is the wrong class
    or lacks a required field.

    Deliberately **not** a `ValueError`: failing data is a distinct event from a
    malformed *declaration*, which raises `ValueError`.  Keeping them on separate
    hierarchies means catching one never silently swallows the other.

    A failed check also says what failed, in the fields `declaration_violations`
    copies into a `Violation`, so a tool need not parse the message.

    Attributes:
        facet: The `Declared` slot that failed, such as `"unit"`, or `None` for a value
            that is not a `DataArray` or a record that does not match.
        declared: The marker that failed, or `None` when `facet` is.
        found: What the array has for that facet, as described on `Violation`.
    """

    facet: str | None
    declared: Any
    found: str | tuple[str, ...] | None

    def __init__(
        self,
        message: str,
        *,
        facet: str | None = None,
        declared: Any = None,
        found: str | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.facet = facet
        self.declared = declared
        self.found = found
