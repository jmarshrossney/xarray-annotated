"""The base class of every failed check, in a module the domains can import."""


class ContractError(Exception):
    """Raised when data fails its declaration.

    The domains subclass this as `UnitError`, `SchemaError` and `FreqError`, so a tool
    can catch every failed check with `except ContractError`.  It is raised directly for
    a value that is not a `DataArray` and for a returned record that is the wrong class
    or lacks a required field.

    Deliberately **not** a `ValueError`: failing data is a distinct event from a
    malformed *declaration*, which raises `ValueError`.  Keeping them on separate
    hierarchies means catching one never silently swallows the other.
    """
