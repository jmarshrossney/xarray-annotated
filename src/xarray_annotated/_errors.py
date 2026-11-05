"""The base class of every failed check, in a module the domains can import."""


class ContractError(Exception):
    """Raised when data fails its declaration.

    The domains subclass this as `UnitError`, `SchemaError` and `FreqError`, so a tool
    can catch every failed check on an array with `except ContractError`.  A value that
    is not a `DataArray` is a `TypeError` instead.

    Deliberately **not** a `ValueError`: failing data is a distinct event from a
    malformed *declaration*, which raises `ValueError`.  Keeping them on separate
    hierarchies means catching one never silently swallows the other.
    """
