"""The base class of every failed check, in a module the domains can import."""

from typing import Any, Literal

type Role = Literal["input", "output"]

#: Why a check failed, or two declarations conflict.  Public API: a value is added
#: only with a new way to fail, and never renamed.
type Kind = Literal[
    "unit_missing",
    "unit_too_long",
    "unit_unparseable",
    "unit_different",
    "unit_incompatible",
    "dims_mismatch",
    "dims_order",
    "coords_missing",
    "dtype_kind",
    "dtype_width",
    "freq_no_axis",
    "freq_ambiguous_axis",
    "freq_irregular",
    "freq_mismatch",
    "freq_phase",
    "not_array",
    "record_class",
    "record_field_missing",
]


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
    copies into a `Violation`, so a tool need not parse the message.  Every check in
    this package sets them; each is `None` on an error raised without it.

    Attributes:
        name: The name of the input or output that failed.
        qualname: The qualified name of the function it belongs to, if any.
        role: `"input"` or `"output"`.
        facet: The `Declared` slot that failed, such as `"unit"`, or `None` for a value
            that is not a `DataArray` or a record that does not match.
        kind: Why it failed, as listed on `Violation`.
        declared: The marker that failed, or `None` when `facet` is.
        found: What the array has for that facet, as described on `Violation`.
        hint: What to do about it, as it appears in the message, or `None` when the
            only fix is to change the data or the declaration.
    """

    name: str | None
    qualname: str | None
    role: Role | None
    facet: str | None
    kind: Kind | None
    declared: Any
    found: str | tuple[str, ...] | None
    hint: str | None

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        qualname: str | None = None,
        role: Role | None = None,
        facet: str | None = None,
        kind: Kind | None = None,
        declared: Any = None,
        found: str | tuple[str, ...] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.qualname = qualname
        self.role = role
        self.facet = facet
        self.kind = kind
        self.declared = declared
        self.found = found
        self.hint = hint


def _label(name: str, qualname: str | None, role: Role) -> str:
    """Name an input or output in a message, as `[f] input 'p'`."""
    return f"[{qualname}] {role} {name!r}" if qualname else f"{role} {name!r}"
