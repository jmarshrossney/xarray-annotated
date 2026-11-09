# AGENTS.md

Commands are in the `justfile`.

## Deliberate omissions

These look like gaps but are decisions, so don't fill them unasked.

- **Every failed check raises, and nothing softens it.**
  There are no severity settings, warnings, `check_annotations` options or global behaviour settings, because a softened marker reads the same as a checked one to everything downstream.
  `xra.enabled` is the only behavioural global, and it is how to prototype.
- **Checks change nothing.**
  `check_annotations` passes on the objects it was given, with no converting, stamping, dequantifying or encoding changes, so a body computes identically under `enabled(False)`.
  "Nothing" means values, shapes and types: stating a label is allowed, but only `stamp_output_units` does it, and it runs whatever `enabled` says.
- **Declarations never convert, and outputs must match once stated.**
  A compatible-but-different unit raises, because whether a conversion is correct depends on the quantity: `g m-2 yr-1` and `g m-2 d-1` are compatible, and treating one as the other is wrong.
  On outputs, the package cannot tell a stale label (`p * 100` keeps `hPa`) from a forgotten conversion, and converting a stale label would apply the arithmetic twice.
  `units.convert`, at the call site, is the only thing that converts or respells a label.
- **`stamp_output_units` only fills missing labels, and is never part of `check_annotations`.**
  Input unit declarations are checked; output ones are stated by the developer, as the package cannot see the body.
  Don't make it raise or warn on a mismatch, which `check_annotations` reports, or let `overwrite` act without naming fields or `True`.
  It stays a separate decorator so that a pint body, which has none, keeps the missing-label error.
  See `docs/adr/0004-output-units-are-stated-and-stamping.md`.
- **`Unit` stays exact, and docs recommend plain arrays between functions.**
  Don't add a dimension-only marker or show quantified arrays crossing a function boundary as the default: a plain body handed a quantified array can silently change its answer.
  See `docs/adr/0001-exact-units-plain-arrays-between-functions.md`.
- **Non-`DataArray` union members are not checked.**
  `A | float` given a string passes, as `n: int` given a string does, because checking those types is a type checker's job and not this package's.
  Don't add `isinstance` checks: what stops a user being misled is that every marker is either enforced or refused at decoration, so `A | list[A]` and `Annotated[DataArray | float, ...]` raise.
  `| None` is the exception, because `None` is the common mistake on an array parameter: without it, `None` raises.
- **A union with another type goes outside the `Annotated`, with no exception for `None`.**
  `Annotated[DataArray | None, ...]` raises, so `Annotated[DataArray, ...] | None` is the one spelling, and markers never read as applying to `None`.
- **`Freq(..., gaps_ok=True)` and `Freq("month")` are declarations, not softened checks.**
  The smallest step must equal the declared one, accepting a loud false alarm over letting every midnight-stamped axis pass.
  See `docs/adr/0002-sampling-period-and-calendar-steps.md`.

## Traps

- **No `from __future__ import annotations`.**
  It makes `get_type_hints` re-evaluate annotations at decoration, so a function-local alias or `TYPE_CHECKING`-only name raises `NameError`.
- **Checks must never read array values.**
  Contracts run before data loads (see `CONTRIBUTING.md`), and the obvious xarray and pint-xarray calls break this:
  - `.pint.units` goes through `da.data` and reads a lazily opened variable from disk, so use `_quantified_units`.
  - `da.indexes` casts every index to pandas and raises on a custom one, so use `da.xindexes`.
  - Detecting cftime from values reads every unindexed object coord, so a time axis is only ever a coord with a `PandasIndex`.
- **pint-xarray's own helpers are wrong here, so don't simplify to them.**
  - `.pint.quantify()` also parses coord labels: `degrees_north` aborts a conversion, and a bare `.sel(lat=20.0, method="nearest")` on a quantified coord silently selects the wrong row.
    `quantify({coord: None})` is no way out, as it deletes the label.
  - `.pint.dequantify()` and `.pint.to` keep `.encoding`, and a stale `scale_factor` overflows on write and silently corrupts the file.
    Clear it with `encoding = {}`, on the array only, since `drop_encoding()` also clears coords.
  - `quantify()` raises when a leftover attrs label disagrees with the `Quantity`, so a quantified array's unit is never read from attrs.
- **Mismatch errors are never `ValueError` or `TypeError`.**
  `UnitError`, `SchemaError` and `FreqError` extend `ContractError`, which extends `Exception`, while a malformed declaration raises `ValueError`, so catching a mismatch never swallows a declaration error.
  A value that is not a `DataArray`, or a returned record of the wrong class or missing a required field, raises `ContractError` itself, so `except ContractError` catches every failed check.
  `quantify` and `convert` are operations rather than checks, and keep raising `ValueError` and `pint.DimensionalityError`.
- **`Freq.freq` stores the raw string, so never normalise or compare `freqstr`.**
  `to_offset("W").freqstr` becomes `"W-SUN"`, which loses whether the phase was pinned.
  `.freq` is not always a pandas alias, so go through `_declared_spacing` rather than `to_offset`.

## Adding a facet

Each of these needs the new marker.
A new `Declared` slot fails `tests/test_facet_wiring.py` until it joins that file's `FACETS` table, which then checks every row below:

| File | Change |
| --- | --- |
| `_writer.py` | `annotate` |
| `_reader.py` | a `Declared` slot and `declaration_from_hint` |
| `_contract.py` | `assert_valid_declaration`, `_facet_checks`, `declaration_conflicts` |
| `_check_annotations.py` | `_check_output`, and the hard-coded marker list in `_markers_in` |
| `__init__.py` | re-export the marker from the root |

Units are checked first on both sides, as that fixed order is the reason there is one decorator, so add the new check after them.

## Docs

- Every `python` fence in `docs/` and the README runs under `tests/test_docs.py`, per page, top to bottom.
  On the line before a fence, mark a fragment with `<!-- docs-test: skip -->` and a deliberate failure with `<!-- docs-test: raises SomeError -->`.
- A `text` fence directly after a `python` fence is checked against what that fence prints, or `SomeError: message` if it raises, with whitespace runs compared equal, so shown output cannot drift.
- ADRs in `docs/adr/` are excluded from the site, so user-facing pages must not link to them.
- `just docs` reporting a failed notebook cell is expected: `examples/notebook.py` demonstrates a `UnitError` by raising one.

## Releasing

Every PR merges to `main`; a release is a tag on it, made by the maintainer only.

1. `uv version --bump rc` (`0.10.0.dev1` → `0.10.0rc1`, `0.10.0rc1` → `0.10.0rc2`), or `--bump stable` for a full release, and commit it to `main`.
2. `just release`, which refuses a dev version, a dirty tree, or a `main` that differs from `origin/main`.
3. After a full release, `uv version --bump minor --bump dev` and commit, so a git install of `main` does not report the released version.
   After a pre-release, `main` keeps reporting the rc version until the next bump; that is accepted.

A pre-release lets downstream projects (conduit, satterc) test unreleased work from PyPI by asking for it explicitly (`>=0.9.0rc1`), while everyone else keeps the last stable release.

The docs site deploys only on a stable release, so it documents what PyPI serves.
To rebuild a release's docs: `gh workflow run docs.yml -f ref=v0.9.0`.
Running it without `ref` deploys `main`, so the site then documents unreleased work; to publish a docs fix properly, release a patch.

TestPyPI is manual only (Actions → "Publish to PyPI" → testpypi), as a dry run after changing the publish workflow or the build.
