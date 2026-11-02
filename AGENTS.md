# AGENTS.md

## Commands

- See justfile.

## Gotchas

- **No `from __future__ import annotations`** — stringizes annotations; `get_type_hints` loses markers silently.
- **No `type Foo = Annotated[...]`** — PEP 695 lazy alias; `get_type_hints` returns alias object, not wrapped `Annotated`.
- **pint registry / policy state are process-global** — `use_cf_units()`/`set_registry()` is one-time startup; two registries can't mix. Tests isolate per test via `conftest.py:_isolate_registry` / `_isolate_policy`.
- **`enabled` is package-wide** — one switch gates all domains. Master switch in shared `_config.py`; each domain's `Policy` resolves from there.
- **`check_units` re-stamps `attrs["units"]`** — pint dequantify writes canonical spelling (e.g. `"pascal"` for `"Pa"`); `check_units` overwrites with declared unit. Bypassing it causes attribute drift.
- **Equal units skip the pint round-trip entirely** — `units_equal(have, declared)` (plain arrays only) returns `da.copy(deep=False)` + restamp. The round-trip copies the whole buffer and canonicalises *coord* attr spellings, both for nothing when there is no conversion to do. Don't "simplify" it back into one path.
- **`Freq.freq` stores raw string — never normalise** — pandas `to_offset("W").freqstr` silently becomes `"W-SUN"`. Raw spelling preserves anchoredness. Never store/compare normalised `freqstr`.
- **Mismatch errors are never `ValueError`** — `SchemaError`/`FreqError` extend `Exception` directly. Malformed declaration raises `ValueError` at decoration time so catching mismatch never swallows declaration error.
- **`check_units` on dimensional mismatch always raises** — no policy knob; always raises `pint.DimensionalityError` regardless of `on_missing`/`on_inexact`.
- **Quantified arrays are read via `.pint.units`, never attrs** — a `Quantity` holds its unit in the data and has empty attrs. `attrs` never wins: `quantify()` *raises* ("Cannot attach units") when a leftover attrs label disagrees, so reading attrs first breaks the pipeline. Inputs come back dequantified+stamped; outputs are **converted and stay quantified** (`dequantify()` copies the whole buffer even with nothing to convert). See `notes/logs/2026-07-31-quantified-arrays.md`.
- **`apply_output_units`' return must be used, never discarded** — in-place stamp for attrs arrays, but a *new* array for a converted quantified one. Same for the decorator's dataclass path (`_set_field`; frozen + conversion is unsupported).
- **`_is_dataarray_type` accepts `array_type` wrappers — never import jaxtyping** — the check is duck-typed so the composition stays structural; a real import would make it a hard dependency (guarded by `test_public_api.py:test_integrations_are_never_imported`). Hence jaxtyping/beartype sit in the **`integrations` dependency group**, not an extra — nothing to install to "enable" it, unlike the `cf` extra which really is imported. jaxtyping's dim strings are positional *size symbols*, not xarray dim names, so `Dims` is not redundant with them.
- **At most one marker of each type per hint** — `_annotations._single_marker` makes each domain extractor raise `ValueError` on a second `Unit`/`Dims`/`Dtype`/`Coords`/`Freq`, which is what makes `Declared`'s one-slot-per-type shape lossless. Don't relax it to first-wins or collect-all; merge `Coords` names into one marker instead. A bare string beside a `Unit` is still a description.
- **`Unit` carries per-declaration overrides; the string readers drop them** — `Unit(..., on_missing=, on_inexact=, on_output=)` mirrors the schema/temporal `on_mismatch` (marker > `enforce` keyword > env > `set_policy`). `check_units`/`apply_output_units` take `str | Unit`. `declarations_from_signature` keeps the marker (via `units._annotations._unit_marker`); `annotated_unit`/`units_from_signature` return plain strings by design, so a consumer that needs the overrides must read `Declared`.
- **`on_output="require"` is the only mode that refuses to stamp** — as `"strict"`, plus an output with no parseable unit of its own raises. Quantified outputs convert as under `"stamp"`. Unlike `"stamp"`/`"strict"`, absence *is* treated as evidence here: it is for wrappers whose declaration is a claim about someone else's convention.
- **No bare-string shorthand for schema or temporal** — only units accepts bare `Annotated[DataArray, "Pa"]`. Bare strings in schema/temporal `Annotated` are silently ignored as descriptions.
- **`just docs` reporting a failed notebook cell is expected** — `examples/notebook.py` demonstrates a `DimensionalityError` by raising one, so marimo prints "Export was successful, but some cells failed to execute" and the export continues. Not a regression; don't "fix" the example. Check a stashed tree before treating any docs failure as new.

## Design steers

- **New domain facet: touch `_writer.py`, `_reader.py` and `_enforce.py`** — `annotate()` writes facet values → `Annotated`; `declarations_from_signature()` reads signature → per-parameter `Declared` (add a slot). Round-trip exact. In `_enforce.py`, add the domain to `_SLOTS`, a `<domain>_<axis>` keyword per policy axis, and its check in the wrapper *after* units on both sides. No per-domain decorator.
- **New property within domain: follow existing patterns** — Schema's `_CHECKERS` dict dispatches marker type → checker; additive via registry. Each domain mirrors layout: `_annotations.py` / `_check.py` / `_config.py`, flat re-export via `__all__` (`_decorator.py` exists only as a deprecated shim in the three original domains).
- **`walk_signature` is the shared reader kernel** — `_annotations.walk_signature(func, extract)`. Each domain supplies an extractor (one `Annotated` hint → payload or `None`); handles `TypedDict`/dataclass/single return shapes once.
- **One decorator, `enforce`, in `_enforce.py`; units always first** — inputs: `check_units` converts, *then* `check_schema`/`check_freq` see the converted array; outputs: `apply_output_units`, *then* schema/freq check what the caller receives. That order is the reason it replaced the stackable per-domain decorators; don't split it back up. Per-function kwargs are `<domain>_<axis>` (`temporal_`, not `freq_`, matching the env vars). It reads through `declarations_from_signature`; `_restrict` blanks other domains' `Declared` slots for the shims.
- **`declare_units`/`declare_schema`/`declare_freq` are deprecated shims** — `_enforce` restricted to one domain, so old stacks behave as before. They import `_enforce` inside the function to avoid an import cycle. Remove them together with `tests/test_deprecated.py`.
- **Root `__init__.py` imports subpackages, never re-exports domain names** — `from xarray_annotated import units; units.check_units`. Cross-domain names (`enforce`, `annotate`, `declarations_from_signature`) are the exception. Prevents name collisions.

## Conventions

- **Docs**: mkdocs-material, Google-style docstrings. API pages use `::: xarray_annotated.units` directive.
- **Doctests**: embedded in docstrings; `just doctest`.
- **Examples**: `examples/notebook.py` is a marimo notebook; `just docs` exports to `docs/example.md` via `marimo-md-export`.
- **Formatting**: ruff (line-length 88), pyright, Python ≥3.12.
- **`notes/` is gitignored on purpose** — a local working area (`plans/`, `logs/`, `upstream_issues/`), not part of the package. Write there freely, but never `git add -f` it, and don't mistake its absence from a clone for something missing. Committed files may still reference it: the pointer is for whoever has the tree, not for the repo.
