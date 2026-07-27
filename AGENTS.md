# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Commands

```sh
just               # lint, typecheck, test, docs (full check)
just lint          # ruff format + ruff check --fix
just lint-check    # ruff format --check + ruff check (CI, no mutation)
just typecheck     # pyright
just test          # pytest --verbose
just test-cov      # pytest with coverage (fails under 95%)
just doctest       # pytest --doctest-modules src/xarray_annotated
just docs          # marimo-md-export examples/notebook.py docs/example.md && zensical build
```

## Gotchas

### Do not use `from __future__ import annotations`

This stringizes annotations. `get_type_hints(..., include_extras=True)` can no longer resolve markers — declarations are silently lost, and the decorators validate nothing. Don't use it in any module that carries `Annotated` declarations (or their aliases).

### Do not alias declarations with `type`

```python
Pressure = Annotated[xr.DataArray, Unit("Pa")]         # ✅ read
type Pressure = Annotated[xr.DataArray, Unit("Pa")]    # ❌ silently ignored
```

PEP 695 `type` aliases are lazy: `get_type_hints` returns the alias object itself rather than the `Annotated` it wraps, so markers inside are never seen. Use plain assignment.

### pint registry / policy state are process-global

pint's application registry is a single process-global object. `use_cf_units()` / `set_registry()` is a one-time startup choice — quantities from two registries can't mix. The test suite snapshots and restores it per test via `conftest.py:_isolate_registry`. Policy override globals are also process-wide and isolated per test (`conftest.py:_isolate_policy`).

### `enabled` is package-wide — one switch gates every domain

Toggling `enabled` via any domain's `set_policy` or `policy` context manager toggles every decorator across all domains. The master switch lives in the shared `_config.py`, and each domain's `Policy` resolves it from there.

### `check_units` re-stamps `attrs["units"]`

After conversion, `pint.dequantify` writes its own canonical spelling (e.g. `"pascal"` for `"Pa"`). `check_units` overwrites it with the *declared* unit to preserve the caller's spelling. If you bypass it and convert yourself, the attribute drifts.

### `Freq.freq` stores the raw string — never normalise

`pandas.tseries.frequencies.to_offset("W").freqstr` silently expands to `"W-SUN"`. `Freq` stores the verbatim declared string because the raw spelling is the only place anchoredness survives. The comparison model uses the raw string to determine whether the anchor was explicitly spelled. Never store or compare the normalised `freqstr`.

### Mismatch errors are never `ValueError`

`SchemaError` and `FreqError` both extend `Exception` directly (not `ValueError`). A malformed *declaration* (bad dtype string, unparseable offset) raises `ValueError` at decoration time. This is deliberate: catching a mismatch must never accidentally swallow a declaration error.

### `check_units` on a dimensional mismatch always raises

There is no policy knob for dimensional incompatibility (e.g. `"kg"` where `"Pa"` is declared). It always raises `pint.DimensionalityError` regardless of `on_missing` / `on_inexact`.

### No bare-string shorthand for schema or temporal

Only `units` accepts a bare string (`Annotated[DataArray, "Pa"]`). For schema and temporal, a string in `Annotated` metadata is always a description — only the typed markers (`Dims(...)`, `Dtype(...)`, `Coords(...)`, `Freq(...)`) are read. If you add a bare-string `"my freq"` it will be silently ignored.

## Design steers

### Adding a domain facet: touch `_writer.py` and `_reader.py`

Every facet touches both cross-domain files at the package root. `annotate(...)` is the writer (facet values → `Annotated` hint); `declarations_from_signature(...)` is the reader (signature → per-parameter `Declared`). Their round-trip is intended to be exact.

### Adding a property within a domain: use the existing patterns

Schema's `_CHECKERS` dict (in `_check.py`) dispatches marker type → checker function — a 4th structural property is additive via this registry. Each domain mirrors the same internal layout: `_annotations.py` / `_check.py` / `_config.py` / `_decorator.py`, all re-exported flat from the domain's `__init__.py` via `__all__`.

### `walk_signature` is the shared driver for all `*_from_signature` readers

`xarray_annotated._annotations.walk_signature(func, extract)` is the generic kernel. Each domain supplies an extractor (one `Annotated` hint → its payload or `None`); `walk_signature` handles `TypedDict`/dataclass/single return shapes once. A third-party facet author reuses this to build their own reader.

### Decorator scaffold is triplicated (deferred consolidation)

Each domain's `_decorator.py` has the same structure but a different leaf verb (units converts+stamps; schema and temporal validate only). This is acknowledged duplication — collapsing it into a shared generic is deferred.

### Package root `__init__.py` imports subpackages but never re-exports domain names

`from xarray_annotated import units; units.declare_units` — the subpackage is the namespace. This prevents domain names from ever colliding in a shared top-level namespace.

## Conventions

- **Docs**: zensical (mkdocs-material), **not** Sphinx/rst. API pages use `::: xarray_annotated.units` directive syntax. Docstrings are Google-style.
- **Doctests**: embedded in source docstrings; run with `just doctest`.
- **Examples**: `examples/notebook.py` is a marimo notebook. `just docs` exports it to `docs/example.md` via `marimo-md-export` before building.
- **Formatting**: ruff (line-length 88), pyright. Python ≥3.12.
