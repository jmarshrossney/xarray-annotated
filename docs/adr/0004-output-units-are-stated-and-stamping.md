# ADR 0004: Output units are stated, and `stamp_output_units` states them

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-07 |
| **Follows** | [ADR 0003](0003-checks-compare-declarations-and-metadata.md), and reverses part of [#27](https://github.com/jmarshrossney/xarray-annotated/pull/27) |

Results below were measured on xarray 2026.9, pint 0.26 and pint-xarray 0.6.

## Context

#27 removed output stamping: an output had to carry its own label, and a missing one raised `UnitError`.
The reasoning was that stamping an unlabelled result was the last way a declared unit could reach a result unchecked.

In practice the fix for that error is always the same, `.assign_attrs(units=...)` at the end of the body, and it is no more checked than a stamp.
satterc shows the cost.
Each model wrapping pyrealm keeps an `_OUTPUT_UNITS` dict that copies its output `TypedDict`, so the output check compares the signature with a copy of itself.
When all three P-Model output annotations were found to be wrong, a test against pyrealm's own metadata caught it, which a dict copied from the annotations could never have done.

The hand fix is also weaker than it looks, because it overwrites.
`xr.apply_ufunc` copies the first input's attrs onto every output, so all twelve SPLASH outputs come back labelled `1`, from the sunshine fraction, and `.assign_attrs` hides that.

The decorator is also renamed, which is recorded here because the name follows from what it now does.

## Decision

1. **Input and output unit declarations mean different things.** An input declaration is checked against the label the caller's data carries. An output declaration is stated by the developer: the package cannot see what the body did, so it can only check that the body and the signature agree.
2. **`xra.stamp_output_units` fills each missing output label from the declaration.** A label that is present is left alone, and a mismatch raises in the checking decorator as before. A quantified output is never touched, as its unit comes from pint's arithmetic.
3. **`overwrite` replaces a present label, on named outputs only.** `overwrite=("field", ...)` names fields of a returned record, and `overwrite=True` covers every output. A bare string, or a name that is not a field, raises `ValueError` at decoration. Each replacement logs at `DEBUG`, with both units.
4. **`stamp_output_units` is a separate decorator, below the checking one, and always runs.** It is an operation like `xra.units.convert`, so `xra.enabled(False)` does not affect it. Placed above the checking decorator, it raises `ValueError` at decoration.
5. **`enforce` is renamed `check_annotations`, and is written `xra.check_annotations` throughout the docs.** It only checks, and nothing is converted, labelled or respelled.

<!-- docs-test: skip -->
```python
@xra.check_annotations
@xra.stamp_output_units(overwrite=True)
def splash(...) -> SplashOut:
    ...
```

## Reasons

### A missing label rarely says anything useful

The package cannot see inside a body, so an error for a missing output label can only say to add one.
Adding one with `.assign_attrs` states the declared unit on trust, as a stamp does.
Making the user write the claim by hand does not make it more likely to be right.

### Filling only missing labels is stricter than the hand fix

`.assign_attrs(units=...)` replaces whatever label is there, so a stale `hPa` on a result in `Pa` and a label copied by `apply_ufunc` both disappear without a trace.
`stamp_output_units` leaves a present label for the checking decorator, which raises on the mismatch.
Only `overwrite` replaces a label, and only on the outputs it names.

### Two decorators keep two kinds of body apart

| Body | Decorators | A missing label means |
|---|---|---|
| Computes with pint and dequantifies | `check_annotations` only | pint lost the unit, as dask `rolling` and `ffill` do, or the body went wrong, so it raises |
| Computes on bare numbers and builds a `DataArray` | `check_annotations` and `stamp_output_units` | Nothing, so the declaration states it |

Folding stamping into the checking decorator would make the two look the same, and would remove the one case where a missing label is worth an error.
A separate decorator is also visible at the definition, as a reminder that the outputs are labelled on trust.

### Stamping always runs, so switching off removes only checks

The package's own operations read `attrs["units"]`: `xra.units.quantify` and `xra.units.convert` raise without it, and a file written without it has no units.
A pipeline whose later bodies call them would work with checks on and raise with them off, if stamping stopped too.

"Checks change nothing" means values, shapes and types.
A label is metadata, so stating one is allowed, but it has to happen whether or not the checks run.

### Named overwrites keep the trust as narrow as the code

`overwrite=True` on a record would also silence a stale label that marks a forgotten conversion in another field.
Naming the fields keeps the claim as narrow as the code that justifies it, and `True` is for bodies where every output comes from one source of bare numbers, as in satterc's SPLASH and P-Model wrappers.
`True` is a bool rather than `"all"`, so it cannot collide with a field name.

### `check_annotations` says what the decorator does

`enforce` dates from when the decorator also converted and stamped.
It now only checks, and the name should say so.
Written `xra.check_annotations`, the name also says which annotations: `n: int` given a string still passes, and a type checker is the tool for that.

## Alternatives

| Alternative | Why not |
|---|---|
| `check_annotations(stamp=True)` | Blurs the two kinds of body above, gives `check_annotations` an option, and part of it would have to run under `enabled(False)` |
| Stamping by default in `check_annotations` | Removes the missing-label error from pint bodies too, where it is informative |
| Stamp raises on a mismatch | Duplicates the check, and since stamping always runs, `enabled(False)` would no longer switch off every check |
| Warn on a mismatch | Every failed check raises, and nothing softens it |
| A function-wide `overwrite` only | Silences a stale label in every field to cover the one that needed it |
| Respell an equal unit to the declared spelling | Left open: pint calls `J` and `N m`, and `Hz` and `rad/s`, equal, so respelling would write a claim nobody made |
| Convert a compatible unit | Changes values, which ADR 0001 and ADR 0003 rule out |

## Consequences

- **Output labels are claims, and the docs say so.** A user guide page lists the ways an output label can be wrong. ADR 0003's consequence that output labels are written by hand in the default path no longer holds.
- **Two decorators in a fixed order** where a body works on bare numbers, with the wrong order refused at decoration.
- **A breaking rename:** `enforce` is removed with no alias. The only users are catena, satterc and xharvest, which migrate in the same change.
- **Wrappers around libraries working on bare numbers still need tests against the library.** A stamp, like a hand label, proves nothing about the values, and satterc's `tests/test_pyrealm_units.py` is the pattern.

## When to revisit

Reconsider respelling equal units if a consumer comparing unit strings, such as a CF checker, is held up by spellings like `kg m2 s-2` for `J`.
Reconsider the missing-label error for pint bodies if `stamp_output_units` turns up on them routinely.
