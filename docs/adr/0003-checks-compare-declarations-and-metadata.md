# ADR 0003: Checks compare declarations and metadata, and pint arithmetic is a neighbouring tool

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-06 |
| **Follows** | [ADR 0001](0001-exact-units-plain-arrays-between-functions.md) |

Results below were measured on xarray 2026.9, pint 0.26 and pint-xarray 0.6.

## Context

Every check in this package compares a declaration with another declaration, or with the metadata an array carries.
Pint has two separate roles around those checks:

1. **Parsing and comparing units.** A registry decides which strings parse and whether `pascal` equals `Pa`, and `xra.units.convert` is built on it.
2. **Computing with units.** A quantified body works out the unit of its result from the arithmetic.

The docs and the worked example presented the second as the natural way to write a declared function, and error messages told users to compute on `xra.units.quantify(...)`.
That raised the question of whether the package should go further, with a decorator that quantifies a body's inputs and dequantifies its result.

## Decision

1. **The docs present the package as comparing declarations with declarations and with metadata.** Those are the checks it can actually make.
2. **Pint's first role stays where it is, and its second moves to the pages on other tools.** `tests/test_docs.py` fails if `xra.units.quantify` or pint-xarray calls appear on any other page or in the worked example.
3. **Docs and error messages say what a label can't prove.** An output check shows that a body and its signature agree, not that the values are right, and a mismatch message asks about the values before offering a fix.
4. **There is no quantifying decorator in the package.**
5. **Quantified arrays and the helpers stay supported.** Checks still read the unit a quantified array holds, and `quantify` and `dequantify` still avoid the problems of their pint-xarray counterparts for anyone computing with pint.

## Reasons

### Pint arithmetic closes some failures and opens others

| Closes | Opens |
|---|---|
| Stale labels (`p * 100` keeping `hPa`) | Silent changes when a body moves to quantified arrays: `1 - rh / 100` gives 0.994 rather than 0.4 for 60 `percent` |
| Hand-written output labels that are wrong | A result's unit depending on the input (`p * 2`, `km + m`) |
| Dimensionally invalid arithmetic | Units lost with no warning by `rolling` and `ffill` on a dask-backed array |
| A forgotten time step, where the total is declared as a total | Wrong selections and corrupted files from `.pint.quantify()` and `.pint.dequantify()` |

This is a good trade for many bodies, but it is a choice about how to compute, and a package that only checks has no need to make it.

### A converting decorator would hide the errors pint finds

A decorator that converts a result to the declared output unit removes the most boilerplate.
Take 2 g/m²/day for every day of 2023, a true total of 730 g/m², and a body declared to return `g / m**2 / year`:

| Body | Converting decorator | No conversion |
|---|---|---|
| `flux.sum("time")` | 266,632.5, with no error | `UnitError` |
| `flux.mean("time") * 365` | 266,632.5, with no error | `UnitError` |

Pint correctly keeps the sum in `g / m**2 / day`, and the conversion then discards that signal.
A declaration would also be changing values, which [ADR 0001](0001-exact-units-plain-arrays-between-functions.md) rules out.

### A decorator that does not convert still hides the main hazard

Identical body text gives 0.994 or 0.4 depending on one line above the function, whereas an explicit `quantify(rh)` sits next to the arithmetic.
It saves less than it appears to, because constants still need a unit, a non-coherent output still needs `convert`, and dask `rolling` and `ffill` must run before quantifying.
As a package feature it would also have to handle records, `| None`, `Dataset`s and unlabelled parameters such as masks.

## Consequences

- **A smaller claim:** the docs promise less about units, but every part of that promise is something a check verifies.
- **Output labels are written by hand in the default path:** `assign_attrs(units=...)` can silence any output `UnitError`, however wrong the values are. Arithmetic needs tests, and the docs say so.
- **The worked example has hand-written factors:** the declarations catch its missing stage, but not a wrong factor inside that stage.
- **Pint remains a required dependency,** for its first role. Whether pint-xarray could become optional is a separate question.

## When to revisit

Reconsider a decorator that never converts if the same one keeps being written in real pipelines, and handling records and `None` proves simple.
It should then be a separate `xra.units` decorator that is never part of `enforce`.
