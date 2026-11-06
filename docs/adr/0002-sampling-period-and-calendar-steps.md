# ADR 0002: Declaring a sampling period and a calendar step with `Freq`

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-05 |
| **Discussion** | [#28](https://github.com/jmarshrossney/xarray-annotated/issues/28), [#29](https://github.com/jmarshrossney/xarray-annotated/issues/29) |

Results below were measured on pandas 3.1 and xarray 2026.9.

## Context

`Freq` only fitted an axis that `xarray.infer_freq` could name, which left two common axes with no declaration:

- **An axis with holes:** January 2025 joined to January 2026 holds daily values, but `Freq("D")` also claims that no timestamp is missing.
- **A calendar axis on no month boundary:** monthly means stamped at midpoints have no pandas frequency, so `"MS"` and `"ME"` both raise.

The only option was to leave `Freq` off, and then nothing stops half-hourly data reaching a function written for daily data.
There are no severity settings, by design, and `Freq("D")` had to keep its meaning.

## Decision

1. **`Freq("D", gaps_ok=True)` declares a sampling period on an axis that may have holes.** Timestamps must strictly increase, every step must be a whole multiple of the declared one, and the smallest step must equal it.
2. **`"month"`, `"year"` and multiples such as `"3month"` declare a calendar step on any day.** They compare consecutive `12 * year + month` values, and ignore the day and time.
3. **`freq_compatible` stays symmetric and ignores `gaps_ok`.** `"month"` has a month spacing and no phase, so it is compatible with both `"ME"` and `"MS"`.

The two options are independent, so `Freq("month", gaps_ok=True)` also works.

## Reasons

### These are declarations, not softer checks

A severity setting makes a passing array mean something weaker while reading the same.
Each spelling here says what it accepts at the signature, and a plain `Freq("D")` still means a regular axis.

### Whole multiples alone would accept almost anything coarser

| Axis, against `Freq("D", gaps_ok=True)` | Whole multiples | With smallest step |
|---|---|---|
| Two runs of daily values | pass | pass |
| Half-hourly | fail | fail |
| Two-day, weekly or month-start | **pass** | fail |
| Daily values with no two adjacent | pass | **fail** |

Without the smallest step, the declaration would reject only data finer than declared.
The last row is the cost, and it depends on the data, so a pipeline can pass for months and raise on one run.
That is a loud false alarm, which we prefer to a silently mislabelled result.

A consequence is that January alone across several years fails `Freq("month", gaps_ok=True)`, and is correctly declared `Freq("year")`.

### A string for the calendar step, a keyword for gaps

Whether an anchor matters is already decided by the spelling, as with `"W"` and `"W-SUN"`.
A keyword such as `Freq("MS", any_day=True)` would name a convention and then cancel it, so the marker would not mean what it reads.
pandas rejects `"month"` and `"year"`, so they cannot collide with an offset, whereas `"M"` stays an error so that pandas' message pointing to `"ME"` survives.

Gaps have no natural spelling in an offset string, and a suffix would be a private syntax on top of pandas aliases.
A separate marker would add a slot to `Declared`, and rules for what happens when it disagrees with `Freq`.

### Business days are refused rather than passed

A business-day or custom offset has no step to take multiples of, so `gaps_ok` could not check anything there.
Passing would make a marker that reads as checked and is not, so it raises `ValueError` at decoration.
A non-positive step is refused for the same reason.

### Symmetry matches what `freq_compatible` already means

It returns `True` unless two declarations are provably inconsistent, and `Freq("W")` into `Freq("W-SUN")` already passes without being proved safe.
A directional check would add public API, and catches nothing until a tool adopts it.

## Alternatives

| Alternative | Why not |
|---|---|
| Whole multiples only | Coarser data passes silently, as above |
| A separate marker for the sampling period | More machinery, and two markers can disagree on one hint |
| `Freq("MS", any_day=True)` | The string names a convention the keyword cancels |
| A directional `freq_satisfies` | Deferred. Adding it later breaks nothing |
| Reading CF `time_bounds` | Depends on variables outside the array |

## Consequences

- **`.freq` is not always a pandas alias:** a tool passing it to pandas, such as to `resample`, must handle `"month"` and `"year"`. conduit's `[[resample]]` validates with `assert_valid_freq`, so it would accept `"month"` and fail later.
- **`gaps_ok` is not a string:** a tool that reads frequencies from a config as plain strings cannot express it without a field of its own.
- **Edges are not proved safe:** a producer declaring `gaps_ok` passes a build-time check against a consumer that does not, and the consumer raises when the data arrives.
- **No check on where a value sits in its month:** `Freq("month")` accepts the 1st of one month and the 31st of the next, as it compares months alone.

## When to revisit

Add a directional check if a pipeline tool needs to catch the unproved edge before running.
Reconsider the smallest-step rule if sparse selections of daily data turn out to be common at function boundaries.
