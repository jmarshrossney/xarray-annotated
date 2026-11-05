# FAQ

## Why raise on a compatible unit, instead of converting it?

Pint knows whether two units are compatible, but not whether converting between them is correct, because that depends on what the data means.
A daily mean rate converts to `g / m**2 / year` without complaint, but the result is an annual total only if every day had that same rate.
A body that relied on conversion would also change its results when checks are [switched off](guides/troubleshooting.md#enabled).

Call [`xra.units.convert`][xarray_annotated.units.convert] where you know the conversion is right for the data, so the decision is visible at the call site.
See [Why conversion is not automatic](guides/units.md#why-conversion-is-not-automatic) for examples.

## Why can't I declare only a physical dimension, such as "a pressure"?

Accepting any pressure is safe only if the function body stays in pint throughout, and a check cannot see the body.
A body that uses `p.data.magnitude / 100` gets 1013.25 from 101325 `Pa`, but 10.13 from the same pressure in `hPa`, with no error.
A dimension does not identify a quantity either: `K` and `degC` are both temperatures, but `T * 2` works in `K` and raises in `degC`.

So `Unit` requires the exact unit, and the caller converts with [`xra.units.convert`][xarray_annotated.units.convert] where needed.
If a function really should accept any compatible unit, leave the input undeclared and start the body with `p = xra.units.convert(p, "Pa")`.
This works for plain and quantified arrays, and raises on a missing label or the wrong dimension.
The signature no longer states the expectation, so say it in the docstring.

See [Converting an input](guides/units.md#converting-an-input).

## Why pass plain arrays between functions, rather than quantified ones?

A body written for plain arrays can give a different answer when handed a quantified one, with no error: `1 - rh / 100` gives 0.4 for 60 `percent` on a plain array, and 0.994 on a quantified one.
If only plain arrays cross function boundaries, each body knows what it will receive.

You still get pint's checks on the arithmetic, by quantifying inside the function: call `xra.units.quantify` on each input you compute with, and return `xra.units.dequantify(xra.units.convert(result, unit))`.
Laziness does not affect the choice, as dask-backed arrays stay lazy either way, so open files with `chunks={}`.

Quantified arrays still pass every check, and they are worth keeping where the code between your functions does arithmetic of its own.
See [Plain arrays between functions](guides/pint.md#between-functions).

## Why can't I declare constraints on values, such as a range?

A check reads only metadata and indexes, never an array's values, which is what lets it run before the data is loaded.
A constraint on values would have to read them, so it is out of scope, however useful it might be.

Write such checks in the function body, or use a tool designed for validating data.
See [Look before you leap](guides/look_before_you_leap.md#what-looking-cannot-tell-you) for what this leaves unchecked.

## Why can't I annotate a `Dataset`?

The variables in a `Dataset` often differ from each other, so an annotation describing each of them would have to be very complex.
Marking up a `Dataset` directly raises `ValueError` when the function is decorated, rather than leaving the marker unchecked.

Pass the arrays you need as separate parameters, as in `f(ds["p"], ds["t"])`.
For several outputs, return a `TypedDict` or dataclass with a declared field for each array.
See [Return several arrays](guides/declaring.md#return-several-arrays).

## Why is there no warning or lenient mode?

A softened check looks the same as a real one to everything downstream, so a passing array could no longer be taken to mean a checked one.
Every failed check therefore raises.

To tolerate something, correct or remove the declaration.
While prototyping, [switch the package off](guides/troubleshooting.md#enabled) with `xra.enabled(False)`.

## Why are there no global settings?

How a function validates should be readable from its signature.
A process-wide setting could loosen a check with nothing at the definition to show it.

The one exception is the on/off switch, which decides whether checks run at all rather than how they behave.
The unit registry is also process-wide, but it decides only [which units parse](guides/registry.md).

## Why doesn't `enforce` fix or label an output's unit?

A label that differs from the declaration could be a forgotten conversion, where the label is true and the values are in the wrong unit, or a stale label left behind by arithmetic.
The package cannot tell them apart, and fixing the wrong one would produce a mislabelled result without any error.
So both raise, as a false alarm is loud and a mislabelled result is silent.

[Compute with pint](guides/pint.md) where you can, so the unit follows from the arithmetic, or state the unit with `.assign_attrs` when the body needs the bare numbers.
See [Output units](guides/units.md#outputs).

## Why not use a static type checker?

Type checkers ignore `Annotated` metadata, and an array's units, dimensions and time axis are usually only known once a file has been opened.
So these contracts are enforced at run time, but they are cheap enough to check before an expensive workflow starts.
See [Look before you leap](guides/look_before_you_leap.md).
