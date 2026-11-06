# Keeping signatures readable

## Reuse a declaration

Writing markers in the signature keeps each declaration in view, so it is a good default.
When signatures get long, or the same markers repeat across many functions, name a declaration with a `type` statement or an assignment:

```python
type Pressure = Annotated[xr.DataArray, Unit("Pa"), Dims("time", "x")]
```

Wrap a named declaration to add markers to it:

```python
type TypedPressure = Annotated[Pressure, Dtype("float64")]
```

An alias is checked exactly like the hint it stands for, but keep these in mind:

- **The markers move out of the signature.**
  A reader, an AI agent or generated documentation sees `Pressure` and has to look up what it declares, so name an alias after what it declares and keep it close to the code that uses it.
- **Only repetition pays for an alias.**
  It takes no marker arguments, so each distinct combination needs its own name, and a one-off function is usually clearer written out.
- **Alias what is shared.**
  A marker cannot be replaced, so a second `Unit` raises `ValueError`.
  Name the parts that repeat, such as dims and frequency, and add the rest where they are used.
- **Import aliases at run time.**
  `enforce` reads them at decoration, so an alias imported only under `TYPE_CHECKING` raises `NameError`.
