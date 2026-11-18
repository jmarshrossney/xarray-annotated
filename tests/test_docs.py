"""Every `python` fence in the docs runs, so a page cannot drift from the code.

A page's fences run top to bottom in one namespace, as do the README's.  `PREAMBLE`
supplies the imports pages share, so a snippet about one marker need not repeat them.
Pages run on the default registry, as a reader's would, so one that uses CF unit strings
has to activate that registry itself.

A comment on the line before a fence changes how it runs:

* `<!-- docs-test: skip -->` for a fragment that is not meant to run;
* `<!-- docs-test: raises SomeError -->` for a deliberate failure, which must raise
  exactly that, by class name.

A `text` fence directly after a `python` fence is the output a reader is shown, so it
must equal what the fence prints, or `SomeError: message` for one that raises.  Runs of
whitespace compare equal, so a long message can be wrapped by hand.

`docs/example.md` is generated from `examples/notebook.py`, and is not checked here.
"""

import contextlib
import io
import logging
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

DOCS = Path(__file__).parent.parent / "docs"

PREAMBLE = """
from typing import Annotated

import xarray as xr
import xarray_annotated as xra
from xarray_annotated import Coords, Dims, Dtype, Freq, Unit
"""

_FENCE = re.compile(
    r"(?:^[ ]*<!-- docs-test: (?P<directive>.+?) -->\n)?"
    r"^[ ]*```python\n(?P<code>.*?)^[ ]*```[ ]*$"
    r"(?:\n(?:[ ]*\n)*^[ ]*```text\n(?P<output>.*?)^[ ]*```)?",
    re.MULTILINE | re.DOTALL,
)


def _pages() -> list[Path]:
    pages = (p for p in DOCS.rglob("*.md") if p.name != "example.md")
    return [*sorted(pages), DOCS.parent / "README.md"]


def _run(
    code: str, namespace: dict, directive: str | None, where: str, output=None
) -> None:
    if directive == "skip":
        return
    expected = None
    if directive is not None:
        expected = directive.removeprefix("raises ")
        assert expected != directive, f"{where}: unknown directive {directive!r}"
    raised = None
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            exec(code, namespace)
    except Exception as exc:
        if expected is None:
            raise
        raised = exc
    if expected is not None:
        assert type(raised).__name__ == expected, f"{where}: got {raised!r}"
        printed = io.StringIO(f"{expected}: {raised}")
    if output is not None:
        assert printed.getvalue().split() == output.split(), (
            f"{where}: shown output differs from\n{printed.getvalue()}"
        )


def _pages_about_declarations() -> list[Path]:
    """Every page but those on other tools, and the ADRs, plus the worked example."""
    pages = (p for p in _pages() if not {"tools", "adr"} & set(p.parts))
    return [*pages, DOCS.parent / "examples" / "notebook.py"]


@pytest.mark.parametrize("page", _pages_about_declarations(), ids=lambda p: p.name)
def test_pint_arithmetic_stays_on_the_pages_about_other_tools(page: Path):
    """How a body computes is not this package's concern; see ADR 0003."""
    assert not re.search(r"units\.(de)?quantify|\.pint\.|\.magnitude", page.read_text())


@pytest.fixture(autouse=True)
def _isolate_logging():
    """Undo a page's logging setup, which would outlive the test."""
    root = logging.getLogger()
    package = logging.getLogger("xarray_annotated")
    handlers, level = root.handlers[:], package.level
    yield
    root.handlers[:] = handlers
    package.setLevel(level)


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_snippets_run(page: Path):
    text = page.read_text()
    if "from jaxtyping import" in text:
        pytest.importorskip("jaxtyping")
        pytest.importorskip("beartype")
    if page.name in ("dask.md", "preflight.md"):
        pytest.importorskip("dask")
    if page.name in ("preflight.md", "index.md"):
        pytest.importorskip("netCDF4")
    namespace: dict = {}
    exec(PREAMBLE, namespace)
    for match in _FENCE.finditer(text):
        line = text.count("\n", 0, match.start("code")) + 1
        code = textwrap.dedent(match["code"])
        where = f"{page.name}:{line}"
        _run(code, namespace, match["directive"], where, match["output"])


class TestShownOutput:
    """A `text` fence after a `python` fence is checked, so shown output cannot drift."""

    @staticmethod
    def _page(code: str, output: str, directive: str = "") -> re.Match:
        page = f"{directive}\n```python\n{code}\n```\n\n```text\n{output}\n```\n"
        match = _FENCE.search(page.lstrip())
        assert match is not None
        return match

    def _check(self, match: re.Match) -> None:
        namespace: dict = {}
        code = textwrap.dedent(match["code"])
        _run(code, namespace, match["directive"], "page", match["output"])

    def test_matching_output_passes(self):
        self._check(self._page("print('a  b')", "a\nb"))

    def test_mismatched_output_fails(self):
        with pytest.raises(AssertionError, match="shown output differs"):
            self._check(self._page("print(1)", "2"))

    def test_mismatched_error_fails(self):
        directive = "<!-- docs-test: raises ValueError -->"
        self._check(self._page("raise ValueError('x')", "ValueError: x", directive))
        with pytest.raises(AssertionError, match="shown output differs"):
            self._check(self._page("raise ValueError('x')", "ValueError: y", directive))

    def test_an_unexpected_error_is_not_swallowed(self):
        with pytest.raises(ZeroDivisionError):
            self._check(self._page("1 / 0", ""))


class TestGettingStarted:
    """The page's versions of a function agree, and its table says what breaks them."""

    PAGE = DOCS / "getting_started.md"

    @staticmethod
    def _hourly(value, units, dims=("time", "site"), freq="h") -> xr.DataArray:
        time = pd.date_range("2024-07-01", "2024-07-02T23:00", freq=freq)
        return xr.DataArray(
            np.full((time.size, 2), value),
            dims=dims,
            coords={"time": time},
            attrs={"units": units},
        )

    @pytest.fixture
    def versions(self) -> list:
        """Each definition of the function in page order, keeping its own `R`."""
        namespace: dict = {}
        exec(PREAMBLE, namespace)
        found = []
        for match in _FENCE.finditer(self.PAGE.read_text()):
            namespace = dict(namespace)
            code = textwrap.dedent(match["code"])
            _run(code, namespace, match["directive"], self.PAGE.name)
            if match["directive"] is None and "def daily_air_density(" in code:
                found.append(namespace["daily_air_density"])
        assert len(found) == 2
        return found

    def test_every_version_gives_the_same_density(self, versions):
        p, t = self._hourly(1013.25, "hPa"), self._hourly(288.15, "K")
        for function in versions:
            np.testing.assert_allclose(function(p, t).data, 1.225, atol=5e-4)

    @pytest.mark.parametrize(
        ("change", "arg", "kwargs", "plain_cell", "plain_value", "error"),
        [
            (
                "`p` as the same pressure in `Pa`",
                "p",
                {"value": 101325.0, "units": "Pa"},
                "122.5, no error",
                122.5,
                "UnitError",
            ),
            (
                "`t` as the same temperature in `degC`",
                "t",
                {"value": 15.0, "units": "degC"},
                "23.53, no error",
                23.53,
                "UnitError",
            ),
            (
                "`t` with dims `(time, station)`",
                "t",
                {"value": 288.15, "units": "K", "dims": ("time", "station")},
                "Gains a `station` dimension, no error",
                1.225,
                "SchemaError",
            ),
            (
                "`t` every three hours",
                "t",
                {"value": 288.15, "units": "K", "freq": "3h"},
                "1.225 from every third hour, no error",
                1.225,
                "FreqError",
            ),
        ],
    )
    def test_the_table(
        self, versions, change, arg, kwargs, plain_cell, plain_value, error
    ):
        row = f"| {change} | {plain_cell} | `{error}` |"
        assert row in self.PAGE.read_text()

        args = {"p": self._hourly(1013.25, "hPa"), "t": self._hourly(288.15, "K")}
        args[arg] = self._hourly(**kwargs)

        plain, *declared = versions
        out = plain(**args)
        np.testing.assert_allclose(out.values, plain_value, atol=5e-3)
        assert ("station" in out.dims) == ("dims" in kwargs)
        if "freq" in kwargs:
            assert out.sizes["time"] == 2

        for function in declared:
            names = {"function": function, **args}
            _run("function(p, t)", names, f"raises {error}", change)
