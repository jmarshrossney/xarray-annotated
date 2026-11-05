"""Every `python` fence in the docs runs, so a page cannot drift from the code.

A page's fences run top to bottom in one namespace, as do the README's.  `PREAMBLE`
supplies the imports pages share, so a snippet about one marker need not repeat them.
Pages run on the default registry, as a reader's would, so one that uses CF unit strings
has to activate that registry itself.

A comment on the line before a fence changes how it runs:

* `<!-- docs-test: skip -->` for a fragment that is not meant to run;
* `<!-- docs-test: raises SomeError -->` for a deliberate failure, which must raise
  exactly that, by class name.

`docs/example.md` is generated from `examples/notebook.py`, and is not checked here.
"""

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
    r"^[ ]*```python\n(?P<code>.*?)^[ ]*```",
    re.MULTILINE | re.DOTALL,
)


def _pages() -> list[Path]:
    pages = (p for p in DOCS.rglob("*.md") if p.name != "example.md")
    return [*sorted(pages), DOCS.parent / "README.md"]


def _run(code: str, namespace: dict, directive: str | None, where: str) -> None:
    if directive == "skip":
        return
    if directive is None:
        exec(code, namespace)
        return
    expected = directive.removeprefix("raises ")
    assert expected != directive, f"{where}: unknown directive {directive!r}"
    raised = None
    try:
        exec(code, namespace)
    except Exception as exc:
        raised = exc
    assert type(raised).__name__ == expected, f"{where}: got {raised!r}"


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
    namespace: dict = {}
    exec(PREAMBLE, namespace)
    for match in _FENCE.finditer(text):
        line = text.count("\n", 0, match.start("code")) + 1
        code = textwrap.dedent(match["code"])
        _run(code, namespace, match["directive"], f"{page.name}:{line}")


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
        assert len(found) == 3
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
