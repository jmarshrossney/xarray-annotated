"""Every `python` fence in the docs runs, so a page cannot drift from the code.

A page's fences run top to bottom in one namespace, as do the README's.  `PREAMBLE`
supplies the imports pages share, so a snippet about one marker need not repeat them.
Pages run on the default registry, as a reader's would, so one that uses CF unit strings
has to activate that registry itself.

A fence inside a top-level `=== "Label"` tab runs in a namespace of its own, shared by
every tab with that label on the page, because linked tabs are alternative versions of
the page.  It sees the page's untabbed names, unless its own tabs have redefined them;
nothing it defines reaches another label or untabbed fences.

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
import pytest

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
_TAB = re.compile(
    r'^===[+!]? "(?P<label>[^"]+)"[ ]*\n(?:(?:[ ]{4}.*)?\n)*',
    re.MULTILINE,
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
    shared: dict = {}
    exec(PREAMBLE, shared)
    tabs = list(_TAB.finditer(text))
    by_label: dict[str, dict] = {}
    for match in _FENCE.finditer(text):
        line = text.count("\n", 0, match.start("code")) + 1
        code = textwrap.dedent(match["code"])
        label = next(
            (t["label"] for t in tabs if t.start() <= match.start() < t.end()), None
        )
        namespace = shared if label is None else by_label.setdefault(label, {})
        namespace.update({**shared, **namespace})
        _run(code, namespace, match["directive"], f"{page.name}:{line}")


class TestGettingStarted:
    """The page says each tab is a complete script and tabulates what breaks it."""

    PAGE = DOCS / "getting_started.md"
    PLAIN, ANNOTATED, QUANTIFIED = (
        "xarray",
        "xarray-annotated",
        "xarray-annotated + pint",
    )

    @pytest.fixture
    def scripts(self) -> dict[str, dict]:
        """Run each tab with nothing supplied, as a reader pasting it would."""
        text = self.PAGE.read_text()
        namespaces = {}
        for tab in _TAB.finditer(text):
            fence = _FENCE.search(tab[0])
            if tab["label"] in (self.PLAIN, self.ANNOTATED, self.QUANTIFIED):
                assert fence is not None
                namespaces[tab["label"]] = {}
                exec(textwrap.dedent(fence["code"]), namespaces[tab["label"]])
        assert len(namespaces) == 3
        return namespaces

    @staticmethod
    def _magnitude(array):
        return getattr(array.data, "magnitude", array.data)

    def test_every_tab_gives_the_same_density(self, scripts):
        for names in scripts.values():
            out = names["daily_air_density"](names["p"], names["t"])
            np.testing.assert_allclose(self._magnitude(out), 1.225, atol=5e-4)

    @pytest.mark.parametrize(
        ("change", "arg", "kwargs", "plain_cell", "plain_value", "error"),
        [
            (
                'hourly(101325.0, "Pa")',
                "p",
                {"value": 101325.0, "units": "Pa"},
                "122.5, no error",
                122.5,
                "ValueError",
            ),
            (
                'hourly(15.0, "degC")',
                "t",
                {"value": 15.0, "units": "degC"},
                "23.53, no error",
                23.53,
                "ValueError",
            ),
            (
                'hourly(288.15, "m")',
                "t",
                {"value": 288.15, "units": "m"},
                "1.225, no error",
                1.225,
                "DimensionalityError",
            ),
            (
                'hourly(288.15, "K", dims=("time", "station"))',
                "t",
                {"value": 288.15, "units": "K", "dims": ("time", "station")},
                "Gains a `station` dimension, no error",
                1.225,
                "SchemaError",
            ),
            (
                'hourly(288.15, "K", freq="3h")',
                "t",
                {"value": 288.15, "units": "K", "freq": "3h"},
                "1.225 from every third hour, no error",
                1.225,
                "FreqError",
            ),
        ],
    )
    def test_the_table(
        self, scripts, change, arg, kwargs, plain_cell, plain_value, error
    ):
        row = f"| `{change}` for `{arg}` | {plain_cell} | `{error}` |"
        assert row in self.PAGE.read_text()

        names = scripts[self.PLAIN]
        args = {"p": names["p"], "t": names["t"], arg: names["hourly"](**kwargs)}
        out = names["daily_air_density"](**args)
        np.testing.assert_allclose(out.values, plain_value, atol=5e-3)
        assert ("station" in out.dims) == ("dims" in kwargs)
        if "freq" in kwargs:
            assert out.sizes["time"] == 2

        for label in (self.ANNOTATED, self.QUANTIFIED):
            names = scripts[label]
            bad = names["hourly"](**kwargs)
            if label == self.QUANTIFIED:
                bad = names["xra"].units.quantify(bad)
            call = "daily_air_density(p, t)"
            _run(call, {**names, arg: bad}, f"raises {error}", f"{label}: {change}")
