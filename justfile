_: lint typecheck test doctest docs

# Format and lint the package using ruff.
[group('check')]
lint:
  ruff format
  ruff check --fix
  marimo check --fix examples/notebook.py

# Variant of `lint` that doesn't cause any changes to files.
[group('check')]
lint-check:
  ruff format --check
  ruff check
  marimo check examples/notebook.py

# Run static type checker.
[group('check')]
typecheck:
  pyright

# Run the full test suite.
[group('test')]
test:
  pytest --verbose

# Run tests with coverage report.
[group('test')]
test-cov:
  pytest --cov=xarray_annotated --cov-report=term-missing --cov-fail-under=95

# Run doctest examples embedded in source docstrings.
[group('test')]
doctest:
  pytest --doctest-modules src/xarray_annotated --verbose

# Build the documentation using Zensical.
[group('docs')]
docs:
  marimo-md-export examples/notebook.py docs/example.md
  zensical build

# Tag the pyproject.toml version and publish it as a GitHub release and to PyPI.
[group('release')]
release:
  #!/usr/bin/env bash
  set -euo pipefail
  v=$(uv version --short)
  [[ $v != *dev* ]] || { echo "refusing to release dev version $v" >&2; exit 1; }
  [[ $(git branch --show-current) == main ]] || { echo "not on main" >&2; exit 1; }
  git diff --quiet HEAD || { echo "uncommitted changes" >&2; exit 1; }
  git fetch --quiet origin main
  [[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] \
    || { echo "main differs from origin/main" >&2; exit 1; }
  gh auth status >/dev/null || { echo "gh is not logged in" >&2; exit 1; }
  pre=()
  [[ ! $v =~ (a|b|rc)[0-9]+$ ]] || pre=(--prerelease)
  git tag -a "v$v" -m "v$v"
  git push origin "v$v"
  gh release create "v$v" --verify-tag --generate-notes ${pre[@]+"${pre[@]}"}
