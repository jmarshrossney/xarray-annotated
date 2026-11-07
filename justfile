_: lint typecheck test doctest docs

# Format and lint the package using ruff.
lint:
  ruff format
  ruff check --fix
  marimo check --fix examples/notebook.py

# Variant of `lint` that doesn't cause any changes to files.
lint-check:
  ruff format --check
  ruff check
  marimo check examples/notebook.py

# Run static type checker.
typecheck:
  pyright

# Run the full test suite.
test:
  pytest --verbose

# Run tests with coverage report.
test-cov:
  pytest --cov=xarray_annotated --cov-report=term-missing --cov-fail-under=95

# Run doctest examples embedded in source docstrings.
doctest:
  pytest --doctest-modules src/xarray_annotated --verbose

# Build the documentation using Zensical.
docs:
  marimo-md-export examples/notebook.py docs/example.md
  zensical build

# Tag the version in pyproject.toml and push the tag, publishing to TestPyPI.
tag:
  #!/usr/bin/env bash
  set -euo pipefail
  v=$(uv version --short)
  [[ $v != *dev* ]] || { echo "refusing to tag dev version $v" >&2; exit 1; }
  [[ $(git branch --show-current) == main ]] || { echo "not on main" >&2; exit 1; }
  git diff --quiet HEAD || { echo "uncommitted changes" >&2; exit 1; }
  git fetch --quiet origin main
  [[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] \
    || { echo "main differs from origin/main" >&2; exit 1; }
  git tag -a "v$v" -m "v$v"
  git push origin "v$v"

# Create a GitHub release for the tagged version, publishing to PyPI.
release:
  #!/usr/bin/env bash
  set -euo pipefail
  v=$(uv version --short)
  git rev-parse --quiet --verify "refs/tags/v$v" >/dev/null \
    || { echo "no tag v$v: run 'just tag' first" >&2; exit 1; }
  pre=()
  [[ ! $v =~ (a|b|rc)[0-9]+$ ]] || pre=(--prerelease)
  gh release create "v$v" --verify-tag --generate-notes "${pre[@]}"
