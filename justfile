_: lint typecheck test docs

# Format and lint the package using ruff.
lint:
  ruff format
  ruff check --fix

# Variant of `lint` that doesn't cause any changes to files.
lint-check:
  ruff format --check
  ruff check

# Run static type checker.
typecheck:
  pyright

# Run the full test suite.
test:
  pytest --verbose

# Run tests with coverage report.
test-cov:
  pytest --cov=xarray_signature_units --cov-report=term-missing --cov-fail-under=95

# Build the documentation using Zensical.
docs:
  zensical build
