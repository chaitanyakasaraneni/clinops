# Contributing

Contributions are welcome — bug reports, documentation improvements, and new features all help.

## Setup

```bash
git clone https://github.com/chaitanyakasaraneni/clinops
cd clinops
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Linting and formatting

```bash
ruff check clinops/
ruff format clinops/
mypy clinops/ --ignore-missing-imports
```

All three must pass before opening a pull request.

## Building docs locally

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Ruff lint and format pass
- [ ] mypy passes
- [ ] Docstrings updated for any changed public API
- [ ] Entry added to the relevant guide page if adding a new feature

## Code of Conduct

See [CODE_OF_CONDUCT.md](https://github.com/chaitanyakasaraneni/clinops/blob/main/CODE_OF_CONDUCT.md).
