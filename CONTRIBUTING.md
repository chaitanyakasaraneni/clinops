# Contributing to clinops

Thank you for your interest in contributing! `clinops` welcomes contributions from researchers, engineers, and clinicians working at the intersection of healthcare data and machine learning.

## Getting Started

```bash
git clone https://github.com/chaitanyakasaraneni/clinops
cd clinops
pip install -e ".[dev]"
```

## Development Workflow

1. Fork the repository and create a feature branch: `git checkout -b feature/my-feature`
2. Write your code with type hints and docstrings
3. Add tests in `tests/` covering your changes
4. Run the test suite: `pytest tests/ -v`
5. Run linting: `ruff check clinops/ && black clinops/ tests/`
6. Submit a pull request with a clear description

## Code Standards

- **Type hints** on all public functions and methods
- **Docstrings** in NumPy style for all public APIs
- **Tests** for all new features (aim for >80% coverage on new code)
- **Logging** via `loguru` — no bare `print()` in library code

## Adding a New Data Source

To add a new clinical data source (e.g. eICU, OMOP CDM):

1. Create `clinops/ingest/your_source.py`
2. Implement a loader class following the pattern in `mimic.py`
3. Export from `clinops/ingest/__init__.py`
4. Add tests in `tests/test_ingest/`
5. Add a usage example in `examples/`

## Reporting Issues

Please use GitHub Issues with:
- A minimal reproducible example
- Your Python version and `clinops` version
- The full error traceback

## Clinical Data Privacy

**Never commit real patient data.** All test fixtures must use synthetic data. If you're testing against MIMIC-IV locally, add `data/` to your `.gitignore`.
