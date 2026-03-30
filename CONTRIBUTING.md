# Contributing to SpectraScope

Thank you for your interest in contributing!

## Prerequisites

- Python 3.9 or higher
- [Git](https://git-scm.com/)

## Development Setup

```bash
git clone https://github.com/TRI-AMDD/spectrascope.git
cd spectrascope
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

This project uses [black](https://black.readthedocs.io/) for formatting:

```bash
black spectrascope/
```

Pre-commit hooks are configured to run black automatically:

```bash
pre-commit install
```

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests where appropriate.
3. Ensure `pytest tests/` passes.
4. Open a pull request with a clear description of what was changed and why.

## Reporting Issues

Please open a [GitHub issue](https://github.com/TRI-AMDD/spectrascope/issues) with:
- A minimal reproducible example
- Expected vs. actual behavior
- Python and package versions (`pip show spectrascope`)
