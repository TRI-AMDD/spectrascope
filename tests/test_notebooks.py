"""Tests for the Grapes and XANES+PDF example notebooks.

Fast tests (always run):
  test_*_syntax        — every code cell parses as valid Python
  test_*_structure     — expected section headers are present

Slow tests (opt-in via ``pytest -m slow``):
  test_*_executes      — full notebook execution via nbconvert
                         Grapes: ~20 min, XANES: ~15 min

Run slow tests:
  pytest -m slow tests/test_notebooks.py

Skip slow tests (default):
  pytest tests/test_notebooks.py
"""

import ast
import json
import pathlib
import shutil
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

GRAPES_NB = EXAMPLES_DIR / "Grapes_example.ipynb"
XANES_NB = EXAMPLES_DIR / "XANES_PDF_example.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_notebook(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def code_cells(nb: dict) -> list[tuple[int, str]]:
    """Return (index, source) for every code cell."""
    result = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            result.append((i, "".join(cell["source"])))
    return result


def strip_magics(source: str) -> str:
    """Remove IPython magic lines (%...) so ast.parse works."""
    return "\n".join(
        line for line in source.split("\n") if not line.startswith("%")
    )


def markdown_text(nb: dict) -> str:
    """Concatenate all markdown cell source for header checks."""
    return "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "markdown"
    )


# ---------------------------------------------------------------------------
# Fast: syntax
# ---------------------------------------------------------------------------

def _check_syntax(nb_path: pathlib.Path):
    nb = load_notebook(nb_path)
    errors = []
    for idx, src in code_cells(nb):
        clean = strip_magics(src)
        try:
            ast.parse(clean)
        except SyntaxError as exc:
            errors.append(f"Cell {idx}: {exc} — {src[:80]!r}")
    assert not errors, "\n".join(errors)


def test_grapes_notebook_syntax():
    _check_syntax(GRAPES_NB)


def test_xanes_notebook_syntax():
    _check_syntax(XANES_NB)


# ---------------------------------------------------------------------------
# Fast: structure (key sections present)
# ---------------------------------------------------------------------------

GRAPES_EXPECTED_SECTIONS = [
    "Load Data",
    "Preprocess",
    "Initialize SpectraScope",
    "Generate Spectral Features",
    "Set Up Hyperparameter",
    "Train Scikit-Learn Models",
    "Train Fused Lasso",
    "Train LCEN",
    "Correlation Analysis",
    "Random Variable",
    "LCEN Hyperparameter Sensitivity",
]

XANES_EXPECTED_SECTIONS = [
    "Load Data",
    "Initialize SpectraScope",
    "Generate",
    "Combine Multi-Modal",
    "Set Up Hyperparameter",
    "Model Training",
    "Benchmarking",
    "Correlation Analysis",
    "Random Variable",
    "Supplementary",
]


def _check_structure(nb_path: pathlib.Path, expected_sections: list[str]):
    nb = load_notebook(nb_path)
    md = markdown_text(nb)
    missing = [s for s in expected_sections if s not in md]
    assert not missing, f"Missing sections in {nb_path.name}: {missing}"


def test_grapes_notebook_structure():
    _check_structure(GRAPES_NB, GRAPES_EXPECTED_SECTIONS)


def test_xanes_notebook_structure():
    _check_structure(XANES_NB, XANES_EXPECTED_SECTIONS)


# ---------------------------------------------------------------------------
# Fast: save paths are relative (no absolute personal paths)
# ---------------------------------------------------------------------------

def _check_save_paths(nb_path: pathlib.Path):
    nb = load_notebook(nb_path)
    bad_patterns = ["/Users/", "/home/", "C:\\Users\\", "C:/Users/"]
    violations = []
    for idx, src in code_cells(nb):
        for pattern in bad_patterns:
            if pattern in src:
                for line in src.split("\n"):
                    if pattern in line:
                        violations.append(f"Cell {idx}: {line.strip()}")
    assert not violations, (
        f"Absolute personal paths found in {nb_path.name}:\n"
        + "\n".join(violations)
    )


def test_grapes_save_paths_are_relative():
    _check_save_paths(GRAPES_NB)


def test_xanes_save_paths_are_relative():
    _check_save_paths(XANES_NB)


# ---------------------------------------------------------------------------
# Fast: no empty code cells
# ---------------------------------------------------------------------------

def _check_no_empty_cells(nb_path: pathlib.Path):
    nb = load_notebook(nb_path)
    empty = [i for i, src in code_cells(nb) if src.strip() == ""]
    assert not empty, f"Empty code cells at indices {empty} in {nb_path.name}"


def test_grapes_no_empty_cells():
    _check_no_empty_cells(GRAPES_NB)


def test_xanes_no_empty_cells():
    _check_no_empty_cells(XANES_NB)


# ---------------------------------------------------------------------------
# Slow: full notebook execution via nbconvert
# ---------------------------------------------------------------------------

def _execute_notebook(nb_path: pathlib.Path, timeout: int = 2400):
    """Execute a notebook in-place; raise on any cell error.

    The notebook runs with its own directory as the working directory so
    that relative paths (e.g. ``../tests/test_files/``) resolve correctly.
    """
    try:
        import nbformat
        from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
    except ImportError:
        pytest.skip("nbconvert not installed — skipping execution test")

    nb = nbformat.read(nb_path, as_version=4)

    # Prepend a cell that forces the non-interactive Agg backend so
    # matplotlib doesn't try to open display windows during execution.
    agg_cell = nbformat.v4.new_code_cell(
        "import matplotlib\nmatplotlib.use('Agg')"
    )
    nb.cells.insert(0, agg_cell)

    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
        extra_arguments=["--HistoryManager.enabled=False"],
    )

    # Run from the examples/ directory so relative paths resolve correctly.
    run_dir = str(nb_path.parent)
    try:
        ep.preprocess(nb, {"metadata": {"path": run_dir}})
    except CellExecutionError as exc:
        pytest.fail(
            f"Cell execution error in {nb_path.name}:\n{exc.traceback}"
        )


def _ensure_xanes_corr_csv():
    """Create a minimal placeholder XANES_corr.csv if it doesn't exist yet.

    Grapes_example.ipynb reads this file for the XANES comparison plot.
    It is normally generated by running XANES_PDF_example.ipynb.  When the
    XANES slow test is skipped (e.g. SPA not installed), we create a stub
    with the correct columns so the Grapes execution test can still run.
    """
    import pandas as pd

    figures_dir = EXAMPLES_DIR / "Figures"
    figures_dir.mkdir(exist_ok=True)
    csv_path = figures_dir / "XANES_corr.csv"
    if csv_path.exists():
        return
    # Minimal stub: correct columns, a few placeholder rows
    stub = pd.DataFrame(
        {"r": [0.5, 0.4], "MIF": [0.3, 0.2], "RF": [2.0, 1.8], "LCEN": [1.9, 1.7]},
        index=["XANES, Intensity", "XANES, Polynomial"],
    )
    stub.index.name = None
    stub.to_csv(csv_path)


# XANES is listed first so it runs before Grapes and produces XANES_corr.csv.
@pytest.mark.slow
def test_xanes_notebook_executes():
    """Execute XANES_PDF_example.ipynb end-to-end (~15 min).

    Skipped automatically if the optional SPA package is not installed.
    """
    try:
        import SPA  # noqa: F401
    except ImportError:
        pytest.skip("SPA (SmartProcessAnalytics) not installed")
    _execute_notebook(XANES_NB, timeout=2400)


@pytest.mark.slow
def test_grapes_notebook_executes():
    """Execute Grapes_example.ipynb end-to-end (~20 min).

    Requires Figures/XANES_corr.csv (produced by test_xanes_notebook_executes).
    If the XANES test was skipped, a minimal stub CSV is created automatically.
    """
    _ensure_xanes_corr_csv()
    _execute_notebook(GRAPES_NB, timeout=2400)
