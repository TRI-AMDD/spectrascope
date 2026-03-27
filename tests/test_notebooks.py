"""Static checks for the Grapes and XANES+PDF example notebooks.

These tests verify notebook quality without executing them:
  test_*_syntax        — every code cell parses as valid Python
  test_*_structure     — expected section headers are present
  test_*_save_paths    — no hardcoded absolute paths
  test_*_no_empty      — no empty code cells

For pipeline coverage, see test_integration.py which exercises the
same SpectraScope workflow (load → featurize → train → select) as
proper pytest tests.
"""

import ast
import json
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

GRAPES_NB = EXAMPLES_DIR / "GRAPES.ipynb"
XANES_NB = EXAMPLES_DIR / "XANES_PDF.ipynb"


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
    "Set Up Hyperparameter",
    "Train Scikit-Learn and LCEN Models",
    "Random Variable Control Experiment",
    "Correlation Analysis",
    "LCEN Hyperparameter Sensitivity",
    "Figure 4",
    "Figure 5",
    "Figure 6",
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
