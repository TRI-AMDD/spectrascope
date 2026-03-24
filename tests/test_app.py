"""Tests for the SpectraScope Streamlit app.

Covers:
- Module imports used by all three app pages
- Home page rendering via Streamlit's AppTest
- Pages 1 and 2 rendering with minimal pre-populated session state
"""

import pathlib
import numpy as np
import pytest

APP_DIR = pathlib.Path(__file__).parent.parent / "app"


# ---------------------------------------------------------------------------
# Import smoke tests — catch any renamed/missing modules immediately
# ---------------------------------------------------------------------------


def test_ssclassification_importable():
    from spectrascope.SSClassification import BinaryClassifications  # noqa: F401


def test_ssplotting_importable():
    from spectrascope.SSPlotting import (  # noqa: F401
        SpectraScopePlotting,
        PlotSpectralFeatures,
        PlotModel,
    )


# ---------------------------------------------------------------------------
# AppTest — renders each page and asserts no unhandled exceptions
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_ss():
    """A minimal SpectraScope instance for session-state pre-population."""
    from spectrascope.SpectraScope import SpectraScope

    X = np.random.rand(10, 20)
    y = np.random.rand(10)
    ss = SpectraScope(X=X, y=y, specnames=["spec0"], yname="target", save_path="/tmp/")
    ss.add_x_axis("spec0", np.linspace(0, 10, 20))
    return ss


def test_home_page_renders():
    """Home page should render its welcome text without raising exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DIR / "home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"home.py raised an exception: {at.exception}"


def test_spectrascope_page_renders(minimal_ss):
    """Page 1 should render with pre-populated session state."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(APP_DIR / "pages" / "1_SpectraScope.py"), default_timeout=30
    )
    at.session_state["data"] = minimal_ss
    at.session_state["yattr"] = "target"
    at.session_state["axis_dict"] = {"spec0": ("x", "y")}
    at.run()
    assert not at.exception, f"1_SpectraScope.py raised an exception: {at.exception}"


def test_visualizations_page_renders(minimal_ss):
    """Page 2 should render with pre-populated session state."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(APP_DIR / "pages" / "2_Visualizations.py"), default_timeout=30
    )
    at.session_state["data"] = minimal_ss
    at.session_state["spec_to_featurize"] = ["spec0"]
    at.session_state["fitted_models"] = []
    at.session_state["yattr"] = "target"
    at.run()
    assert not at.exception, f"2_Visualizations.py raised an exception: {at.exception}"
