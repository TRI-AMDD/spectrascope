import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent / "app"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g. full notebook execution, ~20-35 min).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="slow test — pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
