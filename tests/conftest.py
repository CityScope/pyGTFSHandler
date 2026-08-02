"""Shared pytest fixtures for the pyGTFSHandler test suite.

Small, single-purpose synthetic feeds are built inline in each test module
via `tests.gtfs_builder.write_gtfs` (a `tmp_path`-backed helper) rather than
checked-in static fixture folders -- this keeps a scenario's exact data next
to the assertions that depend on it and makes edge cases trivial to
construct. The only fixtures kept here are for the two real-world Sevilla
feeds under `tests/sevilla_data`, used by the `slow`-marked smoke tests that
exercise the full pipeline against real, large GTFS data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SEVILLA_DIR = TESTS_DIR / "sevilla_data"
# The same two feeds, as plain folders, also live under
# `examples/test_files/sevilla/` (copied there so `examples/route_map_example.ipynb`
# has real data to load) -- `route_map`-focused tests use that copy instead of
# the zips here, per the example notebook.
EXAMPLES_SEVILLA_DIR = TESTS_DIR.parent / "examples" / "test_files" / "sevilla"


@pytest.fixture
def tussam_zip() -> Path:
    matches = list(SEVILLA_DIR.glob("*TUSSAM*.zip"))
    if not matches:
        pytest.skip("Sevilla TUSSAM zip not present")
    return matches[0]


@pytest.fixture
def metro_sevilla_zip() -> Path:
    matches = list(SEVILLA_DIR.glob("*Metro_Sevilla*.zip"))
    if not matches:
        pytest.skip("Sevilla Metro zip not present")
    return matches[0]


@pytest.fixture
def tussam_dir() -> Path:
    path = EXAMPLES_SEVILLA_DIR / "TUSSAM"
    if not path.is_dir():
        pytest.skip("examples/test_files/sevilla/TUSSAM not present")
    return path


@pytest.fixture
def metro_sevilla_dir() -> Path:
    path = EXAMPLES_SEVILLA_DIR / "Metro_Sevilla"
    if not path.is_dir():
        pytest.skip("examples/test_files/sevilla/Metro_Sevilla not present")
    return path
