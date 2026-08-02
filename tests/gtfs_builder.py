"""Helpers for constructing synthetic GTFS feeds on disk for tests.

Why this module exists: authoring dozens of edge-case GTFS folders by hand as
static fixture files (as the pre-existing `tests/sample_gtfs_*` folders were
built) does not scale to the breadth of scenarios this test suite needs
(malformed rows, multi-day trips, colliding IDs across feeds, etc.). Instead,
tests build a GTFS feed in a `tmp_path` directory from plain Python dicts of
row-lists, one dict key per GTFS filename. This keeps each test's fixture data
next to the assertions that depend on it, and makes edge cases trivial to
construct (e.g. a single missing value, a single malformed time string).

How it works: `write_gtfs(directory, files)` takes a mapping of
``{"stops.txt": [{"stop_id": "S1", ...}, ...], ...}`` and writes one CSV per
entry using the union of keys across all row-dicts for that file as the
header (so not every row needs every optional column). Missing keys in a row
become empty CSV fields, matching how real-world GTFS producers often leave
optional fields blank.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def write_gtfs(directory: Path, files: Mapping[str, Sequence[Mapping[str, object]]]) -> Path:
    """Writes a synthetic GTFS feed (as a folder of CSV files) to disk.

    Args:
        directory: Destination folder. Created if missing.
        files: Mapping of GTFS filename (e.g. ``"stops.txt"``) to a sequence
            of row dicts. The header for each file is the union of keys
            across all its rows, in first-seen order.

    Returns:
        The same `directory`, as a `Path`, for convenient chaining.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for filename, rows in files.items():
        header: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in header:
                    header.append(key)

        with open(directory / filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return directory


def minimal_agency() -> list[dict]:
    """Returns a single valid `agency.txt` row, reused by most test feeds."""
    return [
        {
            "agency_id": "AG1",
            "agency_name": "Test Agency",
            "agency_url": "http://example.com",
            "agency_timezone": "Europe/Madrid",
        }
    ]
