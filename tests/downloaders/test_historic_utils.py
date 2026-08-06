"""Tests for `pyGTFSHandler.downloaders.utils.dates` and `.historic`."""

from datetime import datetime

import pytest

from pyGTFSHandler.downloaders.utils.dates import normalize_date_range
from pyGTFSHandler.downloaders.utils.historic import select_versions_covering_range


def test_normalize_date_range_both_none():
    assert normalize_date_range(None, None) == (None, None)


def test_normalize_date_range_fills_missing_end_with_start():
    start, end = normalize_date_range("2024-02-01", None)
    assert start == end == datetime(2024, 2, 1)


def test_normalize_date_range_accepts_iso_strings():
    start, end = normalize_date_range("2024-02-01", "2024-03-01")
    assert start == datetime(2024, 2, 1)
    assert end == datetime(2024, 3, 1)


def test_normalize_date_range_rejects_dd_mm_yyyy():
    # Unlike `downloaders.spain.utils.input_date`, this expects ISO dates.
    with pytest.raises(ValueError):
        normalize_date_range("01-02-2024", None)


def test_select_versions_keeps_entries_inside_range_plus_preceding_one():
    entries = [
        {"id": "before", "start": datetime(2023, 12, 1)},
        {"id": "in_range_1", "start": datetime(2024, 1, 10)},
        {"id": "in_range_2", "start": datetime(2024, 2, 1)},
        {"id": "after", "start": datetime(2024, 6, 1)},
    ]
    selected = select_versions_covering_range(
        entries, datetime(2024, 1, 1), datetime(2024, 3, 1), get_start=lambda e: e["start"]
    )
    ids = [e["id"] for e in selected]
    assert ids == ["before", "in_range_1", "in_range_2"]


def test_select_versions_no_preceding_entry_when_range_starts_before_all():
    entries = [
        {"id": "a", "start": datetime(2024, 2, 1)},
        {"id": "b", "start": datetime(2024, 3, 1)},
    ]
    selected = select_versions_covering_range(
        entries, datetime(2024, 1, 1), datetime(2024, 4, 1), get_start=lambda e: e["start"]
    )
    ids = [e["id"] for e in selected]
    assert ids == ["a", "b"]


def test_select_versions_skips_entries_with_no_start_date():
    entries = [
        {"id": "no_date", "start": None},
        {"id": "has_date", "start": datetime(2024, 1, 15)},
    ]
    selected = select_versions_covering_range(
        entries, datetime(2024, 1, 1), datetime(2024, 2, 1), get_start=lambda e: e["start"]
    )
    assert [e["id"] for e in selected] == ["has_date"]
