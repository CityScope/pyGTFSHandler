"""Characterization tests for basic feed loading: folder vs zip autodetection,
and loading the real Sevilla feeds end-to-end.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _minimal_feed_dir(tmp_path) -> Path:
    return write_gtfs(
        tmp_path / "minimal",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {
                    "service_id": "SVC",
                    "monday": 1,
                    "tuesday": 1,
                    "wednesday": 1,
                    "thursday": 1,
                    "friday": 1,
                    "saturday": 1,
                    "sunday": 1,
                    "start_date": "20240101",
                    "end_date": "20241231",
                }
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )


def test_load_from_folder(tmp_path):
    directory = _minimal_feed_dir(tmp_path)
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_load_from_zip(tmp_path):
    directory = _minimal_feed_dir(tmp_path)
    zip_path = tmp_path / "minimal_feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in directory.glob("*.txt"):
            zf.write(file, arcname=file.name)
    feed = Feed(zip_path)
    assert feed.lf.collect().height == 2


def test_load_from_zip_with_nested_folder(tmp_path):
    """Some GTFS publishers zip the files under a subfolder instead of at
    the zip root; loading must still find them."""
    directory = _minimal_feed_dir(tmp_path)
    zip_path = tmp_path / "nested_feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in directory.glob("*.txt"):
            zf.write(file, arcname=f"gtfs_subfolder/{file.name}")
    feed = Feed(zip_path)
    assert feed.lf.collect().height == 2


def test_folder_and_zip_produce_identical_results(tmp_path):
    directory = _minimal_feed_dir(tmp_path)
    zip_path = tmp_path / "equiv_feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in directory.glob("*.txt"):
            zf.write(file, arcname=file.name)
    from_folder = Feed(directory).lf.collect().sort("stop_id")
    from_zip = Feed(zip_path).lf.collect().sort("stop_id")
    assert from_folder.select("stop_id", "arrival_time", "departure_time").equals(
        from_zip.select("stop_id", "arrival_time", "departure_time")
    )


@pytest.mark.slow
def test_tussam_feed_loads(tussam_zip):
    feed = Feed(tussam_zip)
    assert feed.lf.collect().height > 0


@pytest.mark.slow
def test_tussam_feed_loads_within_time_budget(tussam_zip):
    """Performance tripwire (per the plan's lazy-pushdown pass): TUSSAM's
    ~450k-row, 14MB `stop_times.txt` is the largest file in the local test
    data. Construction should stay comfortably under a few seconds -- a
    regression back to eagerly re-scanning/re-parsing the CSV multiple times
    (as `StopTimes.load` used to, before its early lazy checkpoint) would
    blow well past this budget. The bound is generous on purpose: this is a
    tripwire for gross regressions, not a tight micro-benchmark.
    """
    import time

    start = time.perf_counter()
    feed = Feed(tussam_zip)
    row_count = feed.lf.collect().height
    elapsed = time.perf_counter() - start

    assert row_count > 0
    assert elapsed < 8.0, f"Feed construction + collect took {elapsed:.2f}s (budget: 8.0s)"


@pytest.mark.slow
def test_metro_sevilla_feed_loads(metro_sevilla_zip):
    feed = Feed(metro_sevilla_zip)
    assert feed.lf.collect().height > 0


@pytest.mark.slow
def test_both_sevilla_feeds_together_no_id_cross_contamination(tussam_zip, metro_sevilla_zip):
    feed = Feed([tussam_zip, metro_sevilla_zip])
    assert feed.lf.collect().height > 0


@pytest.mark.slow
def test_route_map_smoke(tussam_dir, metro_sevilla_dir):
    """Smoke test only: `route_map` should return a folium.Map for a normal
    weekday within both real Sevilla feeds' calendar validity range, without
    raising. No assertions on the map's contents -- see `route_map`'s own
    docstring/`examples/route_map_example.ipynb` for what it's expected to
    show. Uses the same `examples/test_files/sevilla/` folder copy the
    example notebook loads from, not the zips under `tests/sevilla_data`.
    """
    import datetime

    folium = pytest.importorskip("folium")

    from pyGTFSHandler.maps import route_map

    feed = Feed([tussam_dir, metro_sevilla_dir])
    the_date = datetime.date(2026, 7, 6)  # a plain Monday within both feeds' validity range

    m = route_map(feed, the_date)

    assert isinstance(m, folium.Map)
