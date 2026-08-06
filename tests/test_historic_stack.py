"""Tests for `pyGTFSHandler.utils.stack_gtfs.historic_stack`.

`historic_stack` is the tool that merges several *successive* GTFS
publications from the same agency/dataset (e.g. weekly NAP publications
covering different, non-overlapping date windows) into a single physical
GTFS folder, namespacing every id by the publication's `_start_date_...`
suffix (parsed from each input folder's name) so ids don't collide across
publications, while still reconciling `service_id`s and folding
`frequencies.txt` into `stop_times.txt`.

These tests build small synthetic "publications" with `gtfs_builder.write_gtfs`
directly under folder names ending in `_start_date_YYYYMMDD` (the suffix
`historic_stack`/`load_stack` requires to tag rows with their origin file),
then assert on the merged output written to disk.
"""

from __future__ import annotations

import polars as pl
import pytest

from pyGTFSHandler.utils.stack_gtfs import historic_stack

from .gtfs_builder import minimal_agency, write_gtfs


def _calendar(service_id, start_date, end_date):
    return [
        {
            "service_id": service_id,
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 0,
            "sunday": 0,
            "start_date": start_date,
            "end_date": end_date,
        }
    ]


def _publication(tmp_path, start_date, *, service_id, trip_id, route_id="R1"):
    """Build one synthetic GTFS publication folder valid from `start_date`."""
    directory = tmp_path / f"feed_start_date_{start_date}"
    stops = [
        {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 40.01, "stop_lon": -3.69},
    ]
    return write_gtfs(
        directory,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(service_id, start_date, "20241231"),
            "routes.txt": [
                {"route_id": route_id, "route_short_name": route_id, "route_long_name": "Line", "route_type": 3}
            ],
            "stops.txt": stops,
            "trips.txt": [{"route_id": route_id, "service_id": service_id, "trip_id": trip_id}],
            "stop_times.txt": [
                {"trip_id": trip_id, "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": trip_id, "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )


def test_historic_stack_merges_two_publications(tmp_path):
    dir_a = _publication(tmp_path, "20240101", service_id="SVC_A", trip_id="TRIP_A")
    dir_b = _publication(tmp_path, "20240108", service_id="SVC_B", trip_id="TRIP_B")

    out_dir = tmp_path / "merged"
    historic_stack([dir_a, dir_b], out_dir)

    assert (out_dir / "stops.txt").exists()
    assert (out_dir / "trips.txt").exists()
    assert (out_dir / "stop_times.txt").exists()
    assert (out_dir / "calendar.txt").exists()

    trips = pl.read_csv(out_dir / "trips.txt")
    # One trip per publication, each namespaced by its file's start date.
    assert trips.height == 2
    trip_ids = set(trips["trip_id"].to_list())
    assert any("20240101" in t for t in trip_ids)
    assert any("20240108" in t for t in trip_ids)

    stop_times = pl.read_csv(out_dir / "stop_times.txt")
    # 2 stop_times rows per trip, 2 trips.
    assert stop_times.height == 4

    stops = pl.read_csv(out_dir / "stops.txt")
    # Same two physical stops appear in both publications.
    assert stops.height == 4


def test_historic_stack_merges_three_publications(tmp_path):
    dirs = [
        _publication(tmp_path, d, service_id=f"SVC_{i}", trip_id=f"TRIP_{i}")
        for i, d in enumerate(["20240101", "20240108", "20240115"])
    ]

    out_dir = tmp_path / "merged3"
    historic_stack(dirs, out_dir)

    trips = pl.read_csv(out_dir / "trips.txt")
    assert trips.height == 3

    calendar = pl.read_csv(out_dir / "calendar.txt")
    assert calendar.height == 3


def test_historic_stack_folds_frequencies_into_stop_times(tmp_path):
    start_date = "20240101"
    directory = tmp_path / f"freqfeed_start_date_{start_date}"
    trip_id = "TRIP_F"
    write_gtfs(
        directory,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("SVC_F", start_date, "20241231"),
            "routes.txt": [
                {"route_id": "RF", "route_short_name": "RF", "route_long_name": "Line F", "route_type": 3}
            ],
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 40.01, "stop_lon": -3.69},
            ],
            "trips.txt": [{"route_id": "RF", "service_id": "SVC_F", "trip_id": trip_id}],
            "stop_times.txt": [
                {"trip_id": trip_id, "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": trip_id, "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
            "frequencies.txt": [
                {"trip_id": trip_id, "start_time": "06:00:00", "end_time": "10:00:00", "headway_secs": 600},
            ],
        },
    )

    out_dir = tmp_path / "merged_freq"
    historic_stack([directory], out_dir)

    assert (out_dir / "frequencies.txt").exists()
    frequencies = pl.read_csv(out_dir / "frequencies.txt")
    assert frequencies.height == 1


def test_historic_stack_overwrites_existing_output_dir(tmp_path):
    dir_a = _publication(tmp_path, "20240101", service_id="SVC_A", trip_id="TRIP_A")
    out_dir = tmp_path / "merged_overwrite"
    out_dir.mkdir()
    (out_dir / "stale_file.txt").write_text("stale")

    historic_stack([dir_a], out_dir)

    assert not (out_dir / "stale_file.txt").exists()
    assert (out_dir / "trips.txt").exists()
