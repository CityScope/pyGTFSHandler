"""Tests that `Feed`'s constructor-time filters (`start_time`/`end_time`,
`route_types`, `service_ids`, `trip_ids`, `stop_ids`, `route_ids`, `aoi`,
`start_date`/`end_date`) never *over*-filter -- i.e. never drop a row they
shouldn't -- in the presence of the trickier data shapes covered elsewhere in
this suite: `frequencies.txt`-only trips, stop_times crossing past 24h (both
explicitly and via implicit wraparound), and stop_times with `None`
intermediate times that must be interpolated before any time-range filter is
applied to them.

These matter because `Feed` is designed to push these filters down as early
as possible (filtering `stops.txt`/`calendar.txt`/`routes.txt`/`trips.txt`
before ever reading the much larger `stop_times.txt`) for performance on
large feeds -- an early, over-eager filter based on incomplete information
(e.g. filtering stop_times by a time window before null times are
interpolated, or before a multi-day stop's real day is known) would silently
drop legitimate rows.
"""

from __future__ import annotations

from datetime import time

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _calendar():
    return [
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
    ]


def _routes():
    return [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}]


def _stops():
    return [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.02, "stop_lon": -3.72},
    ]


def test_start_end_time_filter_keeps_frequency_trip_overlapping_window(tmp_path):
    """A constructor-time `start_time`/`end_time` window that overlaps a
    frequency-only trip's window must keep it, even though the trip has no
    single "departure_time" of its own -- only a `frequencies.txt` window."""
    directory = write_gtfs(
        tmp_path / "freq_initial_filter",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "TPL"}],
            "stop_times.txt": [
                {"trip_id": "TPL", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "TPL", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "TPL", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
            "frequencies.txt": [
                {"trip_id": "TPL", "start_time": "08:00:00", "end_time": "09:00:00", "headway_secs": 1800, "exact_times": 0},
            ],
        },
    )
    # Window [08:30, 08:45] falls inside the frequency's [08:00, 09:00]
    # window but outside any single stop_times row -- the whole template
    # trip must still be kept, not dropped for lacking an exact-time match.
    feed = Feed(directory, start_time=time(8, 30), end_time=time(8, 45))
    assert feed.lf.collect().height > 0


def test_start_end_time_filter_keeps_explicit_multiday_stop_after_interpolation(tmp_path):
    """A stop with a `None` time, on a trip using explicit `>=24:00:00`
    encoding, must be interpolated *before* the constructor-time
    `start_time`/`end_time` window is applied -- filtering on the raw (still
    null) time first would incorrectly drop it."""
    directory = write_gtfs(
        tmp_path / "multiday_initial_filter",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "25:10:00", "departure_time": "25:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    # 00:30 (00:30:00) falls within the (interpolated) 00:00-00:10-ish window
    # of the middle stop's time_of_day, once day_offset is resolved.
    feed = Feed(directory, start_time=time(0, 0), end_time=time(1, 0))
    df = feed.lf.collect()
    assert df.filter(pl.col("stop_id").str.starts_with("S2")).height > 0


def test_route_type_filter_at_construction_keeps_all_stops_of_matching_route(tmp_path):
    """Filtering by `route_types` at construction time must not accidentally
    drop stops that are only reachable via the matching route because
    `stops.txt` was pre-filtered too aggressively before `routes.txt`/
    `trips.txt` were even read."""
    directory = write_gtfs(
        tmp_path / "route_type_initial_filter",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": [
                {"route_id": "R_BUS", "route_short_name": "B", "route_long_name": "Bus", "route_type": 3},
                {"route_id": "R_RAIL", "route_short_name": "R", "route_long_name": "Rail", "route_type": 2},
            ],
            "stops.txt": _stops(),
            "trips.txt": [
                {"route_id": "R_BUS", "service_id": "SVC", "trip_id": "T_BUS"},
                {"route_id": "R_RAIL", "service_id": "SVC", "trip_id": "T_RAIL"},
            ],
            "stop_times.txt": [
                {"trip_id": "T_BUS", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T_BUS", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T_RAIL", "arrival_time": "09:00:00", "departure_time": "09:00:00", "stop_id": "S3", "stop_sequence": 1},
            ],
        },
    )
    feed = Feed(directory, route_types=[3])
    df = feed.lf.collect()
    assert df.filter(pl.col("stop_id").str.starts_with("S1")).height > 0
    assert df.filter(pl.col("stop_id").str.starts_with("S2")).height > 0
    assert df.filter(pl.col("stop_id").str.starts_with("S3")).height == 0


def test_trip_ids_filter_keeps_frequency_window_for_selected_trip(tmp_path):
    """Filtering by an explicit `trip_ids` list at construction time must
    keep the selected trip's `frequencies.txt` window intact."""
    directory = write_gtfs(
        tmp_path / "trip_ids_freq_filter",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [
                {"route_id": "R1", "service_id": "SVC", "trip_id": "TPL_KEEP"},
                {"route_id": "R1", "service_id": "SVC", "trip_id": "TPL_DROP"},
            ],
            "stop_times.txt": [
                {"trip_id": "TPL_KEEP", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "TPL_KEEP", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "TPL_DROP", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "TPL_DROP", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
            "frequencies.txt": [
                {"trip_id": "TPL_KEEP", "start_time": "08:00:00", "end_time": "09:00:00", "headway_secs": 1800, "exact_times": 0},
                {"trip_id": "TPL_DROP", "start_time": "08:00:00", "end_time": "09:00:00", "headway_secs": 1800, "exact_times": 0},
            ],
        },
    )
    feed = Feed(directory, trip_ids=["TPL_KEEP"])
    df = feed.lf.collect()
    trip_ids = set(df["trip_id"].to_list())
    assert len(trip_ids) == 1
    assert next(iter(trip_ids)).startswith("TPL_KEEP")
    assert df.height > 0
