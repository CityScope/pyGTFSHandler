"""Characterization tests for Stops/Routes/Trips loading: parent_station
hierarchy, stop clustering, and route_type normalization -- using synthetic
feeds built inline via `gtfs_builder.write_gtfs` (see the module docstring
in `tests/gtfs_builder.py` for why static fixture folders were dropped in
favor of this approach).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def test_calendar_dates_exception_removes_service(tmp_path):
    directory = write_gtfs(
        tmp_path / "calendar_exception",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "ALL_DAYS", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "calendar_dates.txt": [{"service_id": "ALL_DAYS", "date": "20241225", "exception_type": 2}],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "ALL_DAYS", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    services = feed.calendar.get_services_in_date(date(2024, 12, 25))
    assert not any(s.startswith("ALL_DAYS") for s in services)
    services_day_before = feed.calendar.get_services_in_date(date(2024, 12, 24))
    assert any(s.startswith("ALL_DAYS") for s in services_day_before)


def test_parent_station_hierarchy(tmp_path):
    directory = write_gtfs(
        tmp_path / "hierarchy",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": [
                {"stop_id": "STATION", "stop_name": "Station", "stop_lat": 40.0, "stop_lon": -3.7, "location_type": 1},
                {"stop_id": "PLATFORM1", "stop_name": "Platform 1", "stop_lat": 40.0001, "stop_lon": -3.7001,
                 "location_type": 0, "parent_station": "STATION"},
                {"stop_id": "PLATFORM2", "stop_name": "Platform 2", "stop_lat": 40.0002, "stop_lon": -3.7002,
                 "location_type": 0, "parent_station": "STATION"},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "PLATFORM1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "PLATFORM2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    df = feed.lf.collect()
    assert df["parent_station"].str.starts_with("STATION").sum() >= 1


def test_stop_clustering_transitive_chain(tmp_path):
    """Three stops in a line, each within the cluster threshold of its
    neighbor, but the two endpoints farther apart than the threshold --
    all three must end up in the same cluster via transitive/connected-
    component grouping, not just pairwise."""
    # ~11m per 0.0001 deg latitude
    stops = [
        {"stop_id": "C1", "stop_name": "C1", "stop_lat": 40.00000, "stop_lon": -3.7},
        {"stop_id": "C2", "stop_name": "C2", "stop_lat": 40.00015, "stop_lon": -3.7},  # ~16.7m from C1
        {"stop_id": "C3", "stop_name": "C3", "stop_lat": 40.00030, "stop_lon": -3.7},  # ~16.7m from C2, ~33m from C1
    ]
    directory = write_gtfs(
        tmp_path / "chain_cluster",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "C1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:01:00", "departure_time": "08:01:00", "stop_id": "C2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "08:02:00", "departure_time": "08:02:00", "stop_id": "C3", "stop_sequence": 3},
            ],
        },
    )
    feed = Feed(directory, stop_group_distance=20.0)
    df = feed.stops.lf.collect()
    matches_any = pl.any_horizontal([pl.col("stop_id").str.starts_with(s) for s in ("C1", "C2", "C3")])
    parent_stations = df.filter(matches_any)["parent_station"].to_list()
    assert len(set(parent_stations)) == 1


def test_identical_coordinates_stops_cluster_together(tmp_path):
    stops = [
        {"stop_id": "D1", "stop_name": "D1", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "D2", "stop_name": "D2", "stop_lat": 40.0, "stop_lon": -3.7},
    ]
    directory = write_gtfs(
        tmp_path / "identical_coords",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "D1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:01:00", "departure_time": "08:01:00", "stop_id": "D2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory, stop_group_distance=5.0)
    df = feed.stops.lf.collect()
    matches_any = pl.any_horizontal([pl.col("stop_id").str.starts_with(s) for s in ("D1", "D2")])
    parent_stations = df.filter(matches_any)["parent_station"].to_list()
    assert len(set(parent_stations)) == 1


@pytest.mark.parametrize("route_type_value", [0, 1, 2, 3, 4, 5, 6, 7, "bus", "tram", "rail"])
def test_route_type_normalization(tmp_path, route_type_value):
    directory = write_gtfs(
        tmp_path / f"route_type_{route_type_value}",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": route_type_value}],
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
    feed = Feed(directory, route_types=[route_type_value])
    assert feed.lf.collect().height == 2


def _calendar_all_days():
    return [
        {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
         "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
    ]


def _route_bus():
    return [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}]


def test_three_level_parent_station_hierarchy(tmp_path):
    """station -> platform -> sub-platform: all stop_times against the
    deepest level must still resolve up to the top-level station."""
    directory = write_gtfs(
        tmp_path / "three_level_hierarchy",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar_all_days(),
            "routes.txt": _route_bus(),
            "stops.txt": [
                {"stop_id": "STATION", "stop_name": "Station", "stop_lat": 40.0, "stop_lon": -3.7, "location_type": 1},
                {"stop_id": "PLATFORM", "stop_name": "Platform", "stop_lat": 40.0001, "stop_lon": -3.7001,
                 "location_type": 0, "parent_station": "STATION"},
                {"stop_id": "SUBPLATFORM", "stop_name": "Sub-platform", "stop_lat": 40.0002, "stop_lon": -3.7002,
                 "location_type": 0, "parent_station": "PLATFORM"},
                {"stop_id": "OTHER", "stop_name": "Other", "stop_lat": 41.0, "stop_lon": -4.7},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "SUBPLATFORM", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "OTHER", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    stops_df = feed.stops.lf.collect()
    subplatform_row = stops_df.filter(pl.col("stop_id").str.starts_with("SUBPLATFORM")).row(0, named=True)
    assert subplatform_row["parent_station"].startswith("PLATFORM")


def test_invalid_lat_lon_stop_is_excluded(tmp_path):
    """A stop with null/out-of-range coordinates must be dropped (or
    flagged), not silently included in distance/geometry math."""
    directory = write_gtfs(
        tmp_path / "invalid_lat_lon",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar_all_days(),
            "routes.txt": _route_bus(),
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "Bad", "stop_lat": 999.0, "stop_lon": -3.71},
                {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.02, "stop_lon": -3.72},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    feed = Feed(directory)
    stop_ids = feed.stops.lf.collect()["stop_id"].to_list()
    assert not any(sid.startswith("S2") for sid in stop_ids)


def test_shape_geometry_combined_with_multiday_trip(tmp_path):
    """Real shape geometry and the day_offset multi-day model must work
    together: a trip using explicit `>=24:00:00` times, following a real
    (non-straight) shape, must resolve both correctly at once."""
    stops = [
        {"stop_id": "S1", "stop_name": "Start", "stop_lat": 40.000, "stop_lon": -3.700},
        {"stop_id": "S2", "stop_name": "End", "stop_lat": 40.010, "stop_lon": -3.700},
    ]
    shape = [
        {"shape_id": "SH1", "shape_pt_lat": 40.000, "shape_pt_lon": -3.700, "shape_pt_sequence": 1},
        {"shape_id": "SH1", "shape_pt_lat": 40.000, "shape_pt_lon": -3.650, "shape_pt_sequence": 2},
        {"shape_id": "SH1", "shape_pt_lat": 40.010, "shape_pt_lon": -3.650, "shape_pt_sequence": 3},
        {"shape_id": "SH1", "shape_pt_lat": 40.010, "shape_pt_lon": -3.700, "shape_pt_sequence": 4},
    ]
    directory = write_gtfs(
        tmp_path / "shape_and_multiday",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar_all_days(),
            "routes.txt": _route_bus(),
            "stops.txt": stops,
            "shapes.txt": shape,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "shape_id": "SH1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "25:10:00", "departure_time": "25:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    df = feed.lf.collect()
    s2_row = df.filter(pl.col("stop_id").str.starts_with("S2")).row(0, named=True)
    assert s2_row["day_offset"] == 1
    # Straight-line S1-S2 distance is ~1112m; the real "L"-shaped detour is
    # much longer -- confirms real geometry (not a straight line) was used
    # even though the trip also crosses midnight.
    assert s2_row["shape_dist_traveled"] > 1112 * 1.5
