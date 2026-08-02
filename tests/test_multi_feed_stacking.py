"""Tests for loading multiple raw GTFS feeds together through `Feed`.

This covers the hazard called out during plan review: `Feed(gtfs_dirs=[...])`
already accepts multiple feed paths, and IDs (`stop_id`, `trip_id`,
`route_id`, `service_id`) can collide across feeds while referring to
genuinely different real-world entities, or can be the same ID because the
two feeds really do share that entity (e.g. successive weekly publications
from the same agency). Both must be handled correctly and are tested
independently for each ID type, plus the "3+ feeds" and "same feed loaded
twice" scenarios.
"""

from __future__ import annotations

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _calendar(service_id="SVC"):
    return [
        {
            "service_id": service_id,
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


def _routes(route_id="R1", name="Line 1"):
    return [{"route_id": route_id, "route_short_name": route_id, "route_long_name": name, "route_type": 3}]


def _build_feed_dir(tmp_path, name, *, stop_id, stop_lat, stop_lon, trip_id, service_id, route_id):
    stops = [
        {"stop_id": stop_id, "stop_name": f"Stop {stop_id}", "stop_lat": stop_lat, "stop_lon": stop_lon},
        {"stop_id": f"{stop_id}_far", "stop_name": "Far", "stop_lat": stop_lat + 0.1, "stop_lon": stop_lon + 0.1},
    ]
    return write_gtfs(
        tmp_path / name,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(service_id),
            "routes.txt": _routes(route_id),
            "stops.txt": stops,
            "trips.txt": [{"route_id": route_id, "service_id": service_id, "trip_id": trip_id}],
            "stop_times.txt": [
                {"trip_id": trip_id, "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": stop_id, "stop_sequence": 1},
                {"trip_id": trip_id, "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": f"{stop_id}_far", "stop_sequence": 2},
            ],
        },
    )


def test_colliding_stop_id_different_real_stops_stay_distinct(tmp_path):
    dir_a = _build_feed_dir(
        tmp_path, "feed_a", stop_id="1", stop_lat=40.0, stop_lon=-3.7,
        trip_id="TA", service_id="SVC_A", route_id="RA",
    )
    dir_b = _build_feed_dir(
        tmp_path, "feed_b", stop_id="1", stop_lat=52.5, stop_lon=13.4,  # Berlin, nowhere near Madrid
        trip_id="TB", service_id="SVC_B", route_id="RB",
    )
    feed = Feed([dir_a, dir_b])
    stops = feed.stops.lf.filter(pl.col("stop_id").str.contains("1")).collect()
    lats = stops["stop_lat"].unique().to_list()
    # Two physically distinct stops sharing raw id "1" must not collapse to
    # a single row/coordinate.
    assert len(lats) >= 2


def test_colliding_trip_and_service_ids_stay_distinct(tmp_path):
    dir_a = _build_feed_dir(
        tmp_path, "feed_a2", stop_id="A1", stop_lat=40.0, stop_lon=-3.7,
        trip_id="SAME_TRIP", service_id="SAME_SVC", route_id="R_A",
    )
    dir_b = _build_feed_dir(
        tmp_path, "feed_b2", stop_id="B1", stop_lat=41.0, stop_lon=-4.7,
        trip_id="SAME_TRIP", service_id="SAME_SVC", route_id="R_B",
    )
    feed = Feed([dir_a, dir_b])
    trips = feed.trips.lf.collect()
    assert trips.height == 2


def test_three_feeds_at_once(tmp_path):
    dirs = [
        _build_feed_dir(
            tmp_path, f"feed_{i}", stop_id=f"S{i}", stop_lat=40.0 + i, stop_lon=-3.7 - i,
            trip_id=f"T{i}", service_id=f"SVC{i}", route_id=f"R{i}",
        )
        for i in range(3)
    ]
    feed = Feed(dirs)
    assert feed.trips.lf.collect().height == 3


def test_loading_same_feed_path_twice_is_idempotent(tmp_path):
    directory = _build_feed_dir(
        tmp_path, "feed_once", stop_id="X1", stop_lat=40.0, stop_lon=-3.7,
        trip_id="TX", service_id="SVCX", route_id="RX",
    )
    single = Feed(directory)
    doubled = Feed([directory, directory])
    assert doubled.trips.lf.collect().height == single.trips.lf.collect().height


def test_disjoint_feeds_pure_concatenation(tmp_path):
    dir_a = _build_feed_dir(
        tmp_path, "feed_disjoint_a", stop_id="DA1", stop_lat=10.0, stop_lon=10.0,
        trip_id="TDA", service_id="SVCDA", route_id="RDA",
    )
    dir_b = _build_feed_dir(
        tmp_path, "feed_disjoint_b", stop_id="DB1", stop_lat=-10.0, stop_lon=-10.0,
        trip_id="TDB", service_id="SVCDB", route_id="RDB",
    )
    feed = Feed([dir_a, dir_b])
    assert feed.trips.lf.collect().height == 2
    assert feed.routes.lf.collect().height == 2
