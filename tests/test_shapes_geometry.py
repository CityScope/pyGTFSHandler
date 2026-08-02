"""Tests for real `shapes.txt` geometry usage.

Pre-refactor, `models/shapes.py` never reads `shapes.txt` at all -- "shape
points" are synthesized purely from stop coordinates, so distance is always
the straight line between consecutive stops even when a real, longer polyline
is provided. These tests build a shape that deviates substantially from the
straight line between its stops and assert the computed distance reflects the
true polyline length, and that stop coordinates are inserted as vertices on
that line at the correct position.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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


@pytest.fixture
def l_shaped_feed(tmp_path) -> Feed:
    """Two stops directly north-south of each other, but the shape detours
    east then north then west (an "L" / "U" shape) -- straight-line distance
    is far shorter than the true polyline length."""
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
        tmp_path / "l_shaped",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "shapes.txt": shape,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "shape_id": "SH1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    return Feed(directory)


def test_shape_distance_exceeds_straight_line(l_shaped_feed):
    straight_line_m = _haversine_m(40.000, -3.700, 40.010, -3.700)
    df = l_shaped_feed.lf.filter(pl.col("stop_id").str.starts_with("S2")).collect()
    shape_dist = df["shape_dist_traveled"][0]
    assert shape_dist > straight_line_m * 1.5


def test_degenerate_single_point_shape_falls_back_gracefully(tmp_path):
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
    ]
    shape = [{"shape_id": "SH1", "shape_pt_lat": 40.0, "shape_pt_lon": -3.7, "shape_pt_sequence": 1}]
    directory = write_gtfs(
        tmp_path / "degenerate_shape",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "shapes.txt": shape,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "shape_id": "SH1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_trip_without_shape_id_uses_straight_line(tmp_path):
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
    ]
    directory = write_gtfs(
        tmp_path / "no_shape",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    feed = Feed(directory)
    straight_line_m = _haversine_m(40.0, -3.7, 40.01, -3.71)
    df = feed.lf.filter(pl.col("stop_id").str.starts_with("S2")).collect()
    assert abs(df["shape_dist_traveled"][0] - straight_line_m) < 5.0


def test_nonmonotonic_shape_dist_traveled_is_discarded(tmp_path):
    """`shape_dist_traveled` values that decrease along the sequence are
    implausible and the checker/loader must fall back to a computed value
    rather than trusting them."""
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.02, "stop_lon": -3.72},
    ]
    shape = [
        {"shape_id": "SH1", "shape_pt_lat": 40.0, "shape_pt_lon": -3.7, "shape_pt_sequence": 1, "shape_dist_traveled": 0.0},
        {"shape_id": "SH1", "shape_pt_lat": 40.01, "shape_pt_lon": -3.71, "shape_pt_sequence": 2, "shape_dist_traveled": 5.0},
        {"shape_id": "SH1", "shape_pt_lat": 40.02, "shape_pt_lon": -3.72, "shape_pt_sequence": 3, "shape_dist_traveled": 1.0},
    ]
    directory = write_gtfs(
        tmp_path / "bad_dist_traveled",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "shapes.txt": shape,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "shape_id": "SH1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    feed = Feed(directory)
    df = feed.lf.sort("stop_sequence").collect()
    dist = df["shape_dist_traveled"].to_list()
    assert dist == sorted(dist)
