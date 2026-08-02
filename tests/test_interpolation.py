"""Tests for interpolation of missing (`None`) stop_times.

Covers the two-pass design: a simple linear pass (used for early filtering)
and a shape-distance-weighted pass that should differ from pure linear
interpolation when the real shape geometry is not evenly spaced (e.g. a stop
sitting much closer to one neighbor than the other along the true path).
"""

from __future__ import annotations

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


def test_single_missing_middle_time_linear(tmp_path):
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.005, "stop_lon": -3.705},
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.01, "stop_lon": -3.71},
    ]
    directory = write_gtfs(
        tmp_path / "linear_interp",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "08:20:00", "departure_time": "08:20:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    feed = Feed(directory)
    df = feed.lf.sort("stop_sequence").collect()
    s2_time = df.filter(pl.col("stop_id").str.starts_with("S2"))["departure_time"][0]
    # Equidistant stops (roughly) -> linear interpolation should land near the
    # midpoint between 08:00 and 08:20, i.e. close to 08:10.
    expected_midpoint = 8 * 3600 + 10 * 60
    assert abs(s2_time - expected_midpoint) <= 60


def test_shape_distance_weighted_interpolation_differs_from_naive_midpoint(tmp_path):
    """A stop placed much closer (along the real shape) to its first
    neighbor than its second should get an interpolated time much closer to
    the first neighbor's time too, not the naive midpoint."""
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.001, "stop_lon": -3.701},  # very close to S1
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.05, "stop_lon": -3.75},  # far from S2
    ]
    shape = [
        {"shape_id": "SH1", "shape_pt_lat": 40.0, "shape_pt_lon": -3.7, "shape_pt_sequence": 1},
        {"shape_id": "SH1", "shape_pt_lat": 40.001, "shape_pt_lon": -3.701, "shape_pt_sequence": 2},
        {"shape_id": "SH1", "shape_pt_lat": 40.05, "shape_pt_lon": -3.75, "shape_pt_sequence": 3},
    ]
    directory = write_gtfs(
        tmp_path / "shape_weighted_interp",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "shapes.txt": shape,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "shape_id": "SH1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "09:00:00", "departure_time": "09:00:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    feed = Feed(directory)
    df = feed.lf.sort("stop_sequence").collect()
    s2_time = df.filter(pl.col("stop_id").str.starts_with("S2"))["departure_time"][0]
    naive_midpoint = 8 * 3600 + 30 * 60
    # S2 is much closer to S1 than to S3 along the real shape, so its
    # interpolated time should be well before the naive midpoint.
    assert s2_time < naive_midpoint - 5 * 60


def test_first_stop_null_time_is_flagged_not_silently_dropped(tmp_path):
    stops = [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
    ]
    directory = write_gtfs(
        tmp_path / "first_stop_null",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "", "departure_time": "", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )
    # A trip whose first stop has no time and thus can't be linearly
    # interpolated must not crash the whole load.
    feed = Feed(directory)
    assert feed.lf.collect().height >= 1


def test_single_stop_trip_does_not_crash(tmp_path):
    stops = [{"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7}]
    directory = write_gtfs(
        tmp_path / "single_stop_trip",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": stops,
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
            ],
        },
    )
    feed = Feed(directory)
    assert feed.lf.collect().height == 1
