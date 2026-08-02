"""Characterization tests for AOI / date / time / route-type filtering."""

from __future__ import annotations

from datetime import date, time

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _feed_dir(tmp_path):
    return write_gtfs(
        tmp_path / "filter_feed",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {"service_id": "SVC", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231"}
            ],
            "routes.txt": [
                {"route_id": "R_BUS", "route_short_name": "B", "route_long_name": "Bus Line", "route_type": 3},
                {"route_id": "R_RAIL", "route_short_name": "R", "route_long_name": "Rail Line", "route_type": 2},
            ],
            "stops.txt": [
                {"stop_id": "IN1", "stop_name": "Inside 1", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "IN2", "stop_name": "Inside 2", "stop_lat": 40.001, "stop_lon": -3.701},
                {"stop_id": "OUT1", "stop_name": "Outside", "stop_lat": 45.0, "stop_lon": 10.0},
            ],
            "trips.txt": [
                {"route_id": "R_BUS", "service_id": "SVC", "trip_id": "T_BUS"},
                {"route_id": "R_RAIL", "service_id": "SVC", "trip_id": "T_RAIL"},
            ],
            "stop_times.txt": [
                {"trip_id": "T_BUS", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "IN1", "stop_sequence": 1},
                {"trip_id": "T_BUS", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "IN2", "stop_sequence": 2},
                {"trip_id": "T_RAIL", "arrival_time": "20:00:00", "departure_time": "20:00:00", "stop_id": "IN1", "stop_sequence": 1},
                {"trip_id": "T_RAIL", "arrival_time": "20:10:00", "departure_time": "20:10:00", "stop_id": "OUT1", "stop_sequence": 2},
            ],
        },
    )


def test_route_type_filter(tmp_path):
    directory = _feed_dir(tmp_path)
    feed = Feed(directory, route_types=[3])
    df = feed.lf.collect()
    route_ids = set(df["route_id"].to_list())
    assert len(route_ids) == 1
    assert next(iter(route_ids)).startswith("R_BUS")


def test_aoi_filter_excludes_outside_stop(tmp_path):
    directory = _feed_dir(tmp_path)
    aoi = gpd.GeoDataFrame(geometry=[box(-3.72, 39.99, -3.69, 40.01)], crs="EPSG:4326")
    feed = Feed(directory, aoi=aoi)
    stop_ids = feed.stops.lf.collect()["stop_id"].to_list()
    assert "OUT1" not in stop_ids


def test_aoi_filter_excluding_all_stops_raises(tmp_path):
    directory = _feed_dir(tmp_path)
    aoi = gpd.GeoDataFrame(geometry=[box(80.0, 80.0, 81.0, 81.0)], crs="EPSG:4326")
    with pytest.raises(Exception):
        Feed(directory, aoi=aoi)


def test_time_range_filter(tmp_path):
    directory = _feed_dir(tmp_path)
    feed = Feed(directory)
    filtered = feed.filter_by_time_range(time(7, 0), time(9, 0)).collect()
    trip_ids = set(filtered["trip_id"].to_list())
    assert len(trip_ids) == 1
    assert next(iter(trip_ids)).startswith("T_BUS")


def test_date_filter_returns_all_active_services(tmp_path):
    directory = _feed_dir(tmp_path)
    feed = Feed(directory)
    filtered = feed.filter_by_date(date(2024, 6, 1)).collect()
    assert filtered.height > 0


def test_stop_ids_filter(tmp_path):
    directory = _feed_dir(tmp_path)
    feed = Feed(directory, stop_ids=["IN1", "IN2"])
    stop_ids = feed.stops.lf.collect()["stop_id"].to_list()
    assert "OUT1" not in stop_ids
