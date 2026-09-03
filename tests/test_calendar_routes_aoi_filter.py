"""Tests for AOI filtering of `calendar.txt`/`calendar_dates.txt`/`routes.txt`
via the `service_id`/`route_id` cascade off AOI-restricted trip_ids.

Mirrors the existing `trip_ids_touching_stops` cascade (AOI stops -> AOI
trip_ids) one hop further: once `Feed.load` knows which trip_ids actually
touch the AOI, `io.service_and_route_ids_for_trips` cheaply looks up which
service_ids/route_ids those trips reference, so `Calendar.load`/
`Routes.load` only ever read the calendar/route rows the AOI-restricted
trip set actually uses -- instead of reading every operator's full
calendar.txt/routes.txt (e.g. a nationwide operator's) before any of it
gets discarded.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from pyGTFSHandler.feed import Feed
from pyGTFSHandler.utils import io

from .gtfs_builder import minimal_agency, write_gtfs


def _aoi() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(-3.72, 39.99, -3.69, 40.01)], crs="EPSG:4326")


@pytest.fixture
def two_route_two_service_feed(tmp_path) -> str:
    """Two routes/services: one entirely inside the AOI, one entirely
    outside it (its stop, trip, service, and route never come near the AOI
    bbox at all)."""
    stops = [
        {"stop_id": "S_IN1", "stop_name": "In1", "stop_lat": 40.000, "stop_lon": -3.700},
        {"stop_id": "S_IN2", "stop_name": "In2", "stop_lat": 40.005, "stop_lon": -3.700},
        {"stop_id": "S_OUT1", "stop_name": "Out1", "stop_lat": 50.000, "stop_lon": 10.000},
        {"stop_id": "S_OUT2", "stop_name": "Out2", "stop_lat": 50.005, "stop_lon": 10.000},
    ]
    calendar = [
        {
            "service_id": "SVC_IN", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
            "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231",
        },
        {
            "service_id": "SVC_OUT", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
            "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20240101", "end_date": "20241231",
        },
    ]
    routes = [
        {"route_id": "R_IN", "route_short_name": "In", "route_long_name": "Inside Line", "route_type": 3},
        {"route_id": "R_OUT", "route_short_name": "Out", "route_long_name": "Outside Line", "route_type": 3},
    ]
    trips = [
        {"route_id": "R_IN", "service_id": "SVC_IN", "trip_id": "T_IN"},
        {"route_id": "R_OUT", "service_id": "SVC_OUT", "trip_id": "T_OUT"},
    ]
    stop_times = [
        {"trip_id": "T_IN", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S_IN1", "stop_sequence": 1},
        {"trip_id": "T_IN", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S_IN2", "stop_sequence": 2},
        {"trip_id": "T_OUT", "arrival_time": "09:00:00", "departure_time": "09:00:00", "stop_id": "S_OUT1", "stop_sequence": 1},
        {"trip_id": "T_OUT", "arrival_time": "09:10:00", "departure_time": "09:10:00", "stop_id": "S_OUT2", "stop_sequence": 2},
    ]
    directory = write_gtfs(
        tmp_path / "two_route_two_service",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": calendar,
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    return str(directory)


def test_aoi_filters_out_routes_never_touching_aoi(two_route_two_service_feed):
    feed = Feed(two_route_two_service_feed, aoi=_aoi())
    route_ids = set(feed.routes.lf.select("route_id").collect().to_series().to_list())
    assert route_ids == {"R_IN"}


def test_aoi_filters_out_services_never_touching_aoi(two_route_two_service_feed):
    feed = Feed(two_route_two_service_feed, aoi=_aoi())
    service_ids = set(feed.calendar.lf.select("service_id").collect().to_series().to_list())
    assert service_ids == {"SVC_IN"}


def test_no_aoi_keeps_both_routes_and_services(two_route_two_service_feed):
    feed = Feed(two_route_two_service_feed)
    route_ids = set(feed.routes.lf.select("route_id").collect().to_series().to_list())
    service_ids = set(feed.calendar.lf.select("service_id").collect().to_series().to_list())
    assert route_ids == {"R_IN", "R_OUT"}
    assert service_ids == {"SVC_IN", "SVC_OUT"}


def test_service_and_route_ids_for_trips_helper(two_route_two_service_feed):
    """Direct unit-level check of the new `io.service_and_route_ids_for_trips`
    cascade helper, independent of the full `Feed` pipeline."""
    service_ids, route_ids = io.service_and_route_ids_for_trips(
        [two_route_two_service_feed], ["T_IN"],
    )
    assert set(service_ids) == {"SVC_IN"}
    assert set(route_ids) == {"R_IN"}

    service_ids, route_ids = io.service_and_route_ids_for_trips(
        [two_route_two_service_feed], ["T_IN", "T_OUT"],
    )
    assert set(service_ids) == {"SVC_IN", "SVC_OUT"}
    assert set(route_ids) == {"R_IN", "R_OUT"}

    service_ids, route_ids = io.service_and_route_ids_for_trips(
        [two_route_two_service_feed], [],
    )
    assert service_ids == []
    assert route_ids == []
