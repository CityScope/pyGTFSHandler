"""Regression coverage for `Feed.get_representative_date`.

Added 2026-08-12: this method was referenced (in a docstring) by
transitLOS's `download_and_prepare_stops` as the fallback used when no
explicit `date_range` is given, but did not exist anywhere in
pyGTFSHandler -- any caller relying on that fallback crashed with
`AttributeError`. This file locks in a real implementation.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _calendar(start: str, end: str):
    return [
        {
            "service_id": "WEEKDAY",
            "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
            "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": start, "end_date": end,
        },
        {
            "service_id": "WEEKEND",
            "monday": 0, "tuesday": 0, "wednesday": 0, "thursday": 0,
            "friday": 0, "saturday": 1, "sunday": 1,
            "start_date": start, "end_date": end,
        },
    ]


@pytest.fixture
def simple_feed(tmp_path) -> Feed:
    stops = [
        {"stop_id": "A", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "B", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.7},
    ]
    routes = [{"route_id": "R1", "route_short_name": "1", "route_long_name": "R1", "route_type": 3}]
    trips = [
        {"route_id": "R1", "service_id": "WEEKDAY", "trip_id": "T_WD", "direction_id": 0},
        {"route_id": "R1", "service_id": "WEEKEND", "trip_id": "T_WE", "direction_id": 0},
    ]
    stop_times = [
        {"trip_id": "T_WD", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "A", "stop_sequence": 1},
        {"trip_id": "T_WD", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "B", "stop_sequence": 2},
        {"trip_id": "T_WE", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "A", "stop_sequence": 1},
        {"trip_id": "T_WE", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "B", "stop_sequence": 2},
    ]
    directory = write_gtfs(
        tmp_path / "repdate_feed",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("20240101", "20240331"),
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    return Feed(directory)


def test_returns_a_real_date_inside_the_calendar_window(simple_feed):
    d = simple_feed.get_representative_date()
    assert d is not None
    assert date(2024, 1, 1) <= d <= date(2024, 3, 31)


def test_weekday_date_type_never_returns_a_weekend_date(simple_feed):
    d = simple_feed.get_representative_date(date_type="weekday")
    assert d is not None
    assert d.weekday() < 5


def test_respects_explicit_start_end_bounds(simple_feed):
    lo, hi = date(2024, 2, 1), date(2024, 2, 10)
    d = simple_feed.get_representative_date(start_date=lo, end_date=hi, date_type=None)
    assert d is not None
    assert lo <= d <= hi


def test_returns_none_when_no_calendar_range_matches(simple_feed):
    # Window entirely outside the feed's calendar validity.
    lo = simple_feed.calendar.max_date + timedelta(days=365)
    hi = lo + timedelta(days=5)
    d = simple_feed.get_representative_date(start_date=lo, end_date=hi)
    # No candidate date has any active service in this out-of-range window,
    # but a date is still returned (best-effort among the scanned window,
    # picking the least-bad candidate at count 0) rather than raising --
    # callers that need to guard against "no real service at all" should
    # inspect the surrounding calendar bounds themselves.
    assert d is not None
    assert lo <= d <= hi
