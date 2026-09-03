"""Coverage for the real representative-date algorithm in
`pyGTFSHandler.representative_date.select_representative_date` (called from
`Feed.get_representative_date`).

Unlike `test_representative_date.py` (which only locks in the public
method's contract -- signature, bounds, "always returns *a* date"), these
tests construct synthetic feeds with a clearly intended "winner" so the
actual scoring logic is exercised: median trips (unique departure time) per
parent station, per day; median of valid days per ISO week; best week wins;
the day within it with the most services is returned.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _single_date_service_id(d: date) -> str:
    return f"SVC_{d.isoformat().replace('-', '')}"


def _build_feed(tmp_path, day_trip_counts: dict[date, dict[str, int]], *, name="repdate_algo_feed") -> Feed:
    """Builds a synthetic feed where `day_trip_counts[d][station]` is how
    many trips (each at a distinct departure time, one minute apart
    starting at 06:00) serve that `station` (stop_id, no grouping needed --
    `parent_station` defaults to `stop_id`) on date `d`. One dedicated
    `calendar_dates.txt`-only service_id per date (`exception_type=1`) so
    each day's service is fully independent and easy to reason about.
    """
    stations = sorted({s for counts in day_trip_counts.values() for s in counts})
    stops = [
        {"stop_id": s, "stop_name": s, "stop_lat": 40.0 + i * 0.01, "stop_lon": -3.7}
        for i, s in enumerate(stations)
    ]
    routes = [{"route_id": "R1", "route_short_name": "1", "route_long_name": "R1", "route_type": 3}]

    calendar_dates = []
    trips = []
    stop_times = []
    for d, counts in day_trip_counts.items():
        svc = _single_date_service_id(d)
        calendar_dates.append({"service_id": svc, "date": d.strftime("%Y%m%d"), "exception_type": "1"})
        for station, n in counts.items():
            for i in range(n):
                trip_id = f"T_{svc}_{station}_{i}"
                trips.append({"route_id": "R1", "service_id": svc, "trip_id": trip_id, "direction_id": 0})
                hh = 6 + i // 60
                mm = i % 60
                t = f"{hh:02d}:{mm:02d}:00"
                stop_times.append(
                    {
                        "trip_id": trip_id,
                        "arrival_time": t,
                        "departure_time": t,
                        "stop_id": station,
                        "stop_sequence": 1,
                    }
                )

    directory = write_gtfs(
        tmp_path / name,
        {
            "agency.txt": minimal_agency(),
            "calendar_dates.txt": calendar_dates,
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    return Feed(directory)


def _monday_of(year: int, week: int) -> date:
    return date.fromisocalendar(year, week, 1)


def test_picks_the_week_with_higher_and_more_consistent_weekday_service(tmp_path):
    # Week 2: strong, consistent weekday service (8 distinct-time trips/day
    # at each of 2 stations). Weeks 1 and 3: weak weekday service (2
    # trips/day). Weekends everywhere are intentionally left with NO
    # service at all so a `date_type=None` full-week run would still have
    # to fall back to weekdays for any signal.
    strong_week = _monday_of(2024, 3)
    weak_weeks = [_monday_of(2024, 2), _monday_of(2024, 4)]

    day_counts: dict[date, dict[str, int]] = {}
    for monday in weak_weeks:
        for i in range(5):
            day_counts[monday + timedelta(days=i)] = {"S1": 2, "S2": 2}
    for i in range(5):
        day_counts[strong_week + timedelta(days=i)] = {"S1": 8, "S2": 8}

    feed = _build_feed(tmp_path, day_counts)
    picked = feed.get_representative_date(
        start_date=weak_weeks[0], end_date=weak_weeks[-1] + timedelta(days=6), date_type="weekday"
    )
    assert picked is not None
    iso_year, iso_week, iso_weekday = picked.isocalendar()
    assert (iso_year, iso_week) == strong_week.isocalendar()[:2]
    assert iso_weekday <= 5  # a weekday, not a weekend day


def test_weekday_date_type_ignores_a_weekend_heavy_week(tmp_path):
    # Week 1: huge weekend service (would win on raw totals) but weak
    # weekday service. Week 2: solid, consistent weekday service and no
    # weekend service at all. With date_type="weekday", week 2 must win
    # even though week 1 has vastly more *total* trips.
    week1 = _monday_of(2024, 6)
    week2 = _monday_of(2024, 7)

    day_counts: dict[date, dict[str, int]] = {}
    for i in range(5):  # week1 weekdays: weak
        day_counts[week1 + timedelta(days=i)] = {"S1": 1, "S2": 1}
    for i in (5, 6):  # week1 weekend: huge
        day_counts[week1 + timedelta(days=i)] = {"S1": 50, "S2": 50}
    for i in range(5):  # week2 weekdays: solid
        day_counts[week2 + timedelta(days=i)] = {"S1": 6, "S2": 6}

    feed = _build_feed(tmp_path, day_counts)
    picked = feed.get_representative_date(
        start_date=week1, end_date=week2 + timedelta(days=6), date_type="weekday"
    )
    assert picked is not None
    iso_year, iso_week, iso_weekday = picked.isocalendar()
    assert (iso_year, iso_week) == week2.isocalendar()[:2]
    assert iso_weekday <= 5


def test_duplicate_timestamp_trips_are_deduped_by_unique_departure_time(tmp_path):
    # Day A: 6 trips at station S1, all at the SAME departure time (a
    # duplicate/erroneous timestamp collision) -> should score as if only
    # 1 distinct time exists. Day B: 6 trips at 6 DISTINCT times -> scores
    # 6. Without dedup, both days would look identical (6 trips each); with
    # dedup, day B must clearly win.
    day_a = date(2024, 5, 6)  # Monday
    day_b = date(2024, 5, 13)  # Monday, following week

    stations = ["S1", "S2"]
    stops = [{"stop_id": s, "stop_name": s, "stop_lat": 40.0, "stop_lon": -3.7} for s in stations]
    routes = [{"route_id": "R1", "route_short_name": "1", "route_long_name": "R1", "route_type": 3}]

    svc_a, svc_b = _single_date_service_id(day_a), _single_date_service_id(day_b)
    calendar_dates = [
        {"service_id": svc_a, "date": day_a.strftime("%Y%m%d"), "exception_type": "1"},
        {"service_id": svc_b, "date": day_b.strftime("%Y%m%d"), "exception_type": "1"},
    ]
    trips = []
    stop_times = []

    # Day A: 6 duplicate-time trips at S1 (all depart 08:00:00), plus a
    # baseline single trip at S2 so the station has *some* other score too.
    for i in range(6):
        trip_id = f"T_A_{i}"
        trips.append({"route_id": "R1", "service_id": svc_a, "trip_id": trip_id, "direction_id": 0})
        stop_times.append(
            {"trip_id": trip_id, "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1}
        )
    trips.append({"route_id": "R1", "service_id": svc_a, "trip_id": "T_A_S2", "direction_id": 0})
    stop_times.append(
        {"trip_id": "T_A_S2", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S2", "stop_sequence": 1}
    )

    # Day B: 6 trips at S1, each at a distinct minute.
    for i in range(6):
        trip_id = f"T_B_{i}"
        t = f"08:{i:02d}:00"
        trips.append({"route_id": "R1", "service_id": svc_b, "trip_id": trip_id, "direction_id": 0})
        stop_times.append(
            {"trip_id": trip_id, "arrival_time": t, "departure_time": t, "stop_id": "S1", "stop_sequence": 1}
        )
    trips.append({"route_id": "R1", "service_id": svc_b, "trip_id": "T_B_S2", "direction_id": 0})
    stop_times.append(
        {"trip_id": "T_B_S2", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S2", "stop_sequence": 1}
    )

    directory = write_gtfs(
        tmp_path / "dedup_feed",
        {
            "agency.txt": minimal_agency(),
            "calendar_dates.txt": calendar_dates,
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    feed = Feed(directory)

    picked = feed.get_representative_date(start_date=day_a, end_date=day_b, date_type=None)
    assert picked == day_b, (
        "Day B (6 distinct departure times at S1) must beat Day A (6 duplicate-"
        "timestamp trips at S1, i.e. 1 distinct time) once duplicate timestamps "
        "are deduped -- if this fails, the dedup step regressed."
    )
