"""Tests for `utils.trip_dedupe`: trips with the same parent-station sequence on a common day are one service."""

from __future__ import annotations

import polars as pl
import pytest

from pyGTFSHandler.utils.trip_dedupe import ServiceDays, dedupe_trips, trip_signatures

MON = 20_000 + ((0 - (20_000 + 3)) % 7)   # a Monday as days since 1970-01-01 ((d + 3) % 7 == 0 for Mondays)


def _cal(rows):
    out = []
    for sid, days, a, b in rows:
        out.append({"service_id": sid, **{d: int(d in days) for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}, "start_date": a, "end_date": b})
    return pl.DataFrame(out).lazy()


def _stops():
    # P1 has two platforms (S1a, S1b); every other stop is its own station
    return pl.DataFrame({
        "stop_id": ["S1a", "S1b", "S2", "S3"], "parent_station": ["P1", "P1", None, ""],
    }).lazy()


def _trips(rows):
    return pl.DataFrame({"trip_id": [r[0] for r in rows], "service_id": [r[1] for r in rows]}).lazy()


def _times(trips):
    """trips: (trip_id, [stop ids], first departure seconds)"""
    rows = []
    for tid, stops, t0 in trips:
        for i, s in enumerate(stops):
            rows.append({"trip_id": tid, "stop_id": s, "stop_sequence": i, "departure_time": t0 + 60 * i, "arrival_time": t0 + 60 * i})
    return pl.DataFrame(rows).lazy()


def _freq(rows):
    return pl.DataFrame({
        "trip_id": [r[0] for r in rows], "start_time": [r[1] for r in rows], "end_time": [r[2] for r in rows], "headway_secs": [r[3] for r in rows],
        "n_trips": [max(1, -(-(r[2] - r[1]) // r[3])) for r in rows],
    }).lazy()


WEEK = ("monday", "tuesday", "wednesday", "thursday", "friday")
CAL = _cal([("A", WEEK, MON, MON + 365), ("B", WEEK, MON, MON + 365), ("SUN", ("sunday",), MON, MON + 365)])


def _ids(lf):
    return sorted(lf.collect()["trip_id"].unique().to_list())


def test_signature_uses_parent_stations_and_collapses_repeats():
    st = _times([("t1", ["S1a", "S1b", "S2"], 0), ("t2", ["S1b", "S2"], 0)])
    sig = trip_signatures(st, _stops()).sort("trip_id")
    assert sig["sig"].to_list() == ["P1|S2", "P1|S2"]


def test_scheduled_trip_inside_frequency_window_is_deleted():
    trips = _trips([("F", "A"), ("T", "A"), ("OUT", "A")])
    st = _times([("F", ["S1a", "S2", "S3"], 0), ("T", ["S1b", "S2", "S3"], 30_000), ("OUT", ["S1a", "S2", "S3"], 50_000)])
    t, s, f, rep = dedupe_trips(trips, st, _stops(), _freq([("F", 28_800, 36_000, 600)]), CAL)
    assert _ids(t) == ["F", "OUT"] and _ids(s) == ["F", "OUT"]
    assert rep["scheduled_in_frequency"] == 1
    assert f.collect().height == 1


def test_scheduled_trip_on_other_days_is_kept():
    trips = _trips([("F", "A"), ("T", "SUN")])
    st = _times([("F", ["S1a", "S2"], 0), ("T", ["S1a", "S2"], 30_000)])
    t, *_ = dedupe_trips(trips, st, _stops(), _freq([("F", 28_800, 36_000, 600)]), CAL)
    assert _ids(t) == ["F", "T"]


def test_different_station_sequence_is_kept():
    trips = _trips([("F", "A"), ("T", "A")])
    st = _times([("F", ["S1a", "S2", "S3"], 0), ("T", ["S1a", "S3"], 30_000)])
    t, *_ = dedupe_trips(trips, st, _stops(), _freq([("F", 28_800, 36_000, 600)]), CAL)
    assert _ids(t) == ["F", "T"]


def test_overlapping_frequencies_are_joined_into_one():
    trips = _trips([("F1", "A"), ("F2", "A")])
    st = _times([("F1", ["S1a", "S2"], 0), ("F2", ["S1b", "S2"], 0)])
    t, s, f, rep = dedupe_trips(trips, st, _stops(), _freq([("F1", 28_800, 36_000, 600), ("F2", 32_400, 43_200, 900)]), CAL)
    fr = f.collect()
    assert _ids(t) == ["F1"] and _ids(s) == ["F1"]
    assert fr.height == 1 and fr["start_time"][0] == 28_800 and fr["end_time"][0] == 43_200
    assert fr["headway_secs"][0] == 900                      # headway of the longer window (F2 = 10800 s vs F1 = 7200 s)
    assert fr["n_trips"][0] == 16                            # (43200 - 28800) / 900
    assert rep["frequencies_joined"] == 1


def test_disjoint_frequency_windows_stay_separate():
    trips = _trips([("F1", "A"), ("F2", "A")])
    st = _times([("F1", ["S1a", "S2"], 0), ("F2", ["S1b", "S2"], 0)])
    t, _, f, _ = dedupe_trips(trips, st, _stops(), _freq([("F1", 28_800, 36_000, 600), ("F2", 40_000, 50_000, 900)]), CAL)
    assert _ids(t) == ["F1", "F2"] and f.collect().height == 2


def test_frequencies_of_different_days_are_not_joined():
    trips = _trips([("F1", "A"), ("F2", "SUN")])
    st = _times([("F1", ["S1a", "S2"], 0), ("F2", ["S1a", "S2"], 0)])
    t, _, f, _ = dedupe_trips(trips, st, _stops(), _freq([("F1", 28_800, 36_000, 600), ("F2", 30_000, 40_000, 600)]), CAL)
    assert _ids(t) == ["F1", "F2"] and f.collect().height == 2


def test_repeated_scheduled_trips_collapse_but_distant_ones_stay():
    trips = _trips([("a", "A"), ("b", "A"), ("c", "A")])
    st = _times([("a", ["S1a", "S2"], 30_000), ("b", ["S1b", "S2"], 30_030), ("c", ["S1a", "S2"], 31_000)])
    t, *_, rep = dedupe_trips(trips, st, _stops(), None, CAL)
    assert _ids(t) == ["a", "c"] and rep["scheduled_repeated"] == 1


def test_calendar_exceptions_decide_overlap():
    cal = _cal([("A", ("monday",), MON, MON)])
    exc = pl.DataFrame({"service_id": ["B"], "date": [MON], "exception_type": [1]}).lazy()
    sd = ServiceDays(cal, exc)
    assert sd.overlap("A", "B") and not sd.overlap("A", "UNKNOWN")
    rem = pl.DataFrame({"service_id": ["A"], "date": [MON], "exception_type": [2]}).lazy()
    assert not ServiceDays(cal, rem).days("A")


def test_no_duplicates_returns_inputs_unchanged():
    trips = _trips([("a", "A"), ("b", "A")])
    st = _times([("a", ["S1a", "S2"], 30_000), ("b", ["S1a", "S3"], 30_000)])
    t, s, f, rep = dedupe_trips(trips, st, _stops(), None, CAL)
    assert _ids(t) == ["a", "b"] and f is None and sum(rep.values()) == 0


def test_helper_trips_derived_from_a_frequency_template_are_kept():
    trips = _trips([("TPL", "A"), ("TPL_night", "A"), ("TPL_48", "A")])
    st = _times([("TPL", ["S1a", "S2"], 0), ("TPL_night", ["S1a", "S2"], 0), ("TPL_48", ["S1a", "S2"], 0)])
    freq = pl.DataFrame({
        "trip_id": ["TPL_night", "TPL"], "start_time": [0, 79_200], "end_time": [7_200, 85_799], "headway_secs": [1_800, 1_800],
        "orig_trip_id": ["TPL", "TPL"], "n_trips": [4, 4],
    }).lazy()
    t, _, f, rep = dedupe_trips(trips, st, _stops(), freq, CAL)
    assert _ids(t) == ["TPL", "TPL_48", "TPL_night"] and f.collect().height == 2 and sum(rep.values()) == 0


def test_different_service_ids_are_never_duplicates_even_with_overlapping_days():
    trips = _trips([("F", "A"), ("T", "B"), ("x", "A"), ("y", "B")])   # A and B both run on weekdays
    st = _times([("F", ["S1a", "S2"], 0), ("T", ["S1a", "S2"], 30_000), ("x", ["S1a", "S3"], 40_000), ("y", ["S1b", "S3"], 40_010)])
    t, *_ = dedupe_trips(trips, st, _stops(), _freq([("F", 28_800, 36_000, 600)]), CAL)
    assert _ids(t) == ["F", "T", "x", "y"]


def test_same_service_id_across_loaded_files_counts_as_the_same_service():
    cal = _cal([("WK_file_0", WEEK, MON, MON + 365), ("WK_file_4", WEEK, MON, MON + 365)])
    trips = _trips([("a_file_0", "WK_file_0"), ("a_file_4", "WK_file_4")])
    st = _times([("a_file_0", ["S1a", "S2"], 30_000), ("a_file_4", ["S1b", "S2"], 30_000)])
    t, *_ = dedupe_trips(trips, st, _stops(), None, cal)
    assert _ids(t) == ["a_file_0"]


def test_single_day_mode_ignores_service_ids():
    trips = _trips([("a", "A"), ("b", "B")])
    st = _times([("a", ["S1a", "S2"], 30_000), ("b", ["S1b", "S2"], 30_020)])
    t, *_ = dedupe_trips(trips, st, _stops(), None, CAL, single_day=True)
    assert _ids(t) == ["a"]
    t, *_ = dedupe_trips(trips, st, _stops(), None, CAL)       # without a selected day different services are kept
    assert _ids(t) == ["a", "b"]


# ---- Feed level: the deduplication runs once a service day is selected ----
from datetime import date

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _feed_dir(tmp_path):
    cal = lambda sid, **d: {"service_id": sid, **{k: int(d.get(k, 0)) for k in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}, "start_date": "20241101", "end_date": "20241231"}
    st = lambda tid, s1, t0, t1: [
        {"trip_id": tid, "arrival_time": t0, "departure_time": t0, "stop_id": s1, "stop_sequence": 1},
        {"trip_id": tid, "arrival_time": t1, "departure_time": t1, "stop_id": "S2", "stop_sequence": 2},
    ]
    return write_gtfs(tmp_path / "f", {
        "agency.txt": minimal_agency(),
        "calendar.txt": [cal("WK", monday=1, tuesday=1, wednesday=1, thursday=1, friday=1), cal("WK2", monday=1, tuesday=1, wednesday=1, thursday=1, friday=1), cal("SUN", sunday=1)],
        "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "L1", "route_type": 3}],
        "stops.txt": [{"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7}, {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71}],
        "trips.txt": [{"route_id": "R1", "service_id": "WK", "trip_id": "t1"}, {"route_id": "R1", "service_id": "WK2", "trip_id": "t2"},
                      {"route_id": "R1", "service_id": "SUN", "trip_id": "t3"}],
        "stop_times.txt": st("t1", "S1", "08:00:00", "08:10:00") + st("t2", "S1", "08:00:20", "08:10:20") + st("t3", "S1", "08:00:00", "08:10:00"),
    })


def test_feed_dedupes_when_one_service_day_is_loaded(tmp_path):
    d = _feed_dir(tmp_path)
    wed = date(2024, 11, 6)
    f = Feed(d, start_date=wed, end_date=wed)
    assert f.trips.lf.collect()["trip_id"].to_list() == ["t1"]       # t2 (other service id, same day, same stops/time) is the same trip
    assert f.trip_dedupe_report["scheduled_repeated"] == 1


def test_feed_over_a_week_only_dedupes_after_a_day_is_selected(tmp_path):
    d = _feed_dir(tmp_path)
    f = Feed(d, start_date=date(2024, 11, 4), end_date=date(2024, 11, 10))
    assert sorted(f.trips.lf.collect()["trip_id"].to_list()) == ["t1", "t2", "t3"]
    rep = f.dedupe_trips_for_day(date(2024, 11, 6))
    assert rep["scheduled_repeated"] == 1
    assert sorted(f.trips.lf.collect()["trip_id"].to_list()) == ["t1", "t3"]     # Sunday's trip never ran on the selected Wednesday
    f.dedupe_trips_for_day(date(2024, 11, 6))                                    # idempotent
    assert sorted(f.trips.lf.collect()["trip_id"].to_list()) == ["t1", "t3"]


def test_different_modes_with_the_same_stops_and_times_are_not_one_service():
    trips = pl.DataFrame({"trip_id": ["bus", "rail"], "service_id": ["A", "A"], "route_id": ["RB", "RR"]}).lazy()
    st = _times([("bus", ["S1a", "S2"], 30_000), ("rail", ["S1b", "S2"], 30_000)])
    routes = pl.DataFrame({"route_id": ["RB", "RR"], "route_type": [3, 2]}).lazy()
    t, *_ = dedupe_trips(trips, st, _stops(), None, CAL, single_day=True, routes=routes)
    assert _ids(t) == ["bus", "rail"]
    same = pl.DataFrame({"route_id": ["RB", "RR"], "route_type": [3, 3]}).lazy()
    t, *_ = dedupe_trips(trips, st, _stops(), None, CAL, single_day=True, routes=same)
    assert _ids(t) == ["bus"]
