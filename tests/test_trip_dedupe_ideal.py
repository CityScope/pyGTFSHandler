"""Idealised end-to-end checks: a regular service published several ways must give the headway of ONE copy once the day is selected."""
from __future__ import annotations

from datetime import date, time

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs

DAY = date(2024, 11, 6)       # Wednesday
WEEK = {"monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0}
LONS = [-3.700, -3.690, -3.680]


def _hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _trip_rows(tid, t0):
    return [{"trip_id": tid, "arrival_time": _hms(t0 + i * 300), "departure_time": _hms(t0 + i * 300), "stop_id": f"S{i}", "stop_sequence": i + 1} for i in range(3)]


def _feed(tmp_path, *, copies=(), headway=600, base=True):
    """Base service: R1 every `headway` s from 06:00 to 10:00 (service WK). `copies`: names of republications to add."""
    routes, trips, times, freqs, cal = [], [], [], [], [{"service_id": "WK", **WEEK, "start_date": "20240101", "end_date": "20241231"}]
    def route(rid):
        routes.append({"route_id": rid, "route_short_name": rid, "route_long_name": rid, "route_type": 3})
    def service(sid):
        cal.append({"service_id": sid, **WEEK, "start_date": "20240101", "end_date": "20241231"})
    if base:
        route("R1")
        for k in range(int(4 * 3600 // headway)):
            trips.append({"route_id": "R1", "service_id": "WK", "trip_id": f"b{k}", "direction_id": 0})
            times += _trip_rows(f"b{k}", 6 * 3600 + k * headway)
    for c in copies:
        rid, sid = f"R_{c}", f"WK_{c}"
        route(rid); service(sid)
        if c in ("exact", "offset"):
            off = 0 if c == "exact" else 20
            for k in range(int(4 * 3600 // headway)):
                trips.append({"route_id": rid, "service_id": sid, "trip_id": f"{c}{k}", "direction_id": 0})
                times += _trip_rows(f"{c}{k}", 6 * 3600 + k * headway + off)
        elif c == "freq":          # the same service as one frequency template
            trips.append({"route_id": rid, "service_id": sid, "trip_id": f"{c}T", "direction_id": 0})
            times += _trip_rows(f"{c}T", 6 * 3600)
            freqs.append({"trip_id": f"{c}T", "start_time": "06:00:00", "end_time": "10:00:00", "headway_secs": headway, "exact_times": 1})
        elif c == "freq_split":    # two overlapping frequency windows of the same service on two templates
            for name, a, b in (("A", "06:00:00", "09:00:00"), ("B", "08:00:00", "10:00:00")):
                trips.append({"route_id": rid, "service_id": sid, "trip_id": f"{c}{name}", "direction_id": 0})
                times += _trip_rows(f"{c}{name}", 6 * 3600)
                freqs.append({"trip_id": f"{c}{name}", "start_time": a, "end_time": b, "headway_secs": headway, "exact_times": 1})
    files = {
        "agency.txt": minimal_agency(), "calendar.txt": cal, "routes.txt": routes,
        "stops.txt": [{"stop_id": f"S{i}", "stop_name": f"S{i}", "stop_lat": 40.0, "stop_lon": lon} for i, lon in enumerate(LONS)],
        "trips.txt": trips, "stop_times.txt": times,
    }
    if freqs:
        files["frequencies.txt"] = freqs
    return write_gtfs(tmp_path, files)


def _headway(feed):
    h = feed.get_headway_at_stops(DAY, time(6, 0), time(10, 0), by="shape_direction", at="parent_station", how="best")
    h = h.collect() if hasattr(h, "collect") else h
    return h.filter(pl.col("parent_station") == "S1")["headway"].min()


def _load(tmp_path, **kw):
    return Feed(_feed(tmp_path, **kw), start_date=DAY, end_date=DAY, stop_group_distance=150)


def test_the_ideal_base_service_has_its_headway(tmp_path):
    assert _headway(_load(tmp_path)) == pytest.approx(10.0, rel=0.03)


@pytest.mark.parametrize("copies", [("exact",), ("exact", "exact"), ("offset",), ("exact", "offset")])
def test_scheduled_copies_collapse_to_one_service(tmp_path, copies):
    base = _headway(_load(tmp_path / "base"))
    # the same name twice is the same folder: build distinct copies by suffixing
    f = _load(tmp_path / "dup", copies=tuple(dict.fromkeys(copies)))
    assert _headway(f) == pytest.approx(base, rel=0.03)
    assert f.trips.lf.select(pl.len()).collect().item() == 24          # 4 h every 10 min, nothing extra left
    assert f.trip_dedupe_report["scheduled_repeated"] >= 24


def test_without_deduplication_a_copy_with_small_offsets_would_change_the_headway(tmp_path):
    base = _headway(_load(tmp_path / "base"))
    f = Feed(_feed(tmp_path / "raw", copies=("offset",)), start_date=DAY, end_date=DAY, stop_group_distance=150, dedupe_trips=False)
    assert abs(_headway(f) - base) > 0.05 * base                        # 20 s offsets make 20 s / 580 s gaps: more than the 3% tolerance used above


def test_a_frequency_publication_replaces_the_scheduled_trips_it_covers(tmp_path):
    base = _headway(_load(tmp_path / "base"))
    f = _load(tmp_path / "fq", copies=("freq",))
    ids = f.trips.lf.collect()["trip_id"].to_list()
    assert ids == ["freqT"] or sorted(i for i in ids if i.startswith("b")) == []     # the 24 scheduled trips are inside the window: deleted
    assert f.trip_dedupe_report["scheduled_in_frequency"] == 24
    assert _headway(f) == pytest.approx(base, rel=0.05)


def test_two_overlapping_frequency_windows_become_one(tmp_path):
    ref = _load(tmp_path / "ref", base=False, copies=("freq",))
    f = _load(tmp_path / "split", base=False, copies=("freq_split",))
    assert f.trip_dedupe_report["frequencies_joined"] == 1
    fr = f.stop_times.frequencies.collect()
    assert fr.height == 1 and fr["start_time"][0] == 6 * 3600 and fr["end_time"][0] == 10 * 3600
    assert _headway(f) == pytest.approx(_headway(ref), rel=0.03)


def test_a_service_on_other_days_is_untouched(tmp_path):
    # Saturday-only copy of the same stops and times: the selected Wednesday never sees it
    d = _feed(tmp_path / "x")
    import csv
    with open(d / "calendar.txt", "a", newline="") as fh:
        csv.writer(fh).writerow(["SAT", 0, 0, 0, 0, 0, 1, 0, "20240101", "20241231"])
    f = Feed(d, start_date=DAY, end_date=DAY, stop_group_distance=150)
    assert f.trips.lf.select(pl.len()).collect().item() == 24
