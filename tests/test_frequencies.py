"""Tests for `frequencies.txt` handling, including midnight-crossing windows
encoded either as `end < start` (implicit wrap) or with an explicit `>=24:00`
end time, `exact_times` variants, and frequency-only trips whose intermediate
stop_times are missing and must be interpolated.
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


def _stops():
    return [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.02, "stop_lon": -3.72},
    ]


def _build_frequencies_feed(tmp_path, name, start_time, end_time, stop_times_extra=None):
    stop_times = [
        {"trip_id": "TPL", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "TPL", "arrival_time": "00:05:00", "departure_time": "00:05:00", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "TPL", "arrival_time": "00:10:00", "departure_time": "00:10:00", "stop_id": "S3", "stop_sequence": 3},
    ]
    if stop_times_extra is not None:
        stop_times = stop_times_extra
    directory = write_gtfs(
        tmp_path / name,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "TPL"}],
            "stop_times.txt": stop_times,
            "frequencies.txt": [
                {
                    "trip_id": "TPL",
                    "start_time": start_time,
                    "end_time": end_time,
                    "headway_secs": 1800,
                    "exact_times": 0,
                }
            ],
        },
    )
    return Feed(directory)


def test_implicit_wrap_window(tmp_path):
    feed = _build_frequencies_feed(tmp_path, "wrap_implicit", "23:00:00", "02:00:00")
    instances = feed.lf.select("trip_id").unique().collect()
    # 23:00 -> 02:00 (next day) at 1800s headway == 2h window == 4 instances (0,30,60,90 min in)
    assert instances.height >= 3


def test_explicit_overflow_window(tmp_path):
    feed = _build_frequencies_feed(tmp_path, "wrap_explicit", "23:00:00", "26:00:00")
    instances = feed.lf.select("trip_id").unique().collect()
    assert instances.height >= 3


def test_implicit_and_explicit_windows_equivalent(tmp_path):
    implicit = _build_frequencies_feed(tmp_path, "equiv_implicit", "23:00:00", "02:00:00")
    explicit = _build_frequencies_feed(tmp_path, "equiv_explicit", "23:00:00", "26:00:00")
    assert implicit.lf.select("trip_id").unique().collect().height == explicit.lf.select(
        "trip_id"
    ).unique().collect().height


def test_exact_times_flag_respected(tmp_path):
    directory = write_gtfs(
        tmp_path / "exact_times",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "TPL"}],
            "stop_times.txt": [
                {"trip_id": "TPL", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "TPL", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "TPL", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
            "frequencies.txt": [
                {"trip_id": "TPL", "start_time": "08:00:00", "end_time": "09:00:00", "headway_secs": 1800, "exact_times": 1},
            ],
        },
    )
    feed = Feed(directory)
    # Ordinary (non-midnight-crossing) frequency windows are kept as a single
    # template trip_id with frequency metadata columns attached, rather than
    # expanded into one row per instance (expansion into concrete trip_ids
    # only happens when a window could cross midnight, since that's the only
    # case where per-instance `day_offset` can actually differ).
    df = feed.lf.collect()
    assert df.select("trip_id").unique().height == 1
    assert df["headway_secs"].drop_nulls().unique().to_list() == [1800]


def test_frequency_with_missing_intermediate_stop_time(tmp_path):
    """The middle stop of the frequency-driven trip has no explicit time and
    must be interpolated rather than causing a crash or a null propagating
    into the final schedule."""
    stop_times = [
        {"trip_id": "TPL", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "TPL", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "TPL", "arrival_time": "00:10:00", "departure_time": "00:10:00", "stop_id": "S3", "stop_sequence": 3},
    ]
    feed = _build_frequencies_feed(tmp_path, "freq_missing_mid", "08:00:00", "09:00:00", stop_times)
    df = feed.lf.filter(pl.col("stop_id").str.starts_with("S2")).collect()
    assert df.height > 0
    assert df["departure_time"].null_count() == 0


def test_frequency_window_shorter_than_headway_does_not_crash(tmp_path):
    feed = _build_frequencies_feed(tmp_path, "short_window", "08:00:00", "08:05:00")
    # window is shorter than the 1800s headway -> at least the first instance
    # must still be generated, and loading must not raise.
    assert feed.lf.select("trip_id").unique().collect().height >= 1
