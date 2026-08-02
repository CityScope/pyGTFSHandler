"""The hardest single-feed scenario: frequencies + missing intermediate stop
times (interpolation) + a multi-day trip using explicit `>=24h` clock values
+ a second trip on the same feed using implicit wraparound with no `>=24h`
marker + a frequency window itself crossing midnight — all at once, in one
feed, as requested during plan review ("more challenging tests... combines
frequencies multiple files interpolation multiple days and the 25 hour
format together with the 23:59-00:01 format for day changes").
"""

from __future__ import annotations

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


@pytest.fixture
def hard_case_feed(tmp_path) -> Feed:
    calendar = [
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
    routes = [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}]
    stops = [
        {"stop_id": f"S{i}", "stop_name": f"Stop {i}", "stop_lat": 40.0 + i * 0.01, "stop_lon": -3.7 - i * 0.01}
        for i in range(1, 6)
    ]

    trips = [
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T_EXPLICIT_25H"},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T_IMPLICIT_WRAP"},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "TPL_FREQ"},
    ]

    stop_times = [
        # Explicit >=24h multi-day trip, with a missing intermediate time
        # that must be linearly (then shape-distance) interpolated.
        {"trip_id": "T_EXPLICIT_25H", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "T_EXPLICIT_25H", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "T_EXPLICIT_25H", "arrival_time": "25:10:00", "departure_time": "25:10:00", "stop_id": "S3", "stop_sequence": 3},
        # Implicit wraparound trip (23:58 -> 00:10), no >=24h marker anywhere.
        {"trip_id": "T_IMPLICIT_WRAP", "arrival_time": "23:58:00", "departure_time": "23:58:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "T_IMPLICIT_WRAP", "arrival_time": "00:04:00", "departure_time": "00:04:00", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "T_IMPLICIT_WRAP", "arrival_time": "00:10:00", "departure_time": "00:10:00", "stop_id": "S3", "stop_sequence": 3},
        # Frequency template trip, itself crossing midnight, with a missing
        # intermediate stop time.
        {"trip_id": "TPL_FREQ", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "TPL_FREQ", "arrival_time": "", "departure_time": "", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "TPL_FREQ", "arrival_time": "00:10:00", "departure_time": "00:10:00", "stop_id": "S3", "stop_sequence": 3},
    ]

    frequencies = [
        {"trip_id": "TPL_FREQ", "start_time": "23:00:00", "end_time": "02:00:00", "headway_secs": 1800, "exact_times": 0},
    ]

    directory = write_gtfs(
        tmp_path / "hard_case",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": calendar,
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
            "frequencies.txt": frequencies,
        },
    )
    return Feed(directory)


def test_loads_without_error(hard_case_feed):
    assert hard_case_feed.lf.collect().height > 0


def test_no_null_times_remain(hard_case_feed):
    df = hard_case_feed.lf.collect()
    assert df["departure_time"].null_count() == 0
    assert df["arrival_time"].null_count() == 0


def test_explicit_25h_trip_day_offset(hard_case_feed):
    df = hard_case_feed.lf.filter(pl.col("trip_id").str.starts_with("T_EXPLICIT_25H")).collect()
    last_stop = df.filter(pl.col("stop_id").str.starts_with("S3")).row(0, named=True)
    assert last_stop["day_offset"] == 1


def test_implicit_wrap_trip_day_offset(hard_case_feed):
    df = hard_case_feed.lf.filter(pl.col("trip_id").str.starts_with("T_IMPLICIT_WRAP")).collect()
    last_stop = df.filter(pl.col("stop_id").str.starts_with("S3")).row(0, named=True)
    assert last_stop["day_offset"] == 1


def test_frequency_instances_span_midnight_with_correct_offsets(hard_case_feed):
    df = hard_case_feed.lf.filter(pl.col("gtfs_name").is_not_null()).collect()
    freq_rows = hard_case_feed.lf.collect()
    # At least one generated instance must depart before midnight (offset 0)
    # and at least one after (offset 1), since the window spans 23:00-02:00.
    offsets = (
        freq_rows.filter(pl.col("trip_id").str.starts_with("TPL_FREQ"))["day_offset"]
        .unique()
        .to_list()
    )
    assert 0 in offsets
    assert 1 in offsets


def test_interpolated_middle_stop_between_neighbors(hard_case_feed):
    # `departure_time` is a 0-24h time_of_day, not an absolute clock -- it is
    # NOT expected to keep increasing across a midnight crossing (S1 is late
    # on day 0, S2/S3 are early on day 1). Absolute ordering must instead be
    # checked via day_offset*86400 + departure_time.
    df = hard_case_feed.lf.filter(pl.col("trip_id").str.starts_with("T_EXPLICIT_25H")).sort("stop_sequence").collect()
    df = df.with_columns((pl.col("day_offset") * 86400 + pl.col("departure_time")).alias("absolute_time"))
    s1_time = df.filter(pl.col("stop_id").str.starts_with("S1"))["absolute_time"][0]
    s2_time = df.filter(pl.col("stop_id").str.starts_with("S2"))["absolute_time"][0]
    s3_time = df.filter(pl.col("stop_id").str.starts_with("S3"))["absolute_time"][0]
    assert s1_time < s2_time < s3_time
