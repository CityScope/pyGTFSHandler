"""Specification tests for correct multi-day / overnight time handling.

These encode the *desired* post-refactor behavior described in the project
refactor plan: every stop_time resolves to an unambiguous `day_offset`
(0, 1, 2, ...) regardless of whether the source feed encoded overnight trips
with explicit `>=24:00:00` clock values or by silently wrapping back past
midnight with no marker at all, and the *real calendar date* used for
weekday/weekend/holiday classification is `service_date + day_offset`, not
the nominal `service_date`.

As of the pre-refactor code (`next_day: bool` + `"_night"` service_id
suffix), none of these are expected to pass — `pyGTFSHandler.feed.Feed.lf`
has no `day_offset`/`real_date` concept, and the "_night" suffix approach
only supports a single day of offset. This file is intentionally written
against the target API so the refactor has a concrete, executable spec to
turn green. See `/home/miguel/.claude/plans/wild-baking-sparrow.md`.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _stops():
    return [
        {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
        {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "C", "stop_lat": 40.02, "stop_lon": -3.72},
    ]


def _saturday_calendar(service_id: str = "SVC"):
    # 2024-05-04 is a Saturday.
    return [
        {
            "service_id": service_id,
            "monday": 0,
            "tuesday": 0,
            "wednesday": 0,
            "thursday": 0,
            "friday": 0,
            "saturday": 1,
            "sunday": 0,
            "start_date": "20240401",
            "end_date": "20240601",
        }
    ]


def _routes():
    return [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}]


@pytest.fixture
def explicit_25h_feed(tmp_path) -> Feed:
    directory = write_gtfs(
        tmp_path / "explicit_25h",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _saturday_calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "24:05:00", "departure_time": "24:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "25:10:00", "departure_time": "25:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    return Feed(directory)


@pytest.fixture
def implicit_wraparound_feed(tmp_path) -> Feed:
    """Same schedule as `explicit_25h_feed` but with wraparound clock values
    (no stop time ever written as `>=24:00:00`), which must resolve to the
    same day_offset per stop."""
    directory = write_gtfs(
        tmp_path / "implicit_wrap",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _saturday_calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "00:05:00", "departure_time": "00:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "01:10:00", "departure_time": "01:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    return Feed(directory)


@pytest.fixture
def two_day_offset_feed(tmp_path) -> Feed:
    """A very long trip spanning two full midnights (`50:10:00`)."""
    directory = write_gtfs(
        tmp_path / "two_day",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _saturday_calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "23:50:00", "departure_time": "23:50:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "26:05:00", "departure_time": "26:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "T1", "arrival_time": "50:10:00", "departure_time": "50:10:00", "stop_id": "S3", "stop_sequence": 3},
            ],
        },
    )
    return Feed(directory)


def _row_for_stop(lf: pl.LazyFrame, stop_id: str) -> dict:
    df = lf.filter(pl.col("stop_id").str.starts_with(stop_id)).collect()
    assert df.height == 1, f"expected exactly one row for {stop_id}, got {df.height}"
    return df.row(0, named=True)


class TestExplicitOverflowEncoding:
    def test_day_offsets(self, explicit_25h_feed):
        lf = explicit_25h_feed.lf
        assert _row_for_stop(lf, "S1")["day_offset"] == 0
        assert _row_for_stop(lf, "S2")["day_offset"] == 1
        assert _row_for_stop(lf, "S3")["day_offset"] == 1

    def test_time_of_day_always_0_24(self, explicit_25h_feed):
        lf = explicit_25h_feed.lf
        for stop_id in ("S1", "S2", "S3"):
            tod = _row_for_stop(lf, stop_id)["time_of_day"]
            assert 0 <= tod < 86400

    def test_service_visible_on_next_calendar_day(self, explicit_25h_feed):
        # 2024-05-04 (Saturday) is the nominal service_date (the only weekday
        # the calendar defines); the S3 stop at 25:10:00 actually happens on
        # 2024-05-05 (Sunday), even though no service_id is nominally active
        # on a Sunday. `filter_by_date` must still surface it by checking
        # `date - day_offset` against the calendar, not the literal date.
        trips_next_day = explicit_25h_feed.filter_by_date(date(2024, 5, 5)).collect()
        assert trips_next_day.filter(pl.col("stop_id").str.starts_with("S3")).height > 0

    def test_weekend_classification_uses_real_date(self, explicit_25h_feed):
        # S3 nominally belongs to a Saturday service_date, but its real date
        # (service_date + day_offset) is Sunday. Filtering the whole range by
        # date_type="weekend" must include it even though its own service_id
        # is only flagged active on Saturdays (a weekday) in calendar.txt.
        weekend_rows = explicit_25h_feed.filter_by_date_range(
            date(2024, 5, 4), date(2024, 5, 5), date_type="weekend"
        ).collect()
        assert weekend_rows.filter(pl.col("stop_id").str.starts_with("S3")).height > 0


class TestImplicitWraparoundEncoding:
    def test_matches_explicit_encoding(self, implicit_wraparound_feed):
        lf = implicit_wraparound_feed.lf
        assert _row_for_stop(lf, "S1")["day_offset"] == 0
        assert _row_for_stop(lf, "S2")["day_offset"] == 1
        assert _row_for_stop(lf, "S3")["day_offset"] == 1

    def test_time_of_day_matches_explicit_encoding(self, implicit_wraparound_feed, explicit_25h_feed):
        implicit_row = _row_for_stop(implicit_wraparound_feed.lf, "S3")
        explicit_row = _row_for_stop(explicit_25h_feed.lf, "S3")
        assert implicit_row["time_of_day"] == explicit_row["time_of_day"]


class TestTwoDayOffset:
    def test_day_offset_can_exceed_one(self, two_day_offset_feed):
        lf = two_day_offset_feed.lf
        assert _row_for_stop(lf, "S1")["day_offset"] == 0
        assert _row_for_stop(lf, "S2")["day_offset"] == 1
        assert _row_for_stop(lf, "S3")["day_offset"] == 2

    def test_time_of_day_for_50h10m(self, two_day_offset_feed):
        # 50:10:00 == 180600s -> day_offset 2, time_of_day 180600 - 2*86400 = 7800s = 02:10:00
        row = _row_for_stop(two_day_offset_feed.lf, "S3")
        assert row["time_of_day"] == 2 * 3600 + 10 * 60
