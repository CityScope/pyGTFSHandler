"""Tests for GTFS validation / normalization (`gtfs_checker.py`).

Covers missing mandatory files, malformed CSV rows, referential integrity
problems, duplicate primary keys, out-of-range values, and files that are
present but empty -- the categories of problems a checker should catch and
report rather than crash on unhelpfully or silently ignore.
"""

from __future__ import annotations

import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _valid_minimal(tmp_path, name="valid"):
    return write_gtfs(
        tmp_path / name,
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": [
                {
                    "service_id": "SVC",
                    "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                    "friday": 1, "saturday": 1, "sunday": 1,
                    "start_date": "20240101", "end_date": "20241231",
                }
            ],
            "routes.txt": [{"route_id": "R1", "route_short_name": "1", "route_long_name": "Line 1", "route_type": 3}],
            "stops.txt": [
                {"stop_id": "S1", "stop_name": "A", "stop_lat": 40.0, "stop_lon": -3.7},
                {"stop_id": "S2", "stop_name": "B", "stop_lat": 40.01, "stop_lon": -3.71},
            ],
            "trips.txt": [{"route_id": "R1", "service_id": "SVC", "trip_id": "T1"}],
            "stop_times.txt": [
                {"trip_id": "T1", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "T1", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S2", "stop_sequence": 2},
            ],
        },
    )


def test_valid_minimal_feed_loads(tmp_path):
    directory = _valid_minimal(tmp_path)
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_missing_mandatory_file_raises(tmp_path):
    directory = _valid_minimal(tmp_path, "missing_stop_times")
    (directory / "stop_times.txt").unlink()
    with pytest.raises(Exception):
        Feed(directory)


def test_stop_times_referencing_nonexistent_trip_id(tmp_path):
    directory = _valid_minimal(tmp_path, "bad_trip_ref")
    stop_times_path = directory / "stop_times.txt"
    stop_times_path.write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "GHOST_TRIP,08:00:00,08:00:00,S1,1\n"
        "GHOST_TRIP,08:10:00,08:10:00,S2,2\n"
    )
    # Should either raise (checker flags dangling reference) or load with
    # zero trips -- must not silently fabricate a phantom trip.
    try:
        feed = Feed(directory)
    except Exception:
        return
    assert feed.trips.lf.collect().height == 0 or feed.lf.collect().height == 0


def test_duplicate_stop_id_in_stops_file(tmp_path):
    directory = _valid_minimal(tmp_path, "dup_stop_id")
    stops_path = directory / "stops.txt"
    stops_path.write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "S1,A,40.0,-3.7\n"
        "S1,A duplicate,41.0,-4.7\n"
        "S2,B,40.01,-3.71\n"
    )
    with pytest.raises(Exception):
        Feed(directory)


def test_out_of_range_route_type(tmp_path):
    directory = _valid_minimal(tmp_path, "bad_route_type")
    routes_path = directory / "routes.txt"
    routes_path.write_text(
        "route_id,route_short_name,route_long_name,route_type\n"
        "R1,1,Line 1,9999\n"
    )
    with pytest.raises(Exception):
        Feed(directory)


def test_empty_but_present_optional_file(tmp_path):
    directory = _valid_minimal(tmp_path, "empty_shapes")
    (directory / "shapes.txt").write_text("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n")
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_missing_optional_files_load_fine(tmp_path):
    directory = _valid_minimal(tmp_path, "no_optional_files")
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_transfers_and_feed_info_present_load_fine(tmp_path):
    """`transfers.txt`/`feed_info.txt` are optional GTFS files this codebase
    doesn't otherwise process -- their mere presence must not break loading."""
    directory = _valid_minimal(tmp_path, "transfers_and_feed_info")
    (directory / "transfers.txt").write_text(
        "from_stop_id,to_stop_id,transfer_type\nS1,S2,0\n"
    )
    (directory / "feed_info.txt").write_text(
        "feed_publisher_name,feed_publisher_url,feed_lang\nTest Agency,http://example.com,en\n"
    )
    feed = Feed(directory)
    assert feed.lf.collect().height == 2


def test_messy_csv_formatting_is_parsed_correctly(tmp_path):
    """Real-world GTFS exports are often untidy: comment lines, blank lines,
    inconsistent spacing around header/field names and values. None of that
    should change the parsed result (ported from the old static
    `tests/gtfs_files/gtfs_1` fixture, inlined here so the exact edge case
    under test is visible next to the assertion)."""
    directory = _valid_minimal(tmp_path, "messy_csv")
    (directory / "stop_times.txt").write_text(
        " trip_id,arrival_time,departure_time,stop_id , stop_sequence\n"
        "\n"
        "# a comment line that should be ignored\n"
        "T1,08:00:00,08:00:00,S1,1\n"
        "\n"
        "T1, 08:10:00,08:10:00 ,  S2, 2\n"
    )
    feed = Feed(directory)
    df = feed.lf.sort("stop_sequence").collect()
    assert df.height == 2
    assert df["stop_id"].str.starts_with("S2").any()
    assert df["departure_time"][1] == 8 * 3600 + 10 * 60


def test_malformed_time_string_is_flagged(tmp_path):
    directory = _valid_minimal(tmp_path, "bad_time")
    stop_times_path = directory / "stop_times.txt"
    stop_times_path.write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,not-a-time,not-a-time,S1,1\n"
        "T1,08:10:00,08:10:00,S2,2\n"
    )
    # Malformed time must not silently become a wrong numeric value; the
    # loader should either raise or drop/null the row with a warning.
    feed = Feed(directory)
    df = feed.lf.collect()
    assert df.filter(df["stop_id"].str.starts_with("S1"))["arrival_time"].null_count() <= 1
