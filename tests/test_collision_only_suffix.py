"""Regression test for genuine collision-only `_file_<n>` id suffixing.

`io.read_csv_list` used to suffix *every* id in *every* table whenever more
than one GTFS directory was loaded together, even for ids that never
collide across those directories. `Feed.load` now computes a collision
registry up front (`io.compute_collision_registry`) and threads it through
every component's load, so only literal id values that genuinely appear
under more than one `file_id` get suffixed -- everything else comes back
exactly as authored.

This test builds two synthetic feed directories:
- Directory A defines stops S1, S2, S3.
- Directory B defines stops S4, S5, S6.
- Both directories additionally define a stop_id "SHARED" referring to two
  physically different stops (different lat/lon).

It asserts:
- S1..S6 come back with no `_file_` suffix.
- "SHARED" comes back suffixed, with both directories' variants present and
  distinguishable by coordinates.
- A stop_times.txt row referencing a non-colliding stop still joins
  correctly to that stop's unsuffixed id through a real `Feed` load.
"""

from __future__ import annotations

import polars as pl

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs


def _calendar(service_id):
    return [
        {
            "service_id": service_id,
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


def _build_dir_a(tmp_path):
    stops = [
        {"stop_id": "S1", "stop_name": "S1", "stop_lat": 40.00, "stop_lon": -3.70},
        {"stop_id": "S2", "stop_name": "S2", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "S3", "stop_lat": 40.02, "stop_lon": -3.72},
        {"stop_id": "SHARED", "stop_name": "Shared A", "stop_lat": 40.50, "stop_lon": -3.90},
    ]
    return write_gtfs(
        tmp_path / "feed_a",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("SVC_A"),
            "routes.txt": [{"route_id": "RA", "route_short_name": "RA", "route_long_name": "Route A", "route_type": 3}],
            "stops.txt": stops,
            "trips.txt": [{"route_id": "RA", "service_id": "SVC_A", "trip_id": "TA"}],
            "stop_times.txt": [
                {"trip_id": "TA", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "S1", "stop_sequence": 1},
                {"trip_id": "TA", "arrival_time": "08:05:00", "departure_time": "08:05:00", "stop_id": "S2", "stop_sequence": 2},
                {"trip_id": "TA", "arrival_time": "08:10:00", "departure_time": "08:10:00", "stop_id": "S3", "stop_sequence": 3},
                {"trip_id": "TA", "arrival_time": "08:15:00", "departure_time": "08:15:00", "stop_id": "SHARED", "stop_sequence": 4},
            ],
        },
    )


def _build_dir_b(tmp_path):
    stops = [
        {"stop_id": "S4", "stop_name": "S4", "stop_lat": 41.00, "stop_lon": -4.70},
        {"stop_id": "S5", "stop_name": "S5", "stop_lat": 41.01, "stop_lon": -4.71},
        {"stop_id": "S6", "stop_name": "S6", "stop_lat": 41.02, "stop_lon": -4.72},
        {"stop_id": "SHARED", "stop_name": "Shared B", "stop_lat": 10.00, "stop_lon": 100.00},
    ]
    return write_gtfs(
        tmp_path / "feed_b",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("SVC_B"),
            "routes.txt": [{"route_id": "RB", "route_short_name": "RB", "route_long_name": "Route B", "route_type": 3}],
            "stops.txt": stops,
            "trips.txt": [{"route_id": "RB", "service_id": "SVC_B", "trip_id": "TB"}],
            "stop_times.txt": [
                {"trip_id": "TB", "arrival_time": "09:00:00", "departure_time": "09:00:00", "stop_id": "S4", "stop_sequence": 1},
                {"trip_id": "TB", "arrival_time": "09:05:00", "departure_time": "09:05:00", "stop_id": "S5", "stop_sequence": 2},
                {"trip_id": "TB", "arrival_time": "09:10:00", "departure_time": "09:10:00", "stop_id": "S6", "stop_sequence": 3},
                {"trip_id": "TB", "arrival_time": "09:15:00", "departure_time": "09:15:00", "stop_id": "SHARED", "stop_sequence": 4},
            ],
        },
    )


def test_collision_only_suffixing(tmp_path):
    dir_a = _build_dir_a(tmp_path)
    dir_b = _build_dir_b(tmp_path)

    feed = Feed([dir_a, dir_b])

    stop_ids = set(feed.stops.lf.select("stop_id").collect()["stop_id"].to_list())

    # Non-colliding ids must come back completely unsuffixed.
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        assert sid in stop_ids, f"{sid} missing or wrongly suffixed: {stop_ids}"
        assert not any(s.startswith(f"{sid}_file_") for s in stop_ids), (
            f"non-colliding id {sid} was suffixed: {stop_ids}"
        )

    # The colliding id must be suffixed, and both directories' variants must
    # be present and distinguishable.
    shared_variants = {s for s in stop_ids if s.startswith("SHARED")}
    assert "SHARED" not in stop_ids, "colliding id 'SHARED' was left unsuffixed"
    assert len(shared_variants) == 2, f"expected 2 distinct SHARED variants, got {shared_variants}"

    shared_rows = (
        feed.stops.lf.filter(pl.col("stop_id").is_in(list(shared_variants)))
        .select("stop_id", "stop_lat", "stop_lon")
        .collect()
    )
    lats = set(shared_rows["stop_lat"].to_list())
    assert lats == {40.50, 10.00}, f"SHARED variants lost their distinct coordinates: {shared_rows}"

    # A stop_times.txt row referencing a non-colliding stop must still join
    # correctly to that stop's unsuffixed id through the real Feed load.
    lf_stop_ids = set(feed.lf.select("stop_id").unique().collect()["stop_id"].to_list())
    assert "S1" in lf_stop_ids
    assert "S2" in lf_stop_ids
    assert "S4" in lf_stop_ids
    assert "S5" in lf_stop_ids
    # None of the non-colliding stop_times rows should have gone missing
    # (i.e. no dangling suffix mismatch broke the stop_times <-> stops join).
    trip_a_stops = set(
        feed.lf.filter(pl.col("trip_id") == "TA").select("stop_id").collect()["stop_id"].to_list()
    )
    assert trip_a_stops - shared_variants == {"S1", "S2", "S3"}
    trip_b_stops = set(
        feed.lf.filter(pl.col("trip_id") == "TB").select("stop_id").collect()["stop_id"].to_list()
    )
    assert trip_b_stops - shared_variants == {"S4", "S5", "S6"}


def test_collision_registry_normalizes_whitespace_like_the_real_loader(tmp_path):
    """Regression for a real bug found running against real MBTA + regional
    GTFS data (2026-08-12): one directory's `stops.txt` had a stop_id with
    leading whitespace (`"  30235"`), which `read_csv_lazy` trims away via
    `gtfs_checker.normalize_df` (the default `check_files=True` path) --
    ending up identical to another directory's clean `"30235"`, a genuine
    collision. But `compute_collision_registry` used to scan the *raw*
    CSVs without that same normalization, so it compared `"  30235"` !=
    `"30235"` and never flagged the collision. Both physically distinct
    stops then silently shared one final `stop_id`, which fanned out into
    a memory-exploding join downstream (`_generate_shapes_file`, observed
    OOM-killing a 20GB+ process on real data). This directly recreates that
    whitespace scenario at the unit level.
    """
    stops_a = [
        {"stop_id": "30235", "stop_name": "Clean id stop", "stop_lat": 40.00, "stop_lon": -3.70},
    ]
    dir_a = write_gtfs(
        tmp_path / "feed_ws_a",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("SVC_A"),
            "routes.txt": [{"route_id": "RA", "route_short_name": "RA", "route_long_name": "Route A", "route_type": 3}],
            "stops.txt": stops_a,
            "trips.txt": [{"route_id": "RA", "service_id": "SVC_A", "trip_id": "TA"}],
            "stop_times.txt": [
                {"trip_id": "TA", "arrival_time": "08:00:00", "departure_time": "08:00:00", "stop_id": "30235", "stop_sequence": 1},
            ],
        },
    )

    stops_b = [
        {"stop_id": "  30235", "stop_name": "Whitespace id stop", "stop_lat": 41.86, "stop_lon": -71.35},
    ]
    dir_b = write_gtfs(
        tmp_path / "feed_ws_b",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar("SVC_B"),
            "routes.txt": [{"route_id": "RB", "route_short_name": "RB", "route_long_name": "Route B", "route_type": 3}],
            "stops.txt": stops_b,
            "trips.txt": [{"route_id": "RB", "service_id": "SVC_B", "trip_id": "TB"}],
            "stop_times.txt": [
                {"trip_id": "TB", "arrival_time": "09:00:00", "departure_time": "09:00:00", "stop_id": "  30235", "stop_sequence": 1},
            ],
        },
    )

    feed = Feed([dir_a, dir_b], check_files=True)
    stops_df = feed.stops.lf.select("stop_id", "stop_name").collect()

    # The whitespace variant must be trimmed to the same "30235" as the
    # clean one -- and, being a genuine post-trim collision, both must come
    # back suffixed and distinguishable, never silently merged into one
    # duplicated "30235" row.
    assert stops_df.height == 2, f"expected 2 distinct stops, got: {stops_df}"
    assert stops_df["stop_id"].n_unique() == 2, f"stop_id not unique after normalization: {stops_df}"
    assert set(stops_df["stop_id"].to_list()) == {"30235_file_0", "30235_file_1"}, stops_df
