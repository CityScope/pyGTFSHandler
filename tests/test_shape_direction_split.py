"""Tests for the geometric `shape_direction` clustering used by
`Feed.get_headway_at_stops(by="shape_direction")`.

Fixture mirrors `examples/direction_split_headway_example.ipynb` /
`examples/test_files/shape_test`: six stops in a rough "Y" shape, four frequency-based
trips on one route -- T1/T3 travel "outbound" (S1 towards S4/S6), T2/T4 are
their exact reverse. S2 and S3 are visited by all four trips, so they are
the stops where the direction split actually matters.

Important, verified-by-experiment caveat: `shape_direction_id` is a *local*
cluster label computed independently at each stop -- it carries no meaning
across different stops, so the same trip can legitimately get label `0` at
one stop and `1` at another (e.g. a stop it visits alone, with nothing to
split against, gets an arbitrary single label). The property the mechanism
actually guarantees, and the one these tests check, is *local*: at any given
stop, trips travelling the same physical way get the same label, and trips
travelling the opposite way get the other one.

Also verified by experiment: at a literal geometric junction, bearing-based
clustering can only separate two trips if their bearings actually differ.
The original stop layout placed `S4` exactly due north of `S3`, which is
also the exact mean bearing from `S3` towards `S1`/`S2` (they're symmetric
about that axis) -- making outbound-via-S4 and inbound-via-S2/S1 genuinely
indistinguishable by bearing at `S3`, regardless of clustering algorithm.
`S4` is nudged slightly off that axis below to remove this degeneracy, as
confirmed necessary and sufficient by direct experimentation.
"""

from __future__ import annotations

import datetime

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
            "end_date": "20261231",
        }
    ]


def _routes():
    return [{"route_id": "R1", "route_short_name": "L1", "route_long_name": "Shape Test Line", "route_type": 3}]


def _stops():
    return [
        {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 40.02, "stop_lon": -3.69},
        {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 40.01, "stop_lon": -3.71},
        {"stop_id": "S3", "stop_name": "Stop 3", "stop_lat": 40.00, "stop_lon": -3.70},
        # S4 is nudged off the exact north axis through S3 (rather than
        # (0, 1) i.e. lon -3.70). At the literal symmetric coordinates, the
        # bearing from S3 "towards S4" and the bearing "towards the mean of
        # S1/S2" are mathematically identical (both exactly due north),
        # since S1/S2 sit symmetrically about that axis -- making T1
        # (outbound to S4) and T2 (inbound via S2/S1) genuinely
        # indistinguishable by bearing at S3, regardless of clustering
        # algorithm. Breaking that exact tie lets the existing
        # shape_direction clustering correctly separate them.
        {"stop_id": "S4", "stop_name": "Stop 4", "stop_lat": 40.01, "stop_lon": -3.695},
        {"stop_id": "S5", "stop_name": "Stop 5", "stop_lat": 39.99, "stop_lon": -3.70},
        {"stop_id": "S6", "stop_name": "Stop 6", "stop_lat": 39.98, "stop_lon": -3.70},
    ]


def _trips():
    return [
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T1", "direction_id": 0},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T2", "direction_id": 1},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T3", "direction_id": 0},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T4", "direction_id": 1},
    ]


def _stop_times():
    return [
        # T1: S1 -> S2 -> S3 -> S4
        {"trip_id": "T1", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "T1", "arrival_time": "00:02:00", "departure_time": "00:02:00", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "T1", "arrival_time": "00:04:00", "departure_time": "00:04:00", "stop_id": "S3", "stop_sequence": 3},
        {"trip_id": "T1", "arrival_time": "00:06:00", "departure_time": "00:06:00", "stop_id": "S4", "stop_sequence": 4},
        # T2: S4 -> S3 -> S2 -> S1 (exact reverse of T1)
        {"trip_id": "T2", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S4", "stop_sequence": 1},
        {"trip_id": "T2", "arrival_time": "00:02:00", "departure_time": "00:02:00", "stop_id": "S3", "stop_sequence": 2},
        {"trip_id": "T2", "arrival_time": "00:04:00", "departure_time": "00:04:00", "stop_id": "S2", "stop_sequence": 3},
        {"trip_id": "T2", "arrival_time": "00:06:00", "departure_time": "00:06:00", "stop_id": "S1", "stop_sequence": 4},
        # T3: S1 -> S2 -> S3 -> S5 -> S6
        {"trip_id": "T3", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S1", "stop_sequence": 1},
        {"trip_id": "T3", "arrival_time": "00:02:00", "departure_time": "00:02:00", "stop_id": "S2", "stop_sequence": 2},
        {"trip_id": "T3", "arrival_time": "00:04:00", "departure_time": "00:04:00", "stop_id": "S3", "stop_sequence": 3},
        {"trip_id": "T3", "arrival_time": "00:06:00", "departure_time": "00:06:00", "stop_id": "S5", "stop_sequence": 4},
        {"trip_id": "T3", "arrival_time": "00:08:00", "departure_time": "00:08:00", "stop_id": "S6", "stop_sequence": 5},
        # T4: S6 -> S5 -> S3 -> S2 -> S1 (exact reverse of T3)
        {"trip_id": "T4", "arrival_time": "00:00:00", "departure_time": "00:00:00", "stop_id": "S6", "stop_sequence": 1},
        {"trip_id": "T4", "arrival_time": "00:02:00", "departure_time": "00:02:00", "stop_id": "S5", "stop_sequence": 2},
        {"trip_id": "T4", "arrival_time": "00:04:00", "departure_time": "00:04:00", "stop_id": "S3", "stop_sequence": 3},
        {"trip_id": "T4", "arrival_time": "00:06:00", "departure_time": "00:06:00", "stop_id": "S2", "stop_sequence": 4},
        {"trip_id": "T4", "arrival_time": "00:08:00", "departure_time": "00:08:00", "stop_id": "S1", "stop_sequence": 5},
    ]


def _frequencies():
    return [
        {"trip_id": "T1", "start_time": "07:00:00", "end_time": "09:00:00", "headway_secs": 600, "exact_times": 0},
        {"trip_id": "T2", "start_time": "07:00:00", "end_time": "09:00:00", "headway_secs": 600, "exact_times": 0},
        {"trip_id": "T3", "start_time": "07:00:00", "end_time": "09:00:00", "headway_secs": 900, "exact_times": 0},
        {"trip_id": "T4", "start_time": "07:00:00", "end_time": "09:00:00", "headway_secs": 900, "exact_times": 0},
    ]


@pytest.fixture
def shape_test_feed(tmp_path) -> Feed:
    """The "Y"-shaped synthetic feed from `examples/direction_split_headway_example.ipynb`."""
    directory = write_gtfs(
        tmp_path / "shape_test",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": _trips(),
            "stop_times.txt": _stop_times(),
            "frequencies.txt": _frequencies(),
        },
    )
    return Feed(directory)


def _trip_shape_direction_ids(feed: Feed) -> pl.DataFrame:
    """Maps each `(trip_id, stop_id)` visit to the `shape_direction_id`
    cluster the library assigned it, by joining `get_headway_at_stops`'s
    per-`(stop_id, shape_direction_id)` `shape_ids` groups back onto each
    trip's own (synthetic) `shape_id` at that stop."""
    service_date = datetime.date(2026, 8, 3)
    start_time = datetime.time(7, 0)
    end_time = datetime.time(9, 0)

    headway_by_shape = feed.get_headway_at_stops(
        service_date,
        start_time=start_time,
        end_time=end_time,
        by="shape_direction",
        at="stop_id",
        n_divisions=1,
        how="all",
    )

    stop_to_direction = (
        headway_by_shape.select(["stop_id", "shape_direction_id", "shape_ids"])
        .explode("shape_ids")
        .rename({"shape_ids": "shape_id"})
    )

    trip_stop_shapes = (
        feed.filter(date=service_date, start_time=start_time, end_time=end_time)
        .select(["trip_id", "stop_id", "shape_id"])
        .unique()
        .collect()
    )

    return trip_stop_shapes.join(stop_to_direction, on=["stop_id", "shape_id"], how="inner")


def _direction_id_by_stop(mapping: pl.DataFrame, trip_id: str) -> dict[str, int]:
    """`{stop_id: shape_direction_id}` for one trip's own visits."""
    rows = mapping.filter(pl.col("trip_id") == trip_id)
    return dict(zip(rows["stop_id"], rows["shape_direction_id"]))


def test_outbound_trips_share_direction_id_on_common_stops(shape_test_feed):
    """T1 (S1->S2->S3->S4) and T3 (S1->S2->S3->S5->S6) travel the same way;
    at every stop they share, they must land in the same shape_direction_id
    -- checked per stop, since the label itself is only meaningful locally
    (see module docstring)."""
    mapping = _trip_shape_direction_ids(shape_test_feed)
    t1_by_stop = _direction_id_by_stop(mapping, "T1_file_0")
    t3_by_stop = _direction_id_by_stop(mapping, "T3_file_0")

    common_stops = set(t1_by_stop) & set(t3_by_stop)
    assert common_stops == {"S1_file_0", "S2_file_0", "S3_file_0"}

    for stop_id in common_stops:
        assert t1_by_stop[stop_id] == t3_by_stop[stop_id], (
            f"T1 and T3 disagree at {stop_id}: {t1_by_stop[stop_id]} != {t3_by_stop[stop_id]}"
        )


def test_inbound_trips_share_direction_id_on_common_stops(shape_test_feed):
    """T2 (S4->S3->S2->S1) and T4 (S6->S5->S3->S2->S1) are the reverse
    services; at every stop they share, they must agree on one
    shape_direction_id. Note S1 (both trips' final stop) isn't a "common
    stop" here in practice -- a trip's own terminus has no further stop to
    compute a forward bearing towards, so it gets no shape_direction_id
    there at all."""
    mapping = _trip_shape_direction_ids(shape_test_feed)
    t2_by_stop = _direction_id_by_stop(mapping, "T2_file_0")
    t4_by_stop = _direction_id_by_stop(mapping, "T4_file_0")

    common_stops = set(t2_by_stop) & set(t4_by_stop)
    assert common_stops == {"S2_file_0", "S3_file_0"}

    for stop_id in common_stops:
        assert t2_by_stop[stop_id] == t4_by_stop[stop_id], (
            f"T2 and T4 disagree at {stop_id}: {t2_by_stop[stop_id]} != {t4_by_stop[stop_id]}"
        )


def test_opposing_trip_pairs_get_different_direction_ids_at_shared_stops(shape_test_feed):
    """At every stop visited by both an outbound trip (T1/T3) and an inbound
    trip (T2/T4), the two must land in different shape_direction_id bins --
    that's the whole point of the split. Checked stop-by-stop, since
    shape_direction_id numbering isn't consistent across different stops."""
    mapping = _trip_shape_direction_ids(shape_test_feed)
    by_stop_trip = {
        (row["stop_id"], row["trip_id"]): row["shape_direction_id"]
        for row in mapping.iter_rows(named=True)
    }

    outbound_trips = {"T1_file_0", "T3_file_0"}
    inbound_trips = {"T2_file_0", "T4_file_0"}
    checked_any = False
    for stop_id in mapping["stop_id"].unique():
        outbound_dirs = {by_stop_trip[(stop_id, t)] for t in outbound_trips if (stop_id, t) in by_stop_trip}
        inbound_dirs = {by_stop_trip[(stop_id, t)] for t in inbound_trips if (stop_id, t) in by_stop_trip}
        if not outbound_dirs or not inbound_dirs:
            continue
        checked_any = True
        assert outbound_dirs.isdisjoint(inbound_dirs), (
            f"Outbound and inbound trips share a shape_direction_id at {stop_id}: "
            f"outbound={outbound_dirs} inbound={inbound_dirs}"
        )
    assert checked_any, "No stop had both an outbound and an inbound trip to compare"
