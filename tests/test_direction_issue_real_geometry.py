"""Tests for `Shapes.assign_direction_ids` against a *real* (lat/lon-based,
not hand-crafted bearings) GTFS feed engineered to force a genuine direction
conflict, mirroring `examples/test_files/direction_issue`.

Background: with only 2 shapes on a route, `direction_conflict` can never
be `True` (see `Shapes._widest_gap_split` docstring/discussion) -- a stop
with exactly 2 shapes always splits them into different local bins, so any
later stop with the same 2 shapes can always be oriented (identity or flip)
to agree with both. A genuine, unresolvable conflict needs at least 3
shapes at some stop. This fixture supplies that by adding the *inverse* of
each of the two original trips:

- T1: S1->S2->S3->S4->S5->S6->S7 (straight line east)
- T2: S1->S2->S8->S9->S7->S6->S5 (shares S1/S2 with T1, detours north via
  S8/S9, then re-enters the main line at S7 and back-tracks through S6 to
  S5 -- i.e. T2 traverses the S5<->S6<->S7 stretch in the *opposite* order
  to T1)
- T3: exact reverse of T1 (S7->S6->...->S1)
- T4: exact reverse of T2 (S5->S6->S7->S9->S8->S2->S1)

T2/T4 must extend all the way to S5 (not stop short at S6): a shape's own
first/last stop is excluded from `direction_id` bearing pooling (there's no
"direction of travel" through a point where a vehicle starts or ends), so
if T2/T4 terminated at S6, S6 would only ever see T1/T3 there (trivial,
un-conflictable) and S2 would become the *only* 4-shape station on the
route -- which trivially becomes the anchor and can never conflict with
itself. Reaching S5 makes S6 a genuine 4-shape junction too, so the route
has two independent 4-shape stations (S2, where T1/T2 agree, and S6, the
anchor, where they don't) that cannot be reconciled into one consistent
labeling -- forcing a real conflict that's flagged rather than silently
resolved.
"""

from __future__ import annotations

import warnings

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
    return [{"route_id": "R1", "route_short_name": "L1", "route_long_name": "Direction Issue Line", "route_type": 3}]


def _stops():
    return [
        {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 1.0},
        {"stop_id": "S3", "stop_name": "Stop 3", "stop_lat": 0.0, "stop_lon": 2.0},
        {"stop_id": "S4", "stop_name": "Stop 4", "stop_lat": 0.0, "stop_lon": 3.0},
        {"stop_id": "S5", "stop_name": "Stop 5", "stop_lat": 0.0, "stop_lon": 4.0},
        {"stop_id": "S6", "stop_name": "Stop 6", "stop_lat": 0.0, "stop_lon": 5.0},
        {"stop_id": "S7", "stop_name": "Stop 7", "stop_lat": 0.0, "stop_lon": 6.0},
        {"stop_id": "S8", "stop_name": "Stop 8", "stop_lat": 1.0, "stop_lon": 1.0},
        {"stop_id": "S9", "stop_name": "Stop 9", "stop_lat": 1.0, "stop_lon": 6.0},
    ]


def _trips():
    return [
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T1"},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T2"},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T3"},
        {"route_id": "R1", "service_id": "SVC", "trip_id": "T4"},
    ]


def _stop_sequence(trip_id: str, stop_ids: list[str]) -> list[dict]:
    return [
        {
            "trip_id": trip_id,
            "arrival_time": f"00:{2 * i:02d}:00",
            "departure_time": f"00:{2 * i:02d}:00",
            "stop_id": stop_id,
            "stop_sequence": i + 1,
        }
        for i, stop_id in enumerate(stop_ids)
    ]


def _stop_times():
    return (
        _stop_sequence("T1", ["S1", "S2", "S3", "S4", "S5", "S6", "S7"])
        + _stop_sequence("T2", ["S1", "S2", "S8", "S9", "S7", "S6", "S5"])
        + _stop_sequence("T3", ["S7", "S6", "S5", "S4", "S3", "S2", "S1"])
        + _stop_sequence("T4", ["S5", "S6", "S7", "S9", "S8", "S2", "S1"])
    )


def _frequencies():
    return [
        {"trip_id": t, "start_time": "07:00:00", "end_time": "09:00:00", "headway_secs": 600, "exact_times": 0}
        for t in ["T1", "T2", "T3", "T4"]
    ]


@pytest.fixture
def direction_issue_feed(tmp_path) -> Feed:
    """Matches `examples/test_files/direction_issue` exactly."""
    directory = write_gtfs(
        tmp_path / "direction_issue",
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


def test_four_shapes_are_loaded(direction_issue_feed):
    stop_shapes = direction_issue_feed.shapes.stop_shapes.collect()
    assert stop_shapes["shape_id"].n_unique() == 4


def test_conflicting_shapes_are_flagged(direction_issue_feed):
    stop_shapes = direction_issue_feed.shapes.stop_shapes.collect()

    flagged = stop_shapes.filter(pl.col("direction_conflict"))
    assert flagged.height > 0, "Expected at least one flagged (shape_id, stop_id) row"

    flagged_shape_ids = set(s.split("_file_")[0] for s in flagged["shape_id"].unique().to_list())
    flagged_stop_ids = set(s.split("_file_")[0] for s in flagged["stop_id"].unique().to_list())

    # S2 is the only station where all four shapes are simultaneously
    # non-endpoint (S1 is an endpoint for all four; S6 is the anchor,
    # exempt from conflict-checking by construction -- see
    # `Shapes._assign_direction_ids_for_route`'s docstring, step 3). At S2,
    # T1/T2 genuinely agree (same physical direction), which conflicts with
    # T1's anchor-derived canonical -- T1/T3 end up flagged rather than
    # T2/T4 because of the anchor tie-break (alphabetically last among the
    # tied-for-most-corroborated stations) and the identity-vs-flip
    # tie-break at S2 (ties keep the existing orientation); the underlying
    # conflict is symmetric, but this implementation's tie-breaks
    # deterministically pin the flag on T1/T3 here.
    assert flagged_shape_ids == {"T1", "T3"}
    assert flagged_stop_ids == {"S2"}


def test_clean_shapes_are_not_flagged(direction_issue_feed):
    """T2 and T4 (the detouring trip and its exact reverse) never conflict
    in this fixture's tie-break outcome -- only T1/T3 (the straight-line
    pair) end up flagged. See `test_conflicting_shapes_are_flagged` for why
    it's this pair and not the other."""
    stop_shapes = direction_issue_feed.shapes.stop_shapes.collect()
    for shape_id, sub in stop_shapes.group_by("shape_id"):
        shape_id = shape_id[0]
        if shape_id.startswith("T2_") or shape_id.startswith("T4_"):
            assert not sub["direction_conflict"].any(), f"{shape_id} should never be flagged"


def test_feed_wide_warning_reports_two_of_four_shapes_and_one_of_nine_stops(tmp_path):
    directory = write_gtfs(
        tmp_path / "direction_issue",
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
    with pytest.warns(
        RuntimeWarning,
        match=r"2 of 4 shape_ids \(50\.0%\).*1 of 9 stop\(s\) \(11\.1%\)",
    ):
        Feed(directory)


def test_two_shapes_alone_would_never_conflict(tmp_path):
    """Sanity check for the underlying claim: dropping the two inverse
    trips (T3/T4) and keeping only T1/T2 must produce zero conflicts and no
    warning, however the same S8/S9 detour is still present."""
    directory = write_gtfs(
        tmp_path / "two_shapes_only",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": _routes(),
            "stops.txt": _stops(),
            "trips.txt": _trips()[:2],
            "stop_times.txt": _stop_sequence("T1", ["S1", "S2", "S3", "S4", "S5", "S6", "S7"])
            + _stop_sequence("T2", ["S1", "S2", "S8", "S9", "S7", "S6"]),
            "frequencies.txt": _frequencies()[:2],
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feed = Feed(directory)
        direction_id_warnings = [w for w in caught if "direction_id assignment" in str(w.message)]
    assert direction_id_warnings == []
    assert not feed.shapes.stop_shapes.collect()["direction_conflict"].any()
