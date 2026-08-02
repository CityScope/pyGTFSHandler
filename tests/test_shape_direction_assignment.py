"""Tests for `Shapes.assign_direction_ids` -- the route-wide, branch-aware
reconciliation of per-stop `direction_id` (0/1) labels described in
`pyGTFSHandler/models/shapes.py`.

Two levels are covered:
- Unit tests directly against `_widest_gap_split` and
  `Shapes._assign_direction_ids_for_route`, with hand-built bearing data
  (bypassing real lat/lon geometry) so the reconciliation algorithm itself
  -- anchor selection, ascending-support processing order, orientation
  choice, majority-rule conflict resolution -- is exercised precisely,
  including a deliberately adversarial case that forces a real conflict.
- An integration test building the full `shape_test` synthetic feed (same
  fixture as `test_shape_direction_split.py`) and checking that
  `feed.shapes.stop_shapes` ends up carrying `route_id`/`direction_id`/
  `direction_id_issues` directly, end to end, plus that no warning fires
  when there's nothing to flag.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from pyGTFSHandler.feed import Feed
from pyGTFSHandler.models.shapes import Shapes, _widest_gap_split

from .gtfs_builder import minimal_agency, write_gtfs
from .test_shape_direction_split import _calendar, _frequencies, _routes, _stop_times, _stops, _trips


def _route_df(rows: list[dict]) -> pl.DataFrame:
    """Builds a minimal `route_df` as `_assign_direction_ids_for_route`
    expects it: one row per (stop_id, shape_id) with a pre-computed
    `_adjusted_bearing` (skipping the real bearing-blend formula, so the
    reconciliation logic can be tested against exact, chosen-by-hand
    angles)."""
    return pl.DataFrame(
        rows,
        schema={
            "shape_id": pl.Utf8,
            "stop_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "shape_direction": pl.Float64,
            "shape_direction_backwards": pl.Float64,
            "route_id": pl.Utf8,
            "_adjusted_bearing": pl.Float64,
        },
    )


def test_widest_gap_split_two_clear_clusters():
    bins = _widest_gap_split({"A": 10.0, "B": 15.0, "C": 190.0, "D": 200.0})
    assert bins["A"] == bins["B"]
    assert bins["C"] == bins["D"]
    assert bins["A"] != bins["C"]


def test_widest_gap_split_single_shape_is_trivial():
    assert _widest_gap_split({"A": 42.0}) == {"A": 0}


def test_assign_direction_ids_for_route_consistent_across_stops():
    """Hub stop H has all four shapes, cleanly split A/B vs C/D. Two smaller
    branch stops each see only *one* shape (a single-shape stop has no
    other bearing to be split against, so it trivially adopts whatever its
    lone shape's already-established value is). No conflict should ever be
    flagged, and each shape keeps one direction_id across every stop it
    appears at.

    (Deliberately not testing a smaller stop with exactly 2 *close*
    bearings here: forcing exactly 2 output bins from exactly 2 points
    always splits them, however close, since 2 points define exactly 2
    gaps whose midpoints are unavoidably 180 degrees apart -- a real
    property of this widest-gap approach, not something specific to this
    test, and only actually a problem when a stop has close bearings with
    no genuine opposite-direction traffic to split against, which none of
    this route's real stops do.)"""
    route_df = _route_df(
        [
            {"shape_id": "A", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 10.0},
            {"shape_id": "B", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 20.0},
            {"shape_id": "C", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 190.0},
            {"shape_id": "D", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 200.0},
            {"shape_id": "A", "stop_id": "X", "stop_sequence": 2, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 15.0},
            {"shape_id": "C", "stop_id": "Y", "stop_sequence": 2, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 195.0},
        ]
    )

    result = Shapes()._assign_direction_ids_for_route(route_df)

    assert not result["direction_id_issues"].any()

    by_shape = {
        shape_id[0]: set(sub["direction_id"].to_list())
        for shape_id, sub in result.group_by("shape_id")
    }
    for shape_id, values in by_shape.items():
        assert len(values) == 1, f"{shape_id} got inconsistent direction_ids: {values}"

    assert by_shape["A"] == by_shape["B"]
    assert by_shape["C"] == by_shape["D"]
    assert by_shape["A"] != by_shape["C"]


def test_assign_direction_ids_for_route_flags_genuine_conflict():
    """Hub stop H has 4 shapes (A, B, C, D -- strictly more than X's 3, so H
    is unambiguously the anchor) establishing A, B as direction 0 and C, D
    as direction 1. A smaller stop X sees only A, B, C -- but with A and C
    bearing *together* there and B on the opposite side, incompatible with
    H's grouping no matter how X's own two local clusters are oriented:
    flipping X's {A, C} vs {B} pairing to agree with B and C's own
    established H values (2 agreements) beats keeping it as-is (only 1,
    A's). So B and C keep their own values at X, and A alone is forced to
    the opposite of its own established value there, and flagged."""
    route_df = _route_df(
        [
            # Hub: A & B together (direction 0), C & D together (direction 1).
            {"shape_id": "A", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 0.0},
            {"shape_id": "B", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 10.0},
            {"shape_id": "C", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 180.0},
            {"shape_id": "D", "stop_id": "H", "stop_sequence": 1, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 185.0},
            # X: A and C bearings close together, B on the opposite side --
            # contradicts H's A/B-together, C-alone grouping.
            {"shape_id": "A", "stop_id": "X", "stop_sequence": 2, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 90.0},
            {"shape_id": "B", "stop_id": "X", "stop_sequence": 2, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 270.0},
            {"shape_id": "C", "stop_id": "X", "stop_sequence": 2, "shape_direction": None, "shape_direction_backwards": None, "route_id": "R", "_adjusted_bearing": 100.0},
        ]
    )

    result = Shapes()._assign_direction_ids_for_route(route_df)

    h_rows = result.filter(pl.col("stop_id") == "H").sort("shape_id")
    assert not h_rows["direction_id_issues"].any()
    h_direction = dict(zip(h_rows["shape_id"], h_rows["direction_id"]))
    assert h_direction["A"] == h_direction["B"]
    assert h_direction["C"] == h_direction["D"]
    assert h_direction["C"] != h_direction["A"]

    x_rows = result.filter(pl.col("stop_id") == "X").sort("shape_id")
    x_direction = dict(zip(x_rows["shape_id"], x_rows["direction_id"]))
    x_issues = dict(zip(x_rows["shape_id"], x_rows["direction_id_issues"]))
    # X's local clustering groups {A, C} vs {B}; flipping that pairing to
    # agree with B and C's own established values (2 agreements) beats
    # keeping it as-is (1 agreement, only A would match) -- so B and C keep
    # their own canonical values here, and A alone is forced to the
    # opposite of its own established value at H, and flagged.
    assert x_direction["B"] == h_direction["B"]
    assert x_direction["C"] == h_direction["C"]
    assert x_direction["A"] != h_direction["A"], "A's row at X should be forced away from its own value"

    assert x_issues == {"A": True, "B": False, "C": False}


def test_warn_about_direction_id_issues_reports_counts():
    directions = pl.DataFrame(
        {
            "shape_id": ["A", "A", "B", "C"],
            "stop_id": ["H", "X", "H", "H"],
            "direction_id": [0, 1, 0, 1],
            "direction_id_issues": [False, True, False, False],
        },
        schema={
            "shape_id": pl.Utf8,
            "stop_id": pl.Utf8,
            "direction_id": pl.Int32,
            "direction_id_issues": pl.Boolean,
        },
    )

    with pytest.warns(RuntimeWarning, match=r"1 of 3 shape_ids \(33\.3%\).*1 of 2 stop\(s\) \(50\.0%\)"):
        Shapes()._warn_about_direction_id_issues(directions)


def test_warn_about_direction_id_issues_silent_when_no_issues():
    directions = pl.DataFrame(
        {
            "shape_id": ["A", "B"],
            "stop_id": ["H", "H"],
            "direction_id": [0, 1],
            "direction_id_issues": [False, False],
        },
        schema={
            "shape_id": pl.Utf8,
            "stop_id": pl.Utf8,
            "direction_id": pl.Int32,
            "direction_id_issues": pl.Boolean,
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Shapes()._warn_about_direction_id_issues(directions)


@pytest.fixture
def shape_test_feed(tmp_path) -> Feed:
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


def test_direction_ids_integrated_into_stop_shapes_without_warning(tmp_path):
    # Built here (rather than via a fixture) so the warning check below
    # covers the exact call that runs `assign_direction_ids` -- that call
    # happens during `Feed(...)` construction itself, not on later attribute
    # access.
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feed = Feed(directory)
        direction_id_warnings = [w for w in caught if "direction_id assignment" in str(w.message)]
    assert direction_id_warnings == []

    stop_shapes = feed.shapes.stop_shapes.collect()
    assert stop_shapes.height > 0
    assert set(stop_shapes.columns) >= {
        "route_id", "shape_id", "stop_id", "stop_sequence",
        "shape_direction", "shape_direction_backwards",
        "direction_id", "direction_id_issues",
    }
    assert stop_shapes["direction_id_issues"].dtype == pl.Boolean
    assert not stop_shapes["direction_id_issues"].any()

    by_shape = {
        shape_id[0]: set(sub["direction_id"].to_list())
        for shape_id, sub in stop_shapes.group_by("shape_id")
    }
    for shape_id, values in by_shape.items():
        assert len(values) == 1, f"{shape_id} got inconsistent direction_ids: {values}"

    outbound = by_shape["T1_file_0"]
    assert outbound == by_shape["T3_file_0"]
    inbound = by_shape["T2_file_0"]
    assert inbound == by_shape["T4_file_0"]
    assert outbound != inbound


def test_stop_shapes_bearings_untouched_by_direction_assignment(shape_test_feed):
    """`assign_direction_ids` must not rewrite the raw bearings it reads --
    only add new columns alongside them. T2's own last stop (S1) has no
    *forward* bearing at all (there's no next stop) -- that raw NaN must
    still be there afterwards, not silently patched up."""
    stop_shapes = shape_test_feed.shapes.stop_shapes.collect()
    assert stop_shapes["shape_direction"].dtype == pl.Float64
    assert stop_shapes["shape_direction_backwards"].dtype == pl.Float64

    t2_at_s1 = stop_shapes.filter(pl.col("shape_id") == "T2_file_0", pl.col("stop_id") == "S1_file_0")
    assert t2_at_s1.height == 1
    forward = t2_at_s1["shape_direction"][0]
    assert forward is None or forward != forward  # None or NaN
    assert t2_at_s1["direction_id"][0] is not None  # still resolved despite the missing forward bearing
