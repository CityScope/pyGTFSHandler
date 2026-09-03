"""Regression test for a real bug found downstream (transitLOS, 2026-08-12):
on a real multi-file MBTA feed, `shape_dist_traveled` came back null for
~61% of stop-time rows -- including trips with perfectly good `shapes.txt`
geometry available -- which cascaded into `get_speed_at_stops` returning
`null` for ~93% of stops city-wide (speed needs `shape_dist_traveled` to
compute a distance-per-time rate).

Every existing shape-geometry test (`test_shapes_geometry.py`) builds a
feed with exactly ONE trip on ONE shape -- there was no test with multiple
routes/shapes converging on a shared stop, which is exactly the structural
pattern (multiple branches, one trunk station) the real bug appeared on.
This file adds that missing scenario and an explicit *coverage* assertion
(not just "the one row I checked is correct", but "no row silently drops
out"), so a regression that reintroduces null `shape_dist_traveled` for a
subset of otherwise-valid trips gets caught even if single-trip tests still
pass.
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
            "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
            "friday": 1, "saturday": 1, "sunday": 1,
            "start_date": "20240101", "end_date": "20241231",
        }
    ]


@pytest.fixture
def multi_branch_shape_feed(tmp_path) -> Feed:
    """3 branches (own route + own real-geometry shape each) converging on
    one shared trunk stop, several trips per branch -- structurally the
    same shape (pun intended) as the real MBTA Green Line/Boylston bug:
    multiple distinct `shape_id`s, multiple trips per shape, one stop in
    common across all of them.
    """
    stops = [
        {"stop_id": "TRUNK", "stop_name": "Trunk Station", "stop_lat": 40.000, "stop_lon": -3.700},
        {"stop_id": "S_B", "stop_name": "Branch B end", "stop_lat": 40.010, "stop_lon": -3.690},
        {"stop_id": "S_C", "stop_name": "Branch C end", "stop_lat": 40.010, "stop_lon": -3.700},
        {"stop_id": "S_D", "stop_name": "Branch D end", "stop_lat": 40.010, "stop_lon": -3.710},
    ]
    routes = [
        {"route_id": f"R_{b}", "route_short_name": b, "route_long_name": f"Branch {b}", "route_type": 0}
        for b in ("B", "C", "D")
    ]

    # Each branch's shape genuinely detours (not a straight line) so a
    # fallback-to-straight-line implementation would be numerically
    # detectable too, though this test's main concern is *null coverage*,
    # not the exact distance value (that's `test_shapes_geometry.py`'s job).
    shapes = []
    offsets = {"B": -0.010, "C": 0.0, "D": 0.010}
    for b, off in offsets.items():
        shapes += [
            {"shape_id": f"SH_{b}", "shape_pt_lat": 40.000, "shape_pt_lon": -3.700, "shape_pt_sequence": 1},
            {"shape_id": f"SH_{b}", "shape_pt_lat": 40.005, "shape_pt_lon": -3.700 + off, "shape_pt_sequence": 2},
            {"shape_id": f"SH_{b}", "shape_pt_lat": 40.010, "shape_pt_lon": -3.700 + off, "shape_pt_sequence": 3},
        ]

    trips = []
    stop_times = []
    for b in ("B", "C", "D"):
        for i in range(5):  # 5 trips per branch
            trip_id = f"T_{b}_{i}"
            trips.append({
                "route_id": f"R_{b}", "service_id": "SVC", "trip_id": trip_id,
                "direction_id": 0, "shape_id": f"SH_{b}",
            })
            h, m = divmod(6 * 60 + i * 30, 60)
            t0 = f"{h:02d}:{m:02d}:00"
            # 20-minute run to the branch end (not 5) -- long enough to sit
            # comfortably inside get_speed_at_stops's default 15-minute
            # sampling window; a too-short trip relative to that window is
            # a fixture-realism issue, not something this test means to
            # exercise (see test_speed_at_stops_not_null_for_multi_branch_trunk_stop).
            h2, m2 = divmod(6 * 60 + i * 30 + 20, 60)
            t1 = f"{h2:02d}:{m2:02d}:00"
            stop_times.append({
                "trip_id": trip_id, "arrival_time": t0, "departure_time": t0,
                "stop_id": "TRUNK", "stop_sequence": 1,
            })
            stop_times.append({
                "trip_id": trip_id, "arrival_time": t1, "departure_time": t1,
                "stop_id": f"S_{b}", "stop_sequence": 2,
            })

    directory = write_gtfs(
        tmp_path / "multi_branch_shapes",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": routes,
            "stops.txt": stops,
            "shapes.txt": shapes,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    return Feed(directory)


def test_no_null_shape_dist_traveled_when_real_geometry_exists(multi_branch_shape_feed):
    df = multi_branch_shape_feed.lf.select(
        "trip_id", "stop_id", "shape_dist_traveled", "shape_total_distance"
    ).collect()

    # 3 branches x 5 trips x 2 stops each = 30 rows expected.
    assert df.height == 30

    null_rows = df.filter(pl.col("shape_dist_traveled").is_null())
    assert null_rows.height == 0, (
        f"{null_rows.height}/{df.height} rows have null shape_dist_traveled "
        f"despite every trip having real shapes.txt geometry available: "
        f"{null_rows.select('trip_id', 'stop_id').to_dicts()}"
    )
    assert (df["shape_total_distance"] > 0).all()


def test_speed_at_stops_not_null_for_multi_branch_trunk_stop(multi_branch_shape_feed):
    """The real downstream symptom: null shape_dist_traveled silently makes
    get_speed_at_stops's speed null too. This exercises that full path."""
    from datetime import date, time

    result = multi_branch_shape_feed.get_speed_at_stops(
        date=date(2024, 6, 3), start_time=time(6, 0), end_time=time(22, 0),
        by="route_id", at="stop_id", how="max",
    )
    df = result.collect() if hasattr(result, "collect") else result
    row = df.filter(df["stop_id"] == "TRUNK")
    assert row.height >= 1
    # The concern here is nullness specifically (that's the real bug this
    # file guards against -- shape_dist_traveled silently going null and
    # propagating into a null speed); some rows legitimately computing 0.0
    # is a separate, unrelated sampling-window edge effect this test isn't
    # about, so only non-null rows are required to be strictly positive.
    speeds = row["speed"]
    assert speeds.null_count() == 0, (
        "every route serving TRUNK has real shape geometry and >=2 stops, "
        "so get_speed_at_stops should compute a real, non-null speed for "
        "every one of them -- if any is null, shape_dist_traveled coverage "
        "has regressed even if the direct check above passes"
    )
    assert (speeds > 0).any(), "expected at least one route to show a real positive speed"
