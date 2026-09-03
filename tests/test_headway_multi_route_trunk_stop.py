"""Regression test for a real bug found downstream (transitLOS, 2026-08-12):
`get_headway_at_stops(how="best")`, used as transitLOS's default, reported a
49.68-minute headway for the MBTA Green Line at a shared trunk station
served by 4 branches (B/C/D/E) whose real combined frequency is close to
4-6 minutes -- because `how="best"` groups per `(stop, route, direction)`
and keeps only the single most-frequent one, which can *never* reflect
combined ridership-relevant frequency at a multi-route stop no matter how
good the per-route data is.

None of this package's existing headway tests exercise more than one route
serving the same stop at the same time, so this class of bug had no test
coverage at all -- every existing fixture is structurally a single-branch
station. This file adds that missing scenario and locks in the fix
transitLOS adopted (`how="add", mix_directions=False`): combine all routes'
rates within a direction via harmonic sum, then keep whichever direction has
the better combined headway.
"""

from __future__ import annotations

from datetime import date, time

import pytest

from pyGTFSHandler.feed import Feed

from .gtfs_builder import minimal_agency, write_gtfs

TEST_DATE = date(2024, 6, 3)  # a Monday


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
def trunk_station_feed(tmp_path) -> Feed:
    """3 branches (like Green Line B/C/D), each running every 20 minutes in
    one direction, all serving the same physical trunk stop -- analogous in
    structure (not scale) to Boylston station's B/C/D/E branches. Combined,
    a rider sees a train roughly every 20/3 ~= 6.7 minutes; no single branch
    on its own runs anywhere near that often.
    """
    stops = [
        {"stop_id": "TRUNK", "stop_name": "Trunk Station", "stop_lat": 40.000, "stop_lon": -3.700},
        {"stop_id": "S_B", "stop_name": "Branch B end", "stop_lat": 40.010, "stop_lon": -3.700},
        {"stop_id": "S_C", "stop_name": "Branch C end", "stop_lat": 40.020, "stop_lon": -3.700},
        {"stop_id": "S_D", "stop_name": "Branch D end", "stop_lat": 40.030, "stop_lon": -3.700},
    ]
    routes = [
        {"route_id": f"R_{b}", "route_short_name": b, "route_long_name": f"Branch {b}", "route_type": 0}
        for b in ("B", "C", "D")
    ]

    trips = []
    stop_times = []
    # Each branch: one trip every 20 minutes from 06:00 to 22:00 (48 trips),
    # direction_id=0, calling at TRUNK then its own branch-specific end stop.
    for b in ("B", "C", "D"):
        for i, minute in enumerate(range(0, 16 * 60, 20)):
            trip_id = f"T_{b}_{i}"
            trips.append({
                "route_id": f"R_{b}", "service_id": "SVC", "trip_id": trip_id, "direction_id": 0,
            })
            h, m = divmod(6 * 60 + minute, 60)
            t0 = f"{h:02d}:{m:02d}:00"
            h2, m2 = divmod(6 * 60 + minute + 5, 60)
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
        tmp_path / "trunk_station",
        {
            "agency.txt": minimal_agency(),
            "calendar.txt": _calendar(),
            "routes.txt": routes,
            "stops.txt": stops,
            "trips.txt": trips,
            "stop_times.txt": stop_times,
        },
    )
    return Feed(directory)


def test_best_mode_cannot_beat_single_branch_headway(trunk_station_feed):
    """Locks in the *problem*: how="best" only ever reports one branch's own
    ~20-minute headway, never the combined ~6.7-minute rider-experienced
    frequency, no matter how many branches serve the stop.

    """
    result = trunk_station_feed.get_headway_at_stops(
        TEST_DATE, start_time=time(6, 0), end_time=time(22, 0),
        by="route_id", at="stop_id", how="best",
    )
    row = result.filter(result["stop_id"] == "TRUNK")
    assert row.height == 1
    headway = row["headway"][0]
    # Close to a single branch's own 20-min headway -- nowhere near the true
    # ~6.7-min combined frequency.
    assert 15.0 < headway < 25.0


def test_add_mode_combines_branches_into_much_better_headway(trunk_station_feed):
    """Locks in the *fix*: how="add" (mix_directions=False, transitLOS's new
    default) combines all 3 branches' rates and should land close to the
    true combined ~6.7-minute frequency -- a real, large improvement over
    any single branch's own headway, not a marginal one."""
    result = trunk_station_feed.get_headway_at_stops(
        TEST_DATE, start_time=time(6, 0), end_time=time(22, 0),
        by="route_id", at="stop_id", how="add", mix_directions=False,
    )
    row = result.filter(result["stop_id"] == "TRUNK")
    assert row.height == 1
    headway = row["headway"][0]
    assert headway is not None
    # The core regression-guard invariant: combining routes at a shared
    # stop must produce a *materially* better (lower) headway than any
    # single route could -- roughly headway/n_routes, with slack for the
    # harmonic-sum formula's own variance-weighting behavior.
    assert headway < 12.0, (
        f"combined headway {headway:.2f} min is not much better than a single "
        "branch's own ~20 min -- how='add' should combine rates across "
        "routes at a shared stop, not silently degrade to single-route "
        "behavior (this is the exact shape of the bug found in transitLOS "
        "against the real MBTA Green Line, see module docstring)"
    )


def test_add_mode_beats_best_mode_at_a_multi_route_stop(trunk_station_feed):
    """Direct comparison, same feed, same window: combining must never be
    worse than picking a single route -- if this regresses to `add ==
    best` (or worse), some code path has silently stopped combining rates."""
    best = trunk_station_feed.get_headway_at_stops(
        TEST_DATE, start_time=time(6, 0), end_time=time(22, 0),
        by="route_id", at="stop_id", how="best",
    )
    added = trunk_station_feed.get_headway_at_stops(
        TEST_DATE, start_time=time(6, 0), end_time=time(22, 0),
        by="route_id", at="stop_id", how="add", mix_directions=False,
    )
    best_headway = best.filter(best["stop_id"] == "TRUNK")["headway"][0]
    added_headway = added.filter(added["stop_id"] == "TRUNK")["headway"][0]
    assert added_headway < best_headway * 0.75
