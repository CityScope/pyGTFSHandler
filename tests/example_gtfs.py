"""Synthetic GTFS fixtures for the `examples/*.ipynb` notebooks.

Separate from `gtfs_builder.py` (whose `write_gtfs`/`minimal_agency` this
module reuses): those are generic test-fixture primitives, these are the
specific, named example networks (the "Y-shaped" trunk-and-branches feed)
the notebooks build and narrate against. Kept out of the notebooks
themselves so a notebook cell reads as "load this feed" rather than as
CSV-writing boilerplate.
"""

from __future__ import annotations

from pathlib import Path

from .gtfs_builder import minimal_agency, write_gtfs

DEFAULT_CALENDAR = [
    {
        "service_id": "SVC",
        "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
        "friday": 1, "saturday": 1, "sunday": 1,
        "start_date": "20240101", "end_date": "20261231",
    }
]

# The Y-shape's stops: a trunk `S1->S2->S3`, then two branches,
# `S3->S4->S5` and `S3->S6->S7`, ~100 degrees apart at S3. `S5` sits off
# the `S3->S4` line so the upper branch isn't perfectly straight either.
Y_STOPS = [
    {"stop_id": "S1", "stop_name": "S1", "stop_lat": 40.400, "stop_lon": -3.700},
    {"stop_id": "S2", "stop_name": "S2", "stop_lat": 40.400, "stop_lon": -3.690},
    {"stop_id": "S3", "stop_name": "S3", "stop_lat": 40.400, "stop_lon": -3.680},
    {"stop_id": "S4", "stop_name": "S4", "stop_lat": 40.413, "stop_lon": -3.67156},
    {"stop_id": "S5", "stop_name": "S5", "stop_lat": 40.413, "stop_lon": -3.66156},
    {"stop_id": "S6", "stop_name": "S6", "stop_lat": 40.38851, "stop_lon": -3.67156},
    {"stop_id": "S7", "stop_name": "S7", "stop_lat": 40.38851, "stop_lon": -3.66156},
]
Y_STOP_COORDS = {s["stop_id"]: (s["stop_lon"], s["stop_lat"]) for s in Y_STOPS}


def hms(seconds: int) -> str:
    """`seconds`-since-midnight as a GTFS `HH:MM:SS` string (hours can exceed 24)."""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def make_trips(prefix, route_id, headsign, stop_ids, start_sec, interval_sec, n_trips, step_sec=120):
    """`n_trips` explicit departures `interval_sec` apart, starting at
    `start_sec`, each advancing `step_sec` between consecutive stops --
    full control over exact clock times, unlike `frequencies.txt` (whose
    departure instants get aligned to the query window, not necessarily
    to the phase a caller actually specified).

    Returns:
        `(trips, stop_times)` row-lists, ready for `write_gtfs`.
    """
    trips, stop_times = [], []
    for k in range(n_trips):
        trip_id = f"{prefix}{k}"
        trips.append({"route_id": route_id, "service_id": "SVC", "trip_id": trip_id, "trip_headsign": headsign})
        t0 = start_sec + k * interval_sec
        for i, stop_id in enumerate(stop_ids):
            t = t0 + i * step_sec
            stop_times.append({
                "trip_id": trip_id, "arrival_time": hms(t), "departure_time": hms(t),
                "stop_id": stop_id, "stop_sequence": i + 1,
            })
    return trips, stop_times


# The notebook's query window always starts at 06:00 -- kept as a module
# constant so `build_y_shape*`'s warmup margin (see below) can line up
# with it exactly.
WINDOW_START_SEC = 21600

# How long before (and after) the query window each shape's schedule
# actually starts (and keeps running) -- long enough that every stop
# along the route, however many stops deep, already has a fully "warmed
# up" periodic pattern by the time the window opens. Without this
# margin, a stop several stops into a trip (e.g. S6, reached 6 minutes
# after the trip's own start) doesn't get its first departure until
# partway into the window, inflating the headway formula's wrap-around
# edge gap for that one stop -- a real but purely incidental artifact of
# where the window happens to start, not of the network or schedule
# itself, so it's avoided here rather than left for a reader to puzzle
# over.
WARMUP_MARGIN_SEC = 1800


def build_y_shape(dirpath: Path, route1: str, route2: str, headway1: int, headway2: int, offset2: int = 300, hours: int = 4) -> Path:
    """Writes one Y-shape GTFS feed: shape 1 (`S1->S2->S3->S4->S5`) on
    `route1` every `headway1` seconds, shape 2 (`S1->S2->S3->S6->S7`) on
    `route2` every `headway2` seconds, shape 2 offset `offset2` seconds
    later. `route1 == route2` puts both shapes on one `route_id`. Both
    schedules start `WARMUP_MARGIN_SEC` before the notebook's query
    window and keep running that long past its end too -- see
    `WARMUP_MARGIN_SEC`.
    """
    routes = [{"route_id": route1, "route_short_name": route1, "route_long_name": route1, "route_type": 3}]
    if route2 != route1:
        routes.append({"route_id": route2, "route_short_name": route2, "route_long_name": route2, "route_type": 3})
    span = hours * 3600 + 2 * WARMUP_MARGIN_SEC
    start1 = WINDOW_START_SEC - WARMUP_MARGIN_SEC
    start2 = WINDOW_START_SEC + offset2 - WARMUP_MARGIN_SEC
    t1, st1 = make_trips("T1_", route1, "To S5", ["S1", "S2", "S3", "S4", "S5"], start1, headway1, int(span / headway1))
    t2, st2 = make_trips("T2_", route2, "To S7", ["S1", "S2", "S3", "S6", "S7"], start2, headway2, int(span / headway2))
    return write_gtfs(dirpath, {
        "agency.txt": minimal_agency(), "calendar.txt": DEFAULT_CALENDAR, "routes.txt": routes,
        "stops.txt": Y_STOPS, "trips.txt": t1 + t2, "stop_times.txt": st1 + st2,
    })


def build_y_shape_four(dirpath: Path, headway1: int, headway2: int, headway3: int, headway4: int, offset2: int = 300, offset3: int = 150, offset4: int = 450, hours: int = 4) -> Path:
    """The same plain `Y_STOPS` network as `build_y_shape`, but with two
    more shapes added: shape 3 is the exact reverse of shape 1
    (`S5->S4->S3->S2->S1`), shape 4 the exact reverse of shape 2
    (`S7->S6->S3->S2->S1`).

    Unlike two branches sharing one trunk (which can *never* separate
    into different `n_divisions` sectors at `S3`, however wide their
    angle -- both reconcile against the same backward bearing, capping
    their corrected separation just under 90 degrees), a shape and its
    own exact reverse contribute a genuinely different backward bearing
    at `S3` (shape 3's backward point is shape 1's *forward* point), so
    outbound (1+2) and inbound (3+4) end up on opposite sides of the
    widest gap -- no artificial detour stop needed.
    """
    routes = [{"route_id": rid, "route_short_name": rid, "route_long_name": rid, "route_type": 3} for rid in ("L1", "L2", "L3", "L4")]
    span = hours * 3600 + 2 * WARMUP_MARGIN_SEC
    start1 = WINDOW_START_SEC - WARMUP_MARGIN_SEC
    start2 = WINDOW_START_SEC + offset2 - WARMUP_MARGIN_SEC
    start3 = WINDOW_START_SEC + offset3 - WARMUP_MARGIN_SEC
    start4 = WINDOW_START_SEC + offset4 - WARMUP_MARGIN_SEC
    t1, st1 = make_trips("T1_", "L1", "To S5", ["S1", "S2", "S3", "S4", "S5"], start1, headway1, int(span / headway1))
    t2, st2 = make_trips("T2_", "L2", "To S7", ["S1", "S2", "S3", "S6", "S7"], start2, headway2, int(span / headway2))
    t3, st3 = make_trips("T3_", "L3", "To S1 (via S5)", ["S5", "S4", "S3", "S2", "S1"], start3, headway3, int(span / headway3))
    t4, st4 = make_trips("T4_", "L4", "To S1 (via S7)", ["S7", "S6", "S3", "S2", "S1"], start4, headway4, int(span / headway4))
    return write_gtfs(dirpath, {
        "agency.txt": minimal_agency(), "calendar.txt": DEFAULT_CALENDAR, "routes.txt": routes,
        "stops.txt": Y_STOPS, "trips.txt": t1 + t2 + t3 + t4, "stop_times.txt": st1 + st2 + st3 + st4,
    })


def build_y_shape_variants(base_dir: Path) -> dict[str, Path]:
    """Writes the three Y-shape variants the notebook compares:
    `diff_routes`/`same_route` (both 10 min, differing only in
    `route_id` assignment) and `mixed_headway` (10 + 20 min, different
    `route_id`s).

    Returns:
        `{"diff_routes": Path, "same_route": Path, "mixed_headway": Path}`.
    """
    base_dir = Path(base_dir)
    return {
        "diff_routes": build_y_shape(base_dir / "diff_routes", "L1", "L2", 600, 600),
        "same_route": build_y_shape(base_dir / "same_route", "L1", "L1", 600, 600),
        "mixed_headway": build_y_shape(base_dir / "mixed_headway", "L1", "L2", 600, 1200),
    }
