"""Unit tests for `utils.time_parsing.time_displacement`, isolated from the
full GTFS-loading pipeline (it only needs `trip_id`, `shape_time_traveled`,
`shape_total_travel_time`, `shape_dist_traveled` columns).

Regression coverage for a real bug found downstream (transitLOS, 2026-08-12):
`analysis/stops.py`'s `get_speed_at_stops` returned `null` speed for
30-90% of stops on real MBTA GTFS data (every Green Line branch at Boylston
station, among many others). Traced to this function silently backfilling
any `join_asof` miss (`t_lb`/`t_ub`/`d_lb`/`d_ub`) with the row's OWN point
*before* computing `distance_weight`, which produced two distinct failure
modes once nothing could ever be null again:

1. Whenever `target_time` landed exactly on another stop's own
   `shape_time_traveled` (very common with real, minute-rounded transit
   schedules), both the backward and forward `join_asof` find that *same*
   point, so `t_lb == t_ub`. The old `(d_ub - d_lb) / (t_ub - t_lb))`
   divided by zero -> NaN -> silently null `distance_weight`, even though
   the answer (`d_lb`, no interpolation needed) required no division at
   all.
2. Whenever only one side of the window had a real match (`target_time`
   beyond the trip's first/last known point), the old code's fallback
   could produce `t_ub < t_lb` (inverted), silently extrapolating through
   a wrong-signed denominator instead of clamping to the known side.

No existing test exercised either case (`test_interpolation.py` covers a
different concern -- missing `departure_time` values, not this along-shape
distance-at-time-offset lookup) -- this file adds both directly.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from pyGTFSHandler.utils.time_parsing import time_displacement


def _frame(rows: list[dict]) -> pl.LazyFrame:
    return pl.DataFrame(rows).lazy()


def test_exact_tie_does_not_produce_null_or_nan():
    """target_time landing exactly on another stop's own shape_time_traveled
    (t_lb == t_ub) must return distance_weight == 0 (no interpolation
    needed), not a null/NaN from a 0/0 division."""
    # 3 stops on one trip, 300s apart, distances 0/1000/2000m. A forward
    # displacement of 300s from the first stop (t=0) lands exactly on the
    # second stop's own shape_time_traveled (t=300) -- an exact tie.
    rows = [
        {"trip_id": "T1", "shape_time_traveled": 0, "shape_total_travel_time": 600, "shape_dist_traveled": 0.0},
        {"trip_id": "T1", "shape_time_traveled": 300, "shape_total_travel_time": 600, "shape_dist_traveled": 1000.0},
        {"trip_id": "T1", "shape_time_traveled": 600, "shape_total_travel_time": 600, "shape_dist_traveled": 2000.0},
    ]
    result = time_displacement(_frame(rows), secs_disp=300).collect()

    row0 = result.filter(pl.col("shape_time_traveled") == 0)
    assert row0.height == 1
    dw = row0["distance_weight"][0]
    assert dw is not None, "exact-tie case must not produce a null distance_weight"
    assert not (isinstance(dw, float) and math.isnan(dw))
    # Target position (t=300) is exactly stop 2's own position (1000m);
    # distance_weight is |target_position - own_position| = |1000 - 0| = 1000.
    assert dw == pytest.approx(1000.0)


def test_one_sided_window_clamps_instead_of_inverting():
    """target_time beyond the trip's last known point (only t_lb found, t_ub
    genuinely missing) must clamp to the known side (d_lb), not silently
    extrapolate through an inverted/negative window."""
    rows = [
        {"trip_id": "T1", "shape_time_traveled": 0, "shape_total_travel_time": 1000, "shape_dist_traveled": 0.0},
        {"trip_id": "T1", "shape_time_traveled": 100, "shape_total_travel_time": 1000, "shape_dist_traveled": 500.0},
        # Last known point far from the trip's nominal total_travel_time --
        # a forward window from this row overshoots past every other row.
        {"trip_id": "T1", "shape_time_traveled": 200, "shape_total_travel_time": 1000, "shape_dist_traveled": 900.0},
    ]
    result = time_displacement(_frame(rows), secs_disp=900).collect()

    last_row = result.filter(pl.col("shape_time_traveled") == 200)
    assert last_row.height == 1
    dw = last_row["distance_weight"][0]
    # Clamped to the last known distance (900.0) minus the row's own
    # distance (900.0) = 0 -- not a negative/inverted extrapolation, and
    # not null.
    assert dw is not None
    assert dw == pytest.approx(0.0)


def test_no_nan_distance_weight_across_many_closely_spaced_stops():
    """A denser, more realistic trip (stops every ~60-120s, matching the
    minute-level rounding real GTFS schedules commonly have) should never
    produce a NaN distance_weight -- the aggregate-coverage regression
    guard for this bug (the real MBTA data triggered it on ~30-90% of
    rows)."""
    times = [0, 60, 180, 240, 360, 420, 480, 600, 660, 780, 900, 960, 1080, 1200]
    rows = [
        {
            "trip_id": "T1",
            "shape_time_traveled": t,
            "shape_total_travel_time": 1200,
            "shape_dist_traveled": float(t) * 3.2,  # arbitrary monotonic distance
        }
        for t in times
    ]
    for secs_disp in (900, -900, 300, -300):
        result = time_displacement(_frame(rows), secs_disp=secs_disp).collect()
        nan_count = result["distance_weight"].is_nan().sum()
        assert nan_count == 0, f"secs_disp={secs_disp}: {nan_count} NaN distance_weight rows"
