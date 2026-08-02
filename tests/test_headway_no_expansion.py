"""Equivalence tests for the no-expansion headway computation path.

`Feed.get_headway_at_stops` currently computes headways by fully expanding
`frequencies.txt` rows into materialized per-departure stop_times rows (via
`Feed._frequencies_to_stop_times`), which is expensive. These tests lock in
the CURRENT (expansion-based) behavior as a baseline, and then check that a
new no-expansion code path (`Feed.get_headway_at_stops_no_expand`) produces
numerically identical results on a fixture deliberately designed so that
frequency windows do not divide evenly by their headway (T3's 06:00-08:30
window at 1200s headway = 7.5 periods), which stresses alignment-phase
handling at segment boundaries.
"""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from pyGTFSHandler.feed import Feed

FIXTURE_DIR = Path(__file__).parent / "gtfs_files" / "frequency_no_expansion"
TEST_DATE = date(2024, 6, 3)  # a Monday within calendar's active range


@pytest.fixture(scope="module")
def feed() -> Feed:
    return Feed(str(FIXTURE_DIR))


def _sort_cols(df: pl.DataFrame) -> list[str]:
    cols = []
    for c in ("stop_id", "parent_station", "route_id", "direction_id"):
        if c in df.columns:
            cols.append(c)
    return cols


@pytest.mark.parametrize(
    "start_time,end_time",
    [
        (time(0, 0), time(23, 59, 59)),
        (time(5, 30), time(8, 45)),  # straddles T1's 06:00/07:00 and T3's 06:00/08:30 boundaries
    ],
)
def test_baseline_current_implementation_runs_and_is_sane(feed, start_time, end_time):
    result = feed.get_headway_at_stops(
        TEST_DATE,
        start_time=start_time,
        end_time=end_time,
        by="route_id",
        at="stop_id",
        how="all",
    )
    assert result.height > 0
    assert "headway" in result.columns
    headways = result["headway"].drop_nulls()
    assert (headways > 0).all()


@pytest.mark.parametrize(
    "start_time,end_time",
    [
        (time(0, 0), time(23, 59, 59)),
        (time(5, 30), time(8, 45)),
    ],
)
@pytest.mark.parametrize("by,at", [("route_id", "stop_id")])
@pytest.mark.parametrize("how", ["all", "add", "best"])
def test_no_expand_matches_expansion_based(feed, start_time, end_time, by, at, how):
    expected = feed.get_headway_at_stops(
        TEST_DATE, start_time=start_time, end_time=end_time, by=by, at=at, how=how
    )
    actual = feed.get_headway_at_stops_no_expand(
        TEST_DATE, start_time=start_time, end_time=end_time, by=by, at=at, how=how
    )

    sort_cols = _sort_cols(expected)
    expected = expected.sort(sort_cols)
    actual = actual.sort(sort_cols)

    assert expected.height == actual.height
    assert list(expected.columns) == list(actual.columns) or set(expected.columns) == set(actual.columns)

    # Compare headway numerically with a tight tolerance (float summation
    # order can differ slightly between the two implementations).
    exp_headway = expected["headway"].to_numpy()
    act_headway = actual["headway"].to_numpy()
    import numpy as np

    assert exp_headway.shape == act_headway.shape
    assert np.allclose(exp_headway, act_headway, rtol=1e-9, atol=1e-9, equal_nan=True)
