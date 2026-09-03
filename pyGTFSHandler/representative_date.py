# -*- coding: utf-8 -*-
"""Picks a single real, "typical" service date out of a GTFS feed's
calendar window.

This is the real algorithm behind `Feed.get_representative_date` (kept as a
thin public wrapper in `feed.py`). It replaces an older, weaker proxy that
just counted active `service_id`s per candidate date
(`Calendar.get_services_in_date`) -- a count with no notion of whether a
date's service is actually full/typical, and one that can be dominated by
many small/seasonal calendars while a metro area's dominant,
highest-stop-density agency has zero calendar coverage that day at all
(this is exactly what caused Boston/San Francisco stops to vanish -- see
`transitlos.stops.download_and_prepare_stops`'s module docstring).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import polars as pl


def select_representative_date(
    calendar,
    trips_lf: pl.LazyFrame,
    stop_times_lf: pl.LazyFrame,
    stops_lf: pl.LazyFrame,
    start_date: date,
    end_date: date,
    date_type: Optional[str] = "weekday",
    routes_lf: Optional[pl.LazyFrame] = None,
    route_types: Optional[list] = None,
    max_candidate_weeks: int = 60,
) -> Optional[date]:
    """Picks the single most "typical" real service date in `[start_date,
    end_date]`.

    Algorithm (per the exact user spec):

    1. Stops are already grouped into `parent_station` clusters upstream
       (via `Stops.load(stop_group_distance=...)`) -- `stops_lf` is used
       only to map `stop_id -> parent_station`, no grouping happens here.
    2. For every (date, parent_station) pair in the window, compute the
       count of *distinct departure times* of trips serving that station on
       that date ("unique time" -- dedupes exact duplicate/erroneous
       double-entries at the same timestamp). This is done in one
       vectorized pass over the whole window: build the set of real
       (service_id, parent_station, departure_time) "visit" tuples once,
       get each service_id's per-date active/inactive status once (via
       `Calendar.get_services_in_date_range`, itself vectorized rather than
       a per-date Python loop), then join/explode the two -- never a
       per-candidate-date `.collect()`.
    3. Per date, take the MEDIAN of that per-parent-station distinct-time
       count across every parent station with any service that date --
       this is the date's raw score.
    4. Dates are grouped into Monday-Sunday ISO weeks. `date_type` (see
       `Calendar.VALID_DATE_TYPES`) determines which days of a week are
       "valid" (e.g. `"weekday"` -> Mon-Fri only, a single weekday name ->
       just that one day, `None` -> every day).
    5. Per week, take the MEDIAN of its valid days' raw scores -> the
       week's score.
    6. The week with the highest score wins (earliest week breaks ties).
    7. Within the winning week, return the single valid day with the
       *highest* raw score (most services) -- ties broken by earliest date.

    Args:
        calendar: `pyGTFSHandler.models.calendar.Calendar` instance (its
            `get_services_in_date_range` is reused rather than
            reimplemented -- it's already a real, vectorized way to expand
            a date range to per-date active service_ids in bulk).
        trips_lf: `trip_id`, `service_id`, `route_id` columns.
        stop_times_lf: `trip_id`, `stop_id`, `departure_time` columns
            (departure_time as the 0-24h seconds-of-day int this codebase
            normalizes to; day_offset/overnight precision is intentionally
            *not* modeled here -- see note below).
        stops_lf: `stop_id`, `parent_station` columns.
        start_date: Lower bound (inclusive) of the search window.
        end_date: Upper bound (inclusive) of the search window.
        date_type: Passed to `Calendar.filter_by_date_type` semantics (via
            `get_services_in_date_range`) to decide which weekdays count as
            "valid" within a week. `None` = every day.
        routes_lf: Optional `route_id`, `route_type` frame. If given
            together with `route_types`, trips are pre-filtered to only
            those route types before counting visits -- a cheap, natural
            extension of the old function's unused `route_types` param
            now that trip-level data is loaded anyway.
        route_types: Optional route type filter, only applied if
            `routes_lf` is also given.
        max_candidate_weeks: If the window spans more ISO weeks than this,
            weeks are evenly subsampled (not the first N) before scoring,
            matching the old function's `max_candidates` intent of not
            turning a multi-year calendar range into an unbounded scan.
            Each week that *is* evaluated is still scored exactly (no
            per-day subsampling within an evaluated week).

    Returns:
        The representative `datetime.date`, or `None` if the window is
        empty/invalid. If the window has zero real service at all, a
        best-effort date is still returned (least-bad among whatever was
        evaluated) rather than raising -- callers that need to detect "no
        real service" should inspect the surrounding calendar bounds
        themselves, matching the old function's contract.

    Note on day_offset/overnight trips: like the old proxy this scores
    against each stop_time's *nominal* `service_date` weekday pattern, not
    the day_offset-adjusted real calendar date `Feed._filter_by_date`
    resolves at query time -- acceptable here because this function only
    needs to rank *whole days* against each other by service volume, not
    resolve any individual stop_time's exact real-world date.
    """
    if start_date is None or end_date is None or start_date > end_date:
        return None

    # --- 0. Two-phase bound to `max_candidate_weeks` BEFORE the expensive
    # visits join happens. `Calendar.get_services_in_date_range` expands
    # every day in `[start_date, end_date]` to its active service_ids --
    # that alone is cheap (no trip-level data involved) even across a
    # multi-year range. What is NOT cheap is joining that against the real
    # per-station visit tuples: cost is proportional to (candidate days) x
    # (avg active service_ids/day) x (avg visits/service), which for a
    # multi-year unbounded range is large enough to OOM a memory-
    # constrained machine.
    #
    # So: phase A does ONE cheap full-range `get_services_in_date_range`
    # call and uses `len(service_ids)` per day (the OLD algorithm's proxy
    # score -- cheap, no join) purely to pick which `max_candidate_weeks`
    # weeks are worth scoring accurately. This matters for real multi-feed
    # metros where different agencies' `calendar.txt` validity windows
    # don't overlap and can be narrow relative to the full calendar span
    # (e.g. San Francisco's Muni feed is only valid ~5.5 months out of a
    # 7-year unbounded calendar range) -- a naive evenly-spaced week
    # subsample across the whole span can easily land entirely outside a
    # dominant agency's real window and never discover it. Ranking ALL
    # weeks by this cheap coarse score first, then only doing the accurate
    # (phase B) distinct-departure-time scoring on the top-scoring weeks,
    # keeps the real algorithm's fidelity while still bounding total cost.
    per_date_all = calendar.get_services_in_date_range(start_date, end_date, date_type=None)
    if per_date_all.is_empty():
        return None

    # ISO (year, week) per date -- plain Python over the date list (one
    # tuple per calendar day in range, cheap: no polars/joins involved).
    dates_list = per_date_all["date"].to_list()
    iso_weeks = [d.isocalendar()[:2] for d in dates_list]
    per_date_all = per_date_all.with_columns(
        pl.Series("iso_year", [w[0] for w in iso_weeks]),
        pl.Series("iso_week", [w[1] for w in iso_weeks]),
    )

    n_weeks = len(set(iso_weeks))
    if n_weeks > max_candidate_weeks:
        coarse_scores = (
            per_date_all.lazy()
            .with_columns(pl.col("service_ids").list.len().alias("coarse_score"))
            .group_by(["iso_year", "iso_week"])
            .agg(pl.col("coarse_score").median().alias("week_coarse_score"))
            .sort("week_coarse_score", descending=True)
            .head(max_candidate_weeks)
            .collect()
        )
        per_date = per_date_all.join(
            coarse_scores.select(["iso_year", "iso_week"]), on=["iso_year", "iso_week"], how="semi"
        )
    else:
        per_date = per_date_all

    if per_date.is_empty():
        return None

    # Which days are "valid" for the requested date_type (weekday/weekend/
    # a single named weekday/etc). Only valid days feed into a week's
    # median (step 5); every day still gets a raw score (step 3).
    if date_type is not None:
        valid_dates = set(
            calendar.filter_by_date_type(per_date, date_type, None, None)["date"].to_list()
        )
    else:
        valid_dates = set(per_date["date"].to_list())

    if not valid_dates:
        # Nothing matches the date_type at all in this window -- fall back
        # to treating every day as valid rather than returning None, so a
        # caller-supplied date_type that happens to match nothing in a
        # short window still gets a best-effort date.
        valid_dates = set(per_date["date"].to_list())

    # --- 2. Real (service_id, parent_station, departure_time) visits -----
    trips = trips_lf.select(
        [pl.col("trip_id"), pl.col("service_id").cast(pl.Utf8), pl.col("route_id")]
    ).unique()
    if routes_lf is not None and route_types:
        route_types_norm = (
            [route_types] if isinstance(route_types, (int, str)) else list(route_types)
        )
        keep_routes = routes_lf.filter(pl.col("route_type").is_in(route_types_norm)).select("route_id")
        trips = trips.join(keep_routes, on="route_id", how="semi")

    stop_times = stop_times_lf.select(["trip_id", "stop_id", "departure_time"])
    stops = stops_lf.select(["stop_id", "parent_station"]).unique()

    visits = (
        stop_times.join(trips, on="trip_id", how="inner")
        .join(stops, on="stop_id", how="inner")
        .select(["service_id", "parent_station", "departure_time"])
        .unique()  # dedupe exact duplicate/erroneous same-time entries
    )

    service_dates = (
        per_date.lazy()
        .select(["date", "service_ids"])
        .explode("service_ids")
        .rename({"service_ids": "service_id"})
        .with_columns(pl.col("service_id").cast(pl.Utf8))
        .drop_nulls("service_id")
    )

    # --- 3. Per (date, parent_station) distinct-departure-time count -----
    per_station_daily = (
        service_dates.join(visits, on="service_id", how="inner")
        .select(["date", "parent_station", "departure_time"])
        .unique()
        .group_by(["date", "parent_station"])
        .agg(pl.len().alias("n_visits"))
    )

    # --- 4. Per-day median across parent stations -------------------------
    daily_scores = (
        per_station_daily.group_by("date")
        .agg(pl.median("n_visits").alias("day_score"))
        .collect()
    )

    if daily_scores.is_empty():
        # No service anywhere in the window -- best-effort fallback.
        fallback = sorted(valid_dates)
        return fallback[0] if fallback else sorted(per_date["date"].to_list())[0]

    day_score_map = dict(zip(daily_scores["date"].to_list(), daily_scores["day_score"].to_list()))

    # --- 5/6. Bucket into ISO weeks, score = median of valid days --------
    all_dates = sorted(per_date["date"].to_list())
    weeks: dict[tuple[int, int], list[date]] = {}
    for d in all_dates:
        if d not in valid_dates:
            continue
        iso_year, iso_week, _ = d.isocalendar()
        weeks.setdefault((iso_year, iso_week), []).append(d)

    if not weeks:
        fallback = sorted(valid_dates) or all_dates
        return fallback[0]

    # Week subsampling to `max_candidate_weeks` already happened in step 0
    # (before the expensive calendar-expansion/join above), so every key
    # here is already one of the chosen weeks -- no further trimming needed.
    week_keys = sorted(weeks.keys())

    best_week_key = None
    best_week_score = -1.0
    for wk in week_keys:
        day_scores = [day_score_map.get(d, 0.0) for d in weeks[wk]]
        week_median = float(np.median(day_scores)) if day_scores else 0.0
        if week_median > best_week_score:
            best_week_score = week_median
            best_week_key = wk

    if best_week_key is None:
        fallback = sorted(valid_dates)
        return fallback[0]

    # --- 7. Within winning week, pick the day with the most services -----
    # (per explicit user instruction: ties in the week-level median are
    # broken by taking, among that week's valid days, the single day with
    # the highest raw score -- not the day closest to the week's median.
    # Ties between equally-highest days are broken by earliest date.)
    winning_days = weeks[best_week_key]
    best_day = None
    best_score = None
    for d in sorted(winning_days):
        score = day_score_map.get(d, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_day = d

    return best_day
