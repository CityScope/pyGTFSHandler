# -*- coding: utf-8 -*-
"""Edge (stop-to-stop segment) and per-stop speed/headway metrics for
`route_map.py`'s speed/headway map modes.

Kept separate from `route_map.py` to keep that file within a readable size.
Everything here works off the already-built per-day departures DataFrame
(`day`, from `_expand_frequencies` in `route_map.py`) rather than
re-querying the feed, so it's cheap to compute alongside the rest of the
map's payload.

For a directed edge (stop_id_A -> stop_id_B, consecutive stops on some
trip), several different shapes/routes may serve it. Three "representative"
shapes are picked per edge -- the busiest (most trips), the fastest
(highest average speed) and the shortest (shortest real-geometry
distance) -- so the client can draw the edge as a real curved polyline
(cut out of that shape's full geometry) rather than a straight line
between the two stops, and can switch which one is drawn as the user
switches between headway/speed map modes.
"""

from __future__ import annotations

import math
from typing import Optional

import polars as pl

try:
    from shapely.ops import substring
except Exception:  # pragma: no cover - shapely is a hard dependency elsewhere
    substring = None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _edge_trip_rows(day: pl.DataFrame) -> pl.DataFrame:
    """One row per (trip_id, consecutive stop pair): stop_id_A/B, the
    shape_id/route_id serving it and the time spent traversing it."""
    return (
        day.sort(["trip_id", "stop_sequence"])
        .with_columns(
            pl.col("stop_id").alias("stop_id_A"),
            pl.col("stop_id").shift(-1).over("trip_id").alias("stop_id_B"),
            pl.col("departure_time").alias("dep_A"),
            pl.col("arrival_time").shift(-1).over("trip_id").alias("arr_B"),
        )
        .filter(pl.col("stop_id_B").is_not_null())
        .select(
            [
                "trip_id",
                "route_id",
                "shape_id",
                "direction_id",
                "stop_id_A",
                "stop_id_B",
                "dep_A",
                "arr_B",
            ]
        )
        .drop_nulls(["stop_id_A", "stop_id_B"])
    )


def _shape_dist_lookup(shapes_stop_shapes: Optional[pl.LazyFrame]) -> dict:
    """(shape_id, stop_id) -> (dist_traveled_m, shape_total_distance_m)."""
    if shapes_stop_shapes is None:
        return {}
    df = (
        shapes_stop_shapes.select(["shape_id", "stop_id", "shape_dist_traveled", "shape_total_distance"])
        .drop_nulls(["shape_id", "stop_id"])
        .unique(subset=["shape_id", "stop_id"], keep="first")
        .collect()
    )
    out = {}
    for row in df.iter_rows(named=True):
        out[(row["shape_id"], row["stop_id"])] = (row["shape_dist_traveled"], row["shape_total_distance"])
    return out


def _extract_subline(geom, frac_a: float, frac_b: float) -> Optional[list]:
    if substring is None or geom is None:
        return None
    lo, hi = (frac_a, frac_b) if frac_a <= frac_b else (frac_b, frac_a)
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))
    if hi - lo < 1e-9:
        return None
    sub = substring(geom, lo, hi, normalized=True)
    coords = list(sub.coords)
    if len(coords) < 2:
        return None
    pts = [[lat, lon] for lon, lat in coords]
    if frac_a > frac_b:
        pts.reverse()
    return pts


def compute_edge_and_stop_metrics(day: pl.DataFrame, stops_df: pl.DataFrame, feed) -> dict:
    """Returns a dict with:
      - "edges": {edge_key: {a, b, route_id (best), n_trips, speed, headway,
        geom_freq, geom_fast, geom_short}}
      - "stops": {stop_id: {speed, headway}}   (combined across all routes)
      - "stop_routes_metrics": {stop_id: {route_id: {speed, headway}}}
    Speed is in km/h, headway in minutes. Any value that can't be computed
    (not enough data) is left out / null.
    """
    edge_trips = _edge_trip_rows(day)
    if edge_trips.height == 0:
        return {"edges": {}, "stops": {}, "stop_routes_metrics": {}}

    stop_coords = dict(
        zip(stops_df["stop_id"].to_list(), zip(stops_df["stop_lat"].to_list(), stops_df["stop_lon"].to_list()))
    )
    dist_lookup = _shape_dist_lookup(getattr(feed.shapes, "stop_shapes", None))

    def straight_dist(a, b):
        ca, cb = stop_coords.get(a), stop_coords.get(b)
        if ca is None or cb is None or ca[0] is None or cb[0] is None:
            return None
        return _haversine_m(ca[0], ca[1], cb[0], cb[1])

    def shape_dist(shape_id, a, b):
        da = dist_lookup.get((shape_id, a))
        db = dist_lookup.get((shape_id, b))
        if da is None or db is None:
            return None
        return abs(db[0] - da[0])

    rows = []
    for r in edge_trips.iter_rows(named=True):
        dur = None
        if r["arr_B"] is not None and r["dep_A"] is not None:
            dur = r["arr_B"] - r["dep_A"]
        dist = shape_dist(r["shape_id"], r["stop_id_A"], r["stop_id_B"]) if r["shape_id"] else None
        if dist is None:
            dist = straight_dist(r["stop_id_A"], r["stop_id_B"])
        speed = None
        if dist is not None and dur is not None and dur > 0:
            speed = (dist / 1000.0) / (dur / 3600.0)
        rows.append(
            {
                "stop_id_A": r["stop_id_A"],
                "stop_id_B": r["stop_id_B"],
                "shape_id": r["shape_id"],
                "route_id": r["route_id"],
                "dep_A": r["dep_A"],
                "speed": speed,
                "distance": dist,
            }
        )
    edge_trips_full = pl.DataFrame(rows)

    # Per (edge, shape) aggregates -- used to pick the busiest/fastest/
    # shortest representative shape for that directed edge.
    per_shape = (
        edge_trips_full.group_by(["stop_id_A", "stop_id_B", "shape_id"])
        .agg(
            pl.len().alias("n_trips"),
            pl.col("speed").drop_nulls().mean().alias("avg_speed"),
            pl.col("distance").drop_nulls().mean().alias("avg_distance"),
            pl.col("route_id").drop_nulls().first().alias("route_id"),
        )
        .drop_nulls(["shape_id"])
    )

    # Per-edge combined aggregate (across every shape/route serving it).
    per_edge = edge_trips_full.group_by(["stop_id_A", "stop_id_B"]).agg(
        pl.len().alias("n_trips"),
        pl.col("speed").drop_nulls().mean().alias("speed"),
        pl.col("dep_A").sort().alias("dep_times"),
        pl.col("route_id").unique().alias("route_ids"),
    )

    def headway_minutes(dep_times: list) -> Optional[float]:
        times = sorted(t for t in dep_times if t is not None)
        if len(times) < 2:
            return None
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not gaps:
            return None
        return (sum(gaps) / len(gaps)) / 60.0

    gdf = feed.shapes.gdf
    shape_geom = {}
    if gdf is not None and len(gdf):
        shape_geom = dict(zip(gdf["shape_id"], gdf.geometry))

    def representative_geom(shape_id, a, b):
        if not shape_id:
            return None
        geom = shape_geom.get(shape_id)
        da = dist_lookup.get((shape_id, a))
        db = dist_lookup.get((shape_id, b))
        if geom is None or da is None or db is None or not da[1] or not db[1]:
            return None
        total = da[1] or db[1]
        if not total:
            return None
        return _extract_subline(geom, da[0] / total, db[0] / total)

    edges_json = {}
    for row in per_edge.iter_rows(named=True):
        a, b = row["stop_id_A"], row["stop_id_B"]
        key = a + "->" + b
        shapes_here = per_shape.filter((pl.col("stop_id_A") == a) & (pl.col("stop_id_B") == b))
        if shapes_here.height:
            freq_row = shapes_here.sort("n_trips", descending=True).row(0, named=True)
            fast_rows = shapes_here.filter(pl.col("avg_speed").is_not_null()).sort("avg_speed", descending=True)
            fast_row = fast_rows.row(0, named=True) if fast_rows.height else freq_row
            short_rows = shapes_here.filter(pl.col("avg_distance").is_not_null()).sort("avg_distance")
            short_row = short_rows.row(0, named=True) if short_rows.height else freq_row
        else:
            freq_row = fast_row = short_row = None

        edges_json[key] = {
            "a": a,
            "b": b,
            "n_trips": int(row["n_trips"]),
            "route_ids": [r for r in row["route_ids"] if r is not None],
            "speed": row["speed"],
            "headway": headway_minutes(row["dep_times"]),
            "geom_freq": representative_geom(freq_row["shape_id"], a, b) if freq_row else None,
            "geom_fast": representative_geom(fast_row["shape_id"], a, b) if fast_row else None,
            "geom_short": representative_geom(short_row["shape_id"], a, b) if short_row else None,
        }

    # ------------------------------------------------------------------
    # Per-stop metrics: combine every edge touching a stop (as either end),
    # weighted by trip count, plus a per-(stop, route) breakdown used by the
    # timetable's Speed/Headway columns.
    # ------------------------------------------------------------------
    touch_rows = []
    for e in edges_json.values():
        for sid in (e["a"], e["b"]):
            touch_rows.append({"stop_id": sid, "speed": e["speed"], "headway": e["headway"], "n_trips": e["n_trips"]})
    touch_df = pl.DataFrame(touch_rows) if touch_rows else pl.DataFrame(schema={"stop_id": pl.Utf8})

    stops_metrics = {}
    if touch_df.height:
        agg = touch_df.group_by("stop_id").agg(
            (
                (pl.col("speed").fill_null(0) * pl.col("n_trips")).sum()
                / pl.col("n_trips").filter(pl.col("speed").is_not_null()).sum()
            ).alias("speed"),
            (
                (pl.col("headway").fill_null(0) * pl.col("n_trips")).sum()
                / pl.col("n_trips").filter(pl.col("headway").is_not_null()).sum()
            ).alias("headway"),
        )
        for row in agg.iter_rows(named=True):
            stops_metrics[row["stop_id"]] = {
                "speed": row["speed"] if row["speed"] is not None and not math.isnan(row["speed"]) else None,
                "headway": row["headway"] if row["headway"] is not None and not math.isnan(row["headway"]) else None,
            }

    # Per (stop, route_id): average of the speeds/headways of that route's
    # edges touching the stop.
    route_touch_rows = []
    for e in edges_json.values():
        for rid in e["route_ids"]:
            for sid in (e["a"], e["b"]):
                route_touch_rows.append(
                    {"stop_id": sid, "route_id": rid, "speed": e["speed"], "headway": e["headway"]}
                )
    stop_routes_metrics: dict = {}
    if route_touch_rows:
        rt_df = pl.DataFrame(route_touch_rows)
        rt_agg = rt_df.group_by(["stop_id", "route_id"]).agg(
            pl.col("speed").drop_nulls().mean().alias("speed"),
            pl.col("headway").drop_nulls().mean().alias("headway"),
        )
        for row in rt_agg.iter_rows(named=True):
            d = stop_routes_metrics.setdefault(row["stop_id"], {})
            d[row["route_id"]] = {
                "speed": row["speed"],
                "headway": row["headway"],
            }

    return {"edges": edges_json, "stops": stops_metrics, "stop_routes_metrics": stop_routes_metrics}
