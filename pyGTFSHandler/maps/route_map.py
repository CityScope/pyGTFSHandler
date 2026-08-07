# -*- coding: utf-8 -*-
"""Builds the interactive per-stop / per-route / per-trip folium map.

All GTFS-specific data prep (stops/routes/shapes/departures/trip
itineraries, for a single service `date`) happens here in polars/Python and
is serialized to a compact JSON blob; the actual click-driven interactivity
(mode filters, stop -> timetable, timetable row -> trip itinerary, prev/next
same-route navigation) is plain vanilla JS in `static/route_map.js`/`.css`,
injected into the folium map as a single `folium.Element` rather than
folium's own marker/popup API, which can't express this kind of
client-side-recomputed, multi-box interaction.
"""

from __future__ import annotations

import json
from datetime import date as date_type
from importlib import resources
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import folium
import polars as pl

from .style import (
    ROUTE_TYPE_EMOJI,
    ROUTE_TYPE_EMOJI_FALLBACK,
    ROUTE_TYPE_NAME,
    ROUTE_TYPE_NAME_FALLBACK,
    SQUARE_BADGE_ROUTE_TYPES,
)
from .edge_metrics import compute_edge_and_stop_metrics

if TYPE_CHECKING:
    from ..feed import Feed


def _read_static(name: str) -> str:
    return (Path(__file__).parent / "static" / name).read_text(encoding="utf-8")


def _expand_frequencies(day: pl.DataFrame) -> pl.DataFrame:
    """Expands `frequencies.txt`-defined rows (one row per template/block per
    stop, carrying `start_time`/`end_time`/`headway_secs`) into concrete
    per-departure rows with distinct `departure_time`/`arrival_time`.

    Deliberately done here rather than via `feed.filter(..., frequencies=False)`:
    that decomposes frequency trips across the *entire* feed's stop_times
    (heavier, and duplicates every column per instance), whereas
    `feed.filter_by_date` already gives one lightweight row per template/
    block/stop -- this only expands the handful of columns the map actually
    needs, and only for frequency rows.

    Also fixes a real duplicate-departure bug: `frequencies.txt` commonly
    lists several time-of-day blocks (e.g. peak/off-peak headways) for the
    *same* `trip_id`, and independently generating each block's instances can
    produce the same absolute instance start time at the boundary between two
    adjacent blocks. Deduplicating by `(trip_id, instance_start)` before
    building stop rows removes that double-count regardless of which block
    "produced" it.
    """
    is_freq = day["start_time"].is_not_null()
    plain = day.filter(~is_freq).drop(["start_time", "end_time", "headway_secs"])
    freq = day.filter(is_freq)
    if freq.height == 0:
        return plain

    first_dep = freq.group_by("trip_id").agg(pl.col("departure_time").min().alias("first_dep"))
    freq = freq.join(first_dep, on="trip_id", how="left")

    instances = (
        freq.select(["trip_id", "start_time", "end_time", "headway_secs"])
        .unique()
        .filter(pl.col("headway_secs") > 0)
        .with_columns(
            pl.int_ranges(pl.col("start_time"), pl.col("end_time") + 1, pl.col("headway_secs")).alias(
                "instance_start"
            )
        )
        .explode("instance_start")
        .unique(subset=["trip_id", "instance_start"])
        .sort(["trip_id", "instance_start"])
        .with_columns((pl.col("instance_start").cum_count().over("trip_id") - 1).alias("inst_idx"))
        .select(["trip_id", "instance_start", "inst_idx"])
    )

    expanded = (
        freq.join(instances, on="trip_id", how="inner")
        .with_columns(
            (pl.col("departure_time") - pl.col("first_dep") + pl.col("instance_start")).alias("departure_time"),
            (pl.col("arrival_time") - pl.col("first_dep") + pl.col("instance_start")).alias("arrival_time"),
            (pl.col("trip_id") + pl.lit("#") + pl.col("inst_idx").cast(pl.Utf8)).alias("trip_id"),
        )
        .select(plain.columns)
    )

    result = pl.concat([plain, expanded], how="vertical_relaxed")

    # Final safety net: two *different* trip_ids (e.g. two of a frequency
    # trip's several time-of-day blocks landing on the same instant at a
    # block boundary, or an upstream data quirk producing a stray duplicate
    # block) can still end up describing the literal same real-world
    # departure -- same route+direction+shape, same first-stop time. Keep
    # only one trip_id per such signature (a stable, arbitrary pick) so
    # nothing downstream double-counts it as two separate trips.
    trip_signature = (
        result.sort("stop_sequence")
        .group_by("trip_id")
        .agg(
            pl.col("route_id").first(),
            pl.col("direction_id").first(),
            pl.col("shape_id").first(),
            pl.col("departure_time").first().alias("trip_start"),
        )
        .sort("trip_id")
        .unique(subset=["route_id", "direction_id", "shape_id", "trip_start"], keep="first")
        .select("trip_id")
    )
    return result.join(trip_signature, on="trip_id", how="semi")


def route_map(feed: "Feed", date: date_type, m: Optional[folium.Map] = None, zoom_start: int = 13) -> folium.Map:
    """Builds a self-contained interactive Leaflet map of `feed`'s stops and
    routes active on `date`.

    Args:
        feed: A `pyGTFSHandler.feed.Feed` instance.
        date: The service day to filter trips/departures to (via
            `feed.filter_by_date`, which resolves GTFS calendar/
            calendar_dates -- including day_offset for post-midnight trips --
            rather than re-implementing that logic here).
        m: An existing `folium.Map` to add the layer to. If omitted, a new
            one is created, centered on the feed's stops, using CartoDB
            positron tiles.
        zoom_start: Initial zoom level, only used when `m` is created here.

    Returns:
        The `folium.Map` (same object as `m`, if given).
    """
    day_lf = feed.filter_by_date(date)

    # `start_time`/`end_time`/`headway_secs` only exist at all when the feed
    # has a `frequencies.txt` (see `Feed.build_lf`); a feed without one (e.g.
    # a plain bus network) simply has no frequency rows to expand.
    freq_cols = [c for c in ("start_time", "end_time", "headway_secs") if c in day_lf.collect_schema().names()]
    day = day_lf.select(
        [
            "trip_id",
            "stop_id",
            "route_id",
            "route_type",
            "shape_id",
            "stop_sequence",
            "departure_time",
            "arrival_time",
            "parent_station",
            "direction_id",
        ]
        + freq_cols
    ).collect()
    if not freq_cols:
        day = day.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("start_time"),
            pl.lit(None, dtype=pl.Int64).alias("end_time"),
            pl.lit(None, dtype=pl.Int64).alias("headway_secs"),
        )
    day = _expand_frequencies(day)

    stops_df = feed.stops.lf.select(
        ["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"]
    ).collect()

    routes_df = feed.routes.lf.select(
        ["route_id", "route_short_name", "route_long_name", "route_type", "route_color", "route_text_color"]
    ).unique(subset=["route_id"]).collect()

    trips_headsign = (
        feed.trips.lf.select(["trip_id", "trip_headsign"]).unique(subset=["trip_id"]).collect()
        if "trip_headsign" in feed.trips.lf.collect_schema().names()
        else None
    )

    # ------------------------------------------------------------------
    # stops: name/coords/parent + which route_types serve this *exact*
    # stop_id (not the whole parent_station group -- a platform serving only
    # buses shouldn't show a train emoji just because a sibling platform at
    # the same station serves trains).
    # ------------------------------------------------------------------
    parent_name = dict(zip(stops_df["stop_id"].to_list(), stops_df["stop_name"].to_list()))

    modes_by_stop: dict[str, set] = {}
    for sid_, rtype in zip(day["stop_id"].to_list(), day["route_type"].to_list()):
        modes_by_stop.setdefault(sid_, set()).add(str(rtype))

    stops_json = {}
    for row in stops_df.iter_rows(named=True):
        sid = row["stop_id"]
        parent = row["parent_station"] or sid
        modes = sorted(modes_by_stop.get(sid, set()))
        stops_json[sid] = {
            "stop_name": row["stop_name"],
            "lat": row["stop_lat"],
            "lon": row["stop_lon"],
            "parent": parent,
            "parent_name": parent_name.get(parent, row["stop_name"]),
            "modes": modes,
        }

    # ------------------------------------------------------------------
    # routes, plus a `service_count` (number of distinct trips run in
    # whichever single direction runs more of them, i.e. never double-
    # counting a round trip) used client-side to always list route icons
    # from most- to least-served.
    # ------------------------------------------------------------------
    service_count = (
        day.select(["route_id", "direction_id", "trip_id"])
        .unique()
        .group_by(["route_id", "direction_id"])
        .agg(pl.len().alias("cnt"))
        .group_by("route_id")
        .agg(pl.col("cnt").max().alias("service_count"))
    )
    service_count_map = dict(zip(service_count["route_id"].to_list(), service_count["service_count"].to_list()))

    routes_json = {}
    for row in routes_df.iter_rows(named=True):
        routes_json[row["route_id"]] = {
            "route_short_name": row["route_short_name"],
            "route_long_name": row["route_long_name"],
            "route_type": str(row["route_type"]),
            "route_color": (row["route_color"] or "3388ff").lstrip("#"),
            "route_text_color": (row["route_text_color"] or "ffffff").lstrip("#"),
            "service_count": int(service_count_map.get(row["route_id"], 0)),
        }

    # ------------------------------------------------------------------
    # shapes (real polyline geometry, indexed by shape_id -- shared across
    # trips, so serialized once instead of duplicated per trip/stop).
    # ------------------------------------------------------------------
    shapes_json = {}
    shape_route = {}
    if feed.shapes.gdf is not None and len(feed.shapes.gdf):
        gdf = feed.shapes.gdf
        shape_ids_used = set(day["shape_id"].drop_nulls().to_list())
        for shape_id, geom in zip(gdf["shape_id"], gdf.geometry):
            if shape_id not in shape_ids_used or geom is None:
                continue
            coords = list(geom.coords)
            shapes_json[shape_id] = [[lat, lon] for lon, lat in coords]

    for shape_id, route_id in (
        day.select(["shape_id", "route_id"]).drop_nulls().unique(subset=["shape_id"]).iter_rows()
    ):
        shape_route[shape_id] = route_id

    # ------------------------------------------------------------------
    # stop_routes / stop_shapes: what serves each stop, for badges + the
    # highlight overlay drawn on stop click.
    # ------------------------------------------------------------------
    stop_routes: dict[str, list] = {}
    stop_shapes: dict[str, list] = {}
    stop_route_pairs = (
        day.select(["stop_id", "route_id"]).drop_nulls().unique().join(
            routes_df.select(["route_id", "route_short_name"]), on="route_id", how="left"
        ).sort(["stop_id", "route_short_name"])
    )
    for sid, rid in zip(stop_route_pairs["stop_id"].to_list(), stop_route_pairs["route_id"].to_list()):
        stop_routes.setdefault(sid, [])
        if rid not in stop_routes[sid]:
            stop_routes[sid].append(rid)

    stop_shape_pairs = day.select(["stop_id", "shape_id"]).drop_nulls().unique()
    for sid, shid in zip(stop_shape_pairs["stop_id"].to_list(), stop_shape_pairs["shape_id"].to_list()):
        stop_shapes.setdefault(sid, [])
        if shid not in stop_shapes[sid]:
            stop_shapes[sid].append(shid)

    # ------------------------------------------------------------------
    # departures per stop (for the timetable box) -- destination is each
    # trip's last stop by stop_sequence.
    # ------------------------------------------------------------------
    last_stop = (
        day.sort("stop_sequence")
        .group_by("trip_id")
        .agg(pl.col("stop_id").last().alias("dest_stop_id"))
    )
    dep_df = (
        day.select(["trip_id", "stop_id", "route_id", "departure_time"])
        .join(last_stop, on="trip_id", how="left")
        .drop_nulls(subset=["departure_time"])
        .sort("departure_time")
    )
    departures: dict[str, list] = {}
    for row in dep_df.iter_rows(named=True):
        departures.setdefault(row["stop_id"], []).append(
            {
                "trip_id": row["trip_id"],
                "route_id": row["route_id"],
                "dep_time": row["departure_time"],
                "dest_stop_id": row["dest_stop_id"],
            }
        )

    # ------------------------------------------------------------------
    # trips: full stop sequence, for the trip-itinerary box.
    # ------------------------------------------------------------------
    headsign_by_trip = {}
    if trips_headsign is not None:
        headsign_by_trip = dict(zip(trips_headsign["trip_id"].to_list(), trips_headsign["trip_headsign"].to_list()))

    # Prefer the geometry-resolved `direction_id` from `Shapes.stop_shapes`
    # (see `Shapes.assign_direction_ids`) over `trips.txt`'s own
    # `direction_id` when grouping trips for prev/next-trip navigation: it's
    # reconciled per-route from actual shape bearings (globally consistent
    # 0/1 across all of a route's stops), whereas the raw GTFS column is
    # whatever the agency happened to encode and is sometimes missing or
    # locally inconsistent. Falls back to the GTFS column per-trip when a
    # trip's shape_id has no resolved direction (e.g. no shapes.txt at all).
    shape_dir_map: dict = {}
    shapes_stop_shapes_lf = getattr(feed.shapes, "stop_shapes", None)
    if shapes_stop_shapes_lf is not None:
        shape_dir_df = (
            shapes_stop_shapes_lf.select(["shape_id", "direction_id"])
            .drop_nulls()
            .unique(subset=["shape_id"])
            .collect()
        )
        shape_dir_map = dict(zip(shape_dir_df["shape_id"].to_list(), shape_dir_df["direction_id"].to_list()))

    trips_json = {}
    trip_first_dep = {}
    trip_route_dir = {}
    trip_shape = {}
    for row in day.sort(["trip_id", "stop_sequence"]).iter_rows(named=True):
        tid = row["trip_id"]
        resolved_direction_id = shape_dir_map.get(row["shape_id"], row["direction_id"])
        trip_route_dir[tid] = (row["route_id"], resolved_direction_id)
        if row["shape_id"] is not None:
            trip_shape[tid] = row["shape_id"]
        entry = trips_json.setdefault(
            tid,
            {
                "route_id": row["route_id"],
                "direction_id": resolved_direction_id,
                "shape_id": None,
                "headsign": headsign_by_trip.get(tid),
                "stops": [],
            },
        )
        entry["stops"].append({"stop_id": row["stop_id"], "arr": row["arrival_time"], "dep": row["departure_time"]})
        if trip_first_dep.get(tid) is None:
            trip_first_dep[tid] = row["departure_time"]

    for tid, shid in trip_shape.items():
        trips_json[tid]["shape_id"] = shid

    # Keyed by "route_id||direction_id" (not just route_id) so prev/next-trip
    # navigation in the trip-itinerary box only cycles through trips running
    # the same direction, not the whole route in both directions.
    route_trips: dict[str, list] = {}
    for tid, (rid, did) in trip_route_dir.items():
        key = f"{rid}||{did}"
        route_trips.setdefault(key, []).append(tid)
    for key in route_trips:
        route_trips[key].sort(key=lambda t: (trip_first_dep.get(t) is None, trip_first_dep.get(t) or 0))

    # ------------------------------------------------------------------
    # Speed/headway metrics for the map's mode selector (edges drawn along
    # real shape geometry -- busiest/fastest/shortest shape per stop pair --
    # plus per-stop and per-(stop, route) values for the emoji recoloring
    # and the timetable's Speed/Headway columns).
    # ------------------------------------------------------------------
    metrics = compute_edge_and_stop_metrics(day, stops_df, feed)

    data = {
        "stops": stops_json,
        "routes": routes_json,
        "shapes": shapes_json,
        "shape_route": shape_route,
        "stop_routes": stop_routes,
        "stop_shapes": stop_shapes,
        "departures": departures,
        "trips": trips_json,
        "route_trips": route_trips,
        "route_type_emoji": ROUTE_TYPE_EMOJI,
        "route_type_emoji_fallback": ROUTE_TYPE_EMOJI_FALLBACK,
        "route_type_name": ROUTE_TYPE_NAME,
        "route_type_name_fallback": ROUTE_TYPE_NAME_FALLBACK,
        "square_badge_route_types": sorted(SQUARE_BADGE_ROUTE_TYPES),
        "edges": metrics["edges"],
        "stop_metrics": metrics["stops"],
        "stop_route_metrics": metrics["stop_routes_metrics"],
    }

    # ------------------------------------------------------------------
    # Map assembly
    # ------------------------------------------------------------------
    if m is None:
        mean_lat = stops_df["stop_lat"].mean()
        mean_lon = stops_df["stop_lon"].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=zoom_start, tiles="CartoDB positron")

    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    js_code = _read_static("route_map.js").replace("__MAP_VAR__", m.get_name()).replace("__DATA_JSON__", data_json)
    css_code = _read_static("route_map.css")

    m.get_root().html.add_child(folium.Element(f"<style>{css_code}</style>"))
    m.get_root().script.add_child(folium.Element(js_code))

    return m
