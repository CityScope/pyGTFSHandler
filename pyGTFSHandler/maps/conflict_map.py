# -*- coding: utf-8 -*-
"""Interactive `direction_id` conflict inspector map.

A debugging companion to `route_map`, not a general-purpose feature: stops
are grouped by `parent_station` (falling back to `stop_id` itself when a
stop has none), same as `route_map` does -- a station's separate platform
`stop_id`s are where GTFS conflicts are computed, but a rider (and this
map's marker/panel) thinks in terms of one station, not each individual
platform. Every station with at least one `direction_id_issues=True` row
(see `Shapes.assign_direction_ids` in `models/shapes.py`) at any of its
platforms is drawn as a red marker, every other station as a small gray
dot. Clicking a red station opens a panel listing the `route_id`s that
conflict there (pooled across all of that station's platforms); clicking a
second panel listing that route's conflicting `shape_id`s at that stop
alongside every non-conflicting `shape_id` of that route at that stop,
grouped by `direction_id` (0 and/or 1) and ordered longest-to-shortest (by
`shape_total_distance`). Selecting a direction and stepping a slider
through its shape_ids draws exactly one at a time on the map in blue,
numbered 1, 2, 3, ... in visit order; clicking any conflicting `shape_id`
in the list draws that shape's own stop sequence the same way, in red, on
top of it -- so the two can be visually compared stop-by-stop. Clicking any
individual numbered stop marker (in either sequence) pops up that
shape/stop's `direction_id`, the local widest-gap split's two angle ranges
(see `models.shapes._split_angle`), and both the raw and forward/backward-
forced-180 (`models.shapes._reconcile_fwd_bwd`) bearings -- the same inputs
`assign_direction_ids` itself used to make that call, for auditing exactly
why a given stop got the direction it did.

Same JSON-blob + hand-written JS/CSS technique as `route_map` (see that
module's docstring for why), split into its own small map rather than added
as a mode of `route_map` since it answers a completely different question
(is the geometry-derived direction assignment self-consistent?) with a
completely different interaction model (no timetables/trip itineraries).
"""

from __future__ import annotations

import json
from pathlib import Path

import folium
import polars as pl

from ..models.shapes import _reconcile_fwd_bwd, _split_angle


def _read_static(name: str) -> str:
    return (Path(__file__).parent / "static" / name).read_text(encoding="utf-8")


def conflict_map(feed, zoom_start: int = 12) -> folium.Map:
    """Builds the `direction_id` conflict inspector map for `feed`.

    Args:
        feed: A `Feed` whose `feed.shapes.stop_shapes` has already been
            through `assign_direction_ids` (this happens automatically
            during `Feed` construction whenever `shapes.txt`-derived
            geometry is available).
        zoom_start: Initial Leaflet zoom level.
    """
    stop_shapes_df = (
        feed.shapes.stop_shapes.select(
            [
                "shape_id",
                "stop_id",
                "stop_sequence",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_total_distance",
                "route_id",
                "direction_id",
                "direction_id_issues",
                "shape_direction",
                "shape_direction_backwards",
            ]
        )
        .drop_nulls(["shape_pt_lat", "shape_pt_lon"])
        .collect()
    )

    parent_by_stop = feed.stops.lf.select(["stop_id", "parent_station"]).collect()
    stop_shapes_df = stop_shapes_df.join(parent_by_stop, on="stop_id", how="left").with_columns(
        pl.col("parent_station").fill_null(pl.col("stop_id")).alias("station")
    )

    station_pos = stop_shapes_df.group_by("station").agg(
        pl.col("shape_pt_lat").mean().alias("lat"),
        pl.col("shape_pt_lon").mean().alias("lon"),
    )
    conflict_per_station = stop_shapes_df.group_by("station").agg(
        pl.col("direction_id_issues").any().alias("conflict")
    )
    stations_meta = station_pos.join(conflict_per_station, on="station")

    # The same per-(route_id, station) widest-gap split boundary
    # `_assign_direction_ids_for_route` computes internally, recomputed
    # here from the same forward/backward-forced-180 bearings -- pooled
    # across the station's platform stop_ids, same as the real assignment
    # -- so a clicked stop marker can show *why* it got its direction_id,
    # not just what it is. `None` when there was nothing to split (e.g.
    # every shape here has an identical bearing, or only one shape serves
    # this station).
    split_angle_by_route_station: dict[tuple, float | None] = {}
    for (route_id, station), sub in stop_shapes_df.group_by(["route_id", "station"], maintain_order=True):
        shape_angles = {}
        for shape_id, fwd, bwd in zip(sub["shape_id"], sub["shape_direction"], sub["shape_direction_backwards"]):
            angles = _reconcile_fwd_bwd(fwd, bwd)
            if angles:
                shape_angles[shape_id] = angles
        split_angle_by_route_station[(route_id, station)] = _split_angle(shape_angles) if shape_angles else None

    def _round_or_none(v):
        return None if v is None or (isinstance(v, float) and v != v) else round(float(v), 1)

    def _shape_stops(shape_id: str, route_id: str) -> list[dict]:
        shape_full = stop_shapes_df.filter(
            (pl.col("shape_id") == shape_id) & (pl.col("route_id") == route_id)
        ).sort("stop_sequence")
        out = []
        for r in shape_full.iter_rows(named=True):
            corrected = _reconcile_fwd_bwd(r["shape_direction"], r["shape_direction_backwards"])
            corrected_fwd = corrected[0] if len(corrected) >= 1 else None
            corrected_bwd = corrected[1] if len(corrected) >= 2 else None
            out.append(
                {
                    "stop_id": r["stop_id"],
                    "stop_sequence": r["stop_sequence"],
                    "lat": r["shape_pt_lat"],
                    "lon": r["shape_pt_lon"],
                    "direction_id": r["direction_id"],
                    "split_angle": _round_or_none(split_angle_by_route_station.get((route_id, r["station"]))),
                    "fwd_raw": _round_or_none(r["shape_direction"]),
                    "bwd_raw": _round_or_none(r["shape_direction_backwards"]),
                    "fwd_corrected": _round_or_none(corrected_fwd),
                    "bwd_corrected": _round_or_none(corrected_bwd),
                }
            )
        return out

    conflicts = {}
    for station in stations_meta.filter(pl.col("conflict"))["station"]:
        sub = stop_shapes_df.filter(pl.col("station") == station)
        route_ids = sub.filter(pl.col("direction_id_issues"))["route_id"].unique().to_list()
        route_entries = {}
        for route_id in route_ids:
            route_sub = sub.filter(pl.col("route_id") == route_id)

            conflicting_shape_ids = sorted(
                route_sub.filter(pl.col("direction_id_issues"))["shape_id"].unique().to_list()
            )
            conflicting = {
                shape_id: {"stops": _shape_stops(shape_id, route_id)}
                for shape_id in conflicting_shape_ids
            }

            ok_by_direction = {}
            for dir_value in (0, 1):
                candidates = (
                    route_sub.filter(
                        (~pl.col("direction_id_issues")) & (pl.col("direction_id") == dir_value)
                    )
                    .select(["shape_id", "shape_total_distance"])
                    .unique()
                    .sort("shape_total_distance", descending=True)
                )
                if candidates.height > 0:
                    ok_by_direction[str(dir_value)] = [
                        {
                            "shape_id": r["shape_id"],
                            "length": None if r["shape_total_distance"] is None else round(float(r["shape_total_distance"]), 1),
                            "stops": _shape_stops(r["shape_id"], route_id),
                        }
                        for r in candidates.iter_rows(named=True)
                    ]

            route_entries[route_id] = {"conflicting": conflicting, "ok": ok_by_direction}
        conflicts[station] = route_entries

    data = {
        "stations": [
            {"station": r["station"], "lat": r["lat"], "lon": r["lon"], "conflict": bool(r["conflict"])}
            for r in stations_meta.iter_rows(named=True)
        ],
        "conflicts": conflicts,
    }

    mean_lat = stations_meta["lat"].mean()
    mean_lon = stations_meta["lon"].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=zoom_start, tiles="CartoDB positron")

    panels_html = '<div id="conflict-panel-routes"></div><div id="conflict-panel-shapes"></div>'
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    js_code = (
        _read_static("conflict_map.js")
        .replace("__MAP_VAR__", m.get_name())
        .replace("__DATA_JSON__", data_json)
    )
    css_code = _read_static("conflict_map.css")

    m.get_root().html.add_child(folium.Element(panels_html))
    m.get_root().html.add_child(folium.Element(f"<style>{css_code}</style>"))
    m.get_root().script.add_child(folium.Element(js_code))

    return m
