# -*- coding: utf-8 -*-
"""GTFS shapes.txt handling: real polyline geometry, not just straight lines.

What this module does and why:
-------------------------------
`Feed` groups trips into a *synthetic* `shape_id` purely by identical stop
sequence + travel time (see `StopTimes.generate_shape_ids`); this has no
inherent relationship to whatever real `shape_id` a feed's `trips.txt` may
reference, which is what `shapes.txt`'s actual polyline points are keyed by.
`Feed.load_shapes` bridges the two by attaching, to each synthetic group, the
real `shape_id` its member trips actually use (if any/consistent) as a
`real_shape_id` column on `trip_shape_ids_lf`.

Given that, this module:
1.  Reads `shapes.txt` (when present) into `shape_id, shape_pt_sequence,
    shape_pt_lat, shape_pt_lon` -- the actual polyline, not synthesized stop
    coordinates.
2.  For each synthetic shape group whose `real_shape_id` has real polyline
    points, **inserts each of its stops as a vertex** on that polyline at the
    nearest-segment position (computed entirely in polars: a local planar
    projection onto every candidate segment of the same shape, argmin by
    perpendicular distance, no shapely). Groups without a matching real
    shape (missing `shape_id` in trips.txt, no `shapes.txt`, or the
    referenced id isn't in it) fall back to using the stops themselves as a
    straight-line "shape", exactly as before.
3.  Computes `shape_dist_traveled`/`shape_total_distance` directly in
    EPSG:4326 via the haversine formula (`geo_polars.haversine_distance_m`)
    over the combined (real points + inserted stops) sequence -- accurate
    globally, no UTM reprojection, no shapely/geopandas involved in the
    distance math. `shapely`/`geopandas` are only used at the very end, to
    build an optional `GeoDataFrame` (`self.gdf`) of the resulting
    linestrings for callers who want one.
"""

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import polars as pl

from ..utils import gtfs_checker
from ..utils import io
from ..utils import geo_polars

TRIP_ROUND_TIME = 120


def _widest_gap_split(bearings: Dict[str, float]) -> Dict[str, int]:
    """Clusters a stop's per-`shape_id` bearings into two local bins (0/1) by
    finding the widest circular gap between the distinct observed bearings
    and splitting at its midpoint -- the same "widest separation angle" idea
    `analysis/stops.py` uses, but applied to *deduplicated* per-shape
    bearings (one value per `shape_id`) rather than per-departure-instance
    rows. That dedup matters: clustering on the raw per-departure rows makes
    the widest-gap tie-break depend on how many scheduled departures each
    shape happens to have, which is an unrelated scheduling detail, not a
    geometric one, and can flip the result for two routes with identical
    geometry but different headways.

    Returned bin numbers (0 vs 1) are only meaningful *within this one
    stop*; they carry no relationship to numbering at any other stop, which
    is exactly why `_assign_direction_ids_for_route` below has to reconcile
    them across stops rather than using them directly.
    """
    unique_bearings = sorted(set(round(b, 9) for b in bearings.values()))
    if len(unique_bearings) <= 1:
        return {shape_id: 0 for shape_id in bearings}

    n = len(unique_bearings)
    gaps = []
    for i in range(n):
        start = unique_bearings[i]
        end = unique_bearings[(i + 1) % n]
        gap = (end - start) % 360
        if gap == 0:
            gap = 360.0
        gaps.append((gap, start))
    widest_gap, gap_start = max(gaps, key=lambda item: item[0])
    split_angle = (gap_start + widest_gap / 2) % 360

    return {
        shape_id: (0 if ((bearing - split_angle + 360) % 360) < 180 else 1)
        for shape_id, bearing in bearings.items()
    }


def _adjusted_bearing_expr() -> pl.Expr:
    """The same forward/backward-bearing blend `analysis/stops.py`'s
    `shape_direction` clustering uses (rotates the forward bearing towards
    whichever of {forward, backward+180} bisector requires the shorter
    rotation), reproduced here so the two mechanisms treat a straight-ish
    pass-through stop consistently. Read-only: this only feeds the local
    clustering below, and is never written back onto `shape_direction`.

    A shape's very first/last stop has no backward/forward bearing at all
    (there's no previous/next point to aim from) -- `shape_direction`/
    `shape_direction_backwards` is null or NaN there. Both are coalesced
    against the other (offset 180 degrees) *before* blending, otherwise the
    blend formula propagates that null/NaN straight through and corrupts
    the whole stop's clustering (every other shape at that stop too, via
    `_widest_gap_split`'s `sorted()`/comparisons over a NaN)."""
    raw_fwd = pl.col("shape_direction")
    raw_bwd = pl.col("shape_direction_backwards")
    fwd_valid = raw_fwd.is_not_null() & raw_fwd.is_finite()
    bwd_valid = raw_bwd.is_not_null() & raw_bwd.is_finite()

    fwd = pl.when(fwd_valid).then(raw_fwd).otherwise((raw_bwd + 180) % 360)
    bwd = pl.when(bwd_valid).then(raw_bwd).otherwise((raw_fwd + 180) % 360)

    cond = ((fwd + 360 - bwd) % 360) > ((bwd + 360 - fwd) % 360)
    diff = (
        pl.when(cond)
        .then(-1 * (180 - ((bwd + 360 - fwd) % 360)) / 2)
        .otherwise((180 - ((fwd + 360 - bwd) % 360)) / 2)
    )
    diff = pl.when(diff.is_null() | diff.is_nan()).then(pl.lit(0.0)).otherwise(diff)
    return (fwd + 360 + diff) % 360


class Shapes:
    """
    Manage GTFS shapes.txt data using Polars LazyFrames.

    Attributes:
        lf (pl.LazyFrame): One row per point (real shape point or inserted
            stop) per synthetic `shape_id`, ordered by `shape_pt_sequence`,
            with cumulative `shape_dist_traveled`/`shape_total_distance`.
        stop_shapes (pl.LazyFrame): The subset of `lf` rows that are stops
            (i.e. `stop_id` is not null) -- what `Feed.build_lf` joins against.
            After `assign_direction_ids` runs (during `Feed` construction),
            this also carries `route_id`, a resolved `direction_id` (0/1),
            and a boolean `direction_id_issues` flag -- see that method.
        gdf (gpd.GeoDataFrame): One row per `shape_id`, with a LINESTRING (or
            POINT, if degenerate) geometry built from `lf`'s points.
    """
    def __init__(self,lf=None,stop_shapes=None,gdf=None) -> None:
        self.lf = lf
        self.stop_shapes = stop_shapes
        self.gdf = gdf

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]], None],
        trip_shape_ids_lf,
        stops_lf,
        check_files: bool = False,
        min_file_id=0,
    ):
        """Builds `self.lf`/`self.stop_shapes`/`self.gdf`.

        Args:
            path: GTFS directory/directories to look for `shapes.txt` in.
                `None` (or no `shapes.txt` found) means every group falls
                back to the straight-line-between-stops behavior.
            trip_shape_ids_lf: Output of `StopTimes.generate_shape_ids`,
                additionally carrying a `real_shape_id` column (see
                `Feed.load_shapes`) mapping each synthetic `shape_id` to the
                real `shape_id` its trips reference, if any.
            stops_lf: The feed's stops LazyFrame (`stop_id`, `stop_lat`, `stop_lon`).
        """
        raw_shapes_lf = self._read_shapes_file(path, check_files=check_files, min_file_id=min_file_id)

        self.lf = self._generate_shapes_file(stops_lf, trip_shape_ids_lf, raw_shapes_lf)
        self.lf = self.lf.collect().lazy()
        self.stop_shapes = self.lf.filter(pl.col("stop_id").is_not_null())
        self.stop_shapes = self._generate_shape_direction_column(self.stop_shapes)
        self.stop_shapes = self.stop_shapes.collect().lazy()
        self.gdf = self._get_shapes_gdf(self.lf)

    def _read_shapes_file(
        self, path, check_files: bool = False, min_file_id=0
    ) -> Optional[pl.LazyFrame]:
        """Reads `shapes.txt` (if present) into a lazy `shape_id,
        shape_pt_sequence, shape_pt_lat, shape_pt_lon,
        shape_dist_traveled_orig` frame, or None if unavailable/empty.
        """
        if path is None:
            return None

        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]

        shape_paths = []
        for p in paths:
            new_p = io.search_file(p, file="shapes.txt")
            if new_p is not None:
                shape_paths.append(new_p)
        if not shape_paths:
            return None

        schema_dict, _ = gtfs_checker.get_df_schema_dict("shapes.txt")
        shapes = io.read_csv_list(
            shape_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id
        )
        if shapes is None:
            return None

        shapes = shapes.with_columns(
            pl.col("shape_pt_lat").cast(pl.Float64, strict=False),
            pl.col("shape_pt_lon").cast(pl.Float64, strict=False),
            pl.col("shape_pt_sequence").cast(pl.Float64, strict=False),
        ).drop_nulls(["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]).filter(
            pl.col("shape_pt_lat").is_finite() & pl.col("shape_pt_lon").is_finite()
        )

        if shapes.select(pl.len()).collect().item() == 0:
            return None

        columns = shapes.collect_schema().names()
        if "shape_dist_traveled" in columns:
            shapes = shapes.rename({"shape_dist_traveled": "shape_dist_traveled_orig"}).with_columns(
                pl.col("shape_dist_traveled_orig").cast(pl.Float64, strict=False)
            )
        else:
            shapes = shapes.with_columns(pl.lit(None, dtype=pl.Float64).alias("shape_dist_traveled_orig"))

        return shapes.select(
            "shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon", "shape_dist_traveled_orig"
        )

    def _generate_shape_direction_column(self, stop_shapes):
        """
        Generate a new column 'shape_direction' in the stop_shapes LazyFrame,
        representing the approximate direction of shape segments in degrees,
        rounded to the nearest multiple of `round` degrees (default 10º).

        The direction is calculated based on the angle between the current
        stop point and the mean position of all other stops on the same shape,
        effectively estimating the direction of travel at each stop.

        Args:
            stop_shapes (pl.LazyFrame): A Polars LazyFrame containing shape points with
                columns including 'shape_id', 'stop_sequence', 'shape_pt_lat', 'shape_pt_lon'.
            round (int, optional): The degree rounding factor to quantize directions.
                Defaults to 10.

        Returns:
            pl.LazyFrame: The input LazyFrame with an additional column 'shape_direction',
            which is the rounded direction in degrees [0, 360).
        """
        deg2rad = math.pi / 180  # conversion factor degrees to radians
        rad2deg = 180 / math.pi  # conversion factor radians to degrees

        # Calculate cumulative mean latitude and longitude including the current stop.
        # cum_sum / n_stops gives running average for each shape_id group.

        stop_shapes = stop_shapes.with_columns(
                pl.col("shape_pt_lat").cast(float,strict=False),
                pl.col("shape_pt_lon").cast(float,strict=False),
                pl.col("stop_sequence").cast(int,strict=False),
        ).drop_nulls(["shape_pt_lat","shape_pt_lon", "stop_sequence"]).filter(
            pl.col("shape_pt_lat").is_finite() &
            pl.col("shape_pt_lon").is_finite()
        )

        stop_shapes = (
            stop_shapes.sort(["shape_id", "stop_sequence"], descending=True)
            .with_columns(
                [
                    (pl.col("shape_pt_lat").cum_sum().over("shape_id")).alias(
                        "mean_lat"
                    ),
                    (pl.col("shape_pt_lon").cum_sum().over("shape_id")).alias(
                        "mean_lon"
                    ),
                    (
                        pl.col("stop_sequence").max().over("shape_id")
                        - pl.col("stop_sequence")
                    ).alias("stop_sequence_rev"),
                ]
            )
            .collect()
            .lazy()
            .sort(["shape_id", "stop_sequence_rev"], descending=True)
            .with_columns(
                [
                    (pl.col("shape_pt_lat").cum_sum().over("shape_id")).alias(
                        "mean_lat_rev"
                    ),
                    (pl.col("shape_pt_lon").cum_sum().over("shape_id")).alias(
                        "mean_lon_rev"
                    ),
                ]
            )
        )

        # Recalculate mean_lat and mean_lon to exclude the current point.
        # Formula: ((n * mean) - current_value) / (n - 1)
        stop_shapes = stop_shapes.with_columns(
            [
                (
                    (pl.col("mean_lat") - pl.col("shape_pt_lat"))
                    / (pl.col("stop_sequence_rev"))
                ).alias("mean_lat"),
                (
                    (pl.col("mean_lon") - pl.col("shape_pt_lon"))
                    / (pl.col("stop_sequence_rev"))
                ).alias("mean_lon"),
                (
                    (pl.col("mean_lat_rev") - pl.col("shape_pt_lat"))
                    / (pl.col("stop_sequence"))
                ).alias("mean_lat_rev"),
                (
                    (pl.col("mean_lon_rev") - pl.col("shape_pt_lon"))
                    / (pl.col("stop_sequence"))
                ).alias("mean_lon_rev"),
            ]
        )

        # Calculate the angle in degrees from north to the vector from current point to the mean of others.
        # Using spherical trigonometry (arctan2 formula adapted for lat/lon).
        # The angle is normalized to [0, 360) degrees and rounded to nearest multiple of `round`.
        stop_shapes = (
            stop_shapes.with_columns(
                [
                    # radians
                    ((pl.col("mean_lon") - pl.col("shape_pt_lon")) * deg2rad).alias(
                        "dlon_rad"
                    ),
                    (pl.col("shape_pt_lat") * deg2rad).alias("lat1_rad"),
                    (pl.col("mean_lat") * deg2rad).alias("lat2_rad"),
                    ((pl.col("mean_lon_rev") - pl.col("shape_pt_lon")) * deg2rad).alias(
                        "dlon_rad_rev"
                    ),
                    (pl.col("mean_lat_rev") * deg2rad).alias("lat2_rad_rev"),
                ]
            )
            .with_columns(
                [
                    # calculate y and x
                    (pl.col("dlon_rad").sin() * pl.col("lat2_rad").cos()).alias("y"),
                    (
                        pl.col("lat1_rad").cos() * pl.col("lat2_rad").sin()
                        - pl.col("lat1_rad").sin()
                        * pl.col("lat2_rad").cos()
                        * pl.col("dlon_rad").cos()
                    ).alias("x"),
                    (pl.col("dlon_rad_rev").sin() * pl.col("lat2_rad_rev").cos()).alias(
                        "y_rev"
                    ),
                    (
                        pl.col("lat1_rad").cos() * pl.col("lat2_rad_rev").sin()
                        - pl.col("lat1_rad").sin()
                        * pl.col("lat2_rad_rev").cos()
                        * pl.col("dlon_rad_rev").cos()
                    ).alias("x_rev"),
                ]
            )
            .with_columns(
                [
                    # angle and direction
                    (
                        (rad2deg * pl.arctan2(pl.col("y"), pl.col("x")) + 360) % 360
                    ).alias("shape_direction"),
                    (
                        (rad2deg * pl.arctan2(pl.col("y_rev"), pl.col("x_rev")) + 360)
                        % 360
                    ).alias("shape_direction_backwards"),
                ]
            )
            .drop(
                "mean_lat",
                "mean_lon",
                "dlon_rad",
                "lat1_rad",
                "lat2_rad",
                "y",
                "x",
                "mean_lat_rev",
                "mean_lon_rev",
                "dlon_rad_rev",
                "lat2_rad_rev",
                "y_rev",
                "x_rev",
            )
        )

        return stop_shapes

    def _generate_shapes_file(self, stops_lf, trip_shape_ids_lf, raw_shapes_lf: Optional[pl.LazyFrame]):
        """Builds the combined per-`shape_id` point sequence (real shape
        points + inserted stops where real geometry is available, or just
        the stops themselves as a straight line otherwise)."""
        trip_shape_ids_lf = (
            trip_shape_ids_lf.select(["shape_id", "real_shape_id", "stop_ids", "stop_sequence"])
            .explode(["stop_ids", "stop_sequence"])
            .rename({"stop_ids": "stop_id"})
        )

        trip_shape_ids_lf = trip_shape_ids_lf.join(
            stops_lf.select("stop_id", "stop_lat", "stop_lon"),
            on=["stop_id"],
            how="left",
        ).with_columns(
            pl.col("stop_lat").cast(pl.Float64, strict=False),
            pl.col("stop_lon").cast(pl.Float64, strict=False),
            pl.col("stop_sequence").cast(pl.Int64, strict=False),
        ).drop_nulls(["stop_lat", "stop_lon", "stop_sequence"])

        # Split synthetic shape groups into those with usable real geometry
        # (their `real_shape_id` exists in `shapes.txt`, with >=2 points) and
        # those that must fall back to a straight line between stops.
        if raw_shapes_lf is not None:
            real_shape_point_counts = raw_shapes_lf.group_by("shape_id").agg(pl.len().alias("n_points"))
            groups_with_real_shape = (
                trip_shape_ids_lf.select("shape_id", "real_shape_id")
                .unique()
                .filter(pl.col("real_shape_id").is_not_null())
                .join(
                    real_shape_point_counts.rename({"shape_id": "real_shape_id"}),
                    on="real_shape_id",
                    how="inner",
                )
                .filter(pl.col("n_points") >= 2)
                .select("shape_id", "real_shape_id")
                .collect()
            )
        else:
            groups_with_real_shape = pl.DataFrame(schema={"shape_id": pl.Utf8, "real_shape_id": pl.Utf8})

        real_geometry_shape_ids = set(groups_with_real_shape["shape_id"].to_list())

        fallback_stops = trip_shape_ids_lf.filter(~pl.col("shape_id").is_in(list(real_geometry_shape_ids)))
        fallback_points = self._stops_as_straight_line_shape(fallback_stops)

        if not real_geometry_shape_ids:
            return self._generate_shape_dist_traveled_column(fallback_points)

        real_geometry_stops = trip_shape_ids_lf.join(
            groups_with_real_shape.lazy(), on=["shape_id", "real_shape_id"], how="inner"
        )
        inserted_stop_points = self._insert_stops_into_real_shapes(
            real_geometry_stops, raw_shapes_lf, groups_with_real_shape
        )

        # The real shape's own vertices must be kept too (not just the
        # inserted stops) -- otherwise the polyline collapses to however few
        # stops there are, losing exactly the real-geometry detail this is
        # meant to preserve.
        real_shape_vertices = (
            raw_shapes_lf.join(
                groups_with_real_shape.lazy().rename({"real_shape_id": "shape_id", "shape_id": "synthetic_shape_id"}),
                on="shape_id",
                how="inner",
            )
            .select(
                pl.col("synthetic_shape_id").alias("shape_id"),
                pl.lit(None, dtype=pl.Utf8).alias("stop_id"),
                pl.lit(None, dtype=pl.Int64).alias("stop_sequence"),
                "shape_pt_sequence",
                "shape_pt_lat",
                "shape_pt_lon",
            )
        )

        real_geometry_points = pl.concat(
            [real_shape_vertices, inserted_stop_points], how="diagonal_relaxed"
        )

        combined = pl.concat(
            [fallback_points, real_geometry_points], how="diagonal_relaxed"
        )
        return self._generate_shape_dist_traveled_column(combined)

    def _stops_as_straight_line_shape(self, stops: pl.LazyFrame) -> pl.LazyFrame:
        """The pre-refactor fallback: each stop becomes a shape point,
        connected in stop_sequence order (a straight line between stops)."""
        return (
            stops.select(["shape_id", "stop_id", "stop_sequence", "stop_lat", "stop_lon"])
            .rename({"stop_lat": "shape_pt_lat", "stop_lon": "shape_pt_lon"})
            .with_columns(
                pl.col("stop_sequence").alias("shape_pt_sequence"),
            )
        )

    def _insert_stops_into_real_shapes(
        self,
        stops: pl.LazyFrame,
        raw_shapes_lf: pl.LazyFrame,
        groups_with_real_shape: pl.DataFrame,
    ) -> pl.LazyFrame:
        """For each synthetic `shape_id` with real polyline points, inserts
        its stops as vertices at their nearest-segment position along that
        polyline -- entirely in polars, no shapely.

        How it works: the real shape's points define consecutive segments
        (`shift(-1)` per `real_shape_id`, ordered by `shape_pt_sequence`).
        Each stop is joined against every segment of its shape (bounded --
        typically dozens of points per route, not a feed-wide cross join)
        and, using a local planar approximation (equirectangular projection
        centered on each segment), we compute the projection scalar `t`
        (clamped to `[0, 1]`) and perpendicular offset; the nearest segment
        per stop is the row with minimum perpendicular offset. The stop's
        position within the merged sequence is then `segment_index + t`
        (a fractional key sorting it correctly between the segment's two
        endpoints, and correctly relative to other stops on the same
        segment), and its along-shape distance is `segment_start_distance +
        t * segment_length` (using the exact haversine segment length, not
        the planar approximation, which is only used to rank candidate
        segments).
        """
        real_shape_ids = groups_with_real_shape["real_shape_id"].unique().to_list()
        shape_points = (
            raw_shapes_lf.filter(pl.col("shape_id").is_in(real_shape_ids))
            .sort(["shape_id", "shape_pt_sequence"])
            .rename({"shape_id": "real_shape_id"})
        )

        segments = shape_points.with_columns(
            pl.col("shape_pt_lat").alias("seg_end_lat"),
            pl.col("shape_pt_lon").alias("seg_end_lon"),
            pl.col("shape_pt_sequence").alias("seg_end_sequence"),
        ).with_columns(
            pl.col("shape_pt_lat").shift(1).over("real_shape_id").alias("seg_start_lat"),
            pl.col("shape_pt_lon").shift(1).over("real_shape_id").alias("seg_start_lon"),
            pl.col("shape_pt_sequence").shift(1).over("real_shape_id").alias("seg_start_sequence"),
        ).drop_nulls(["seg_start_lat", "seg_start_lon"])

        segments = segments.with_columns(
            geo_polars.haversine_distance_m(
                "seg_start_lat", "seg_start_lon", "seg_end_lat", "seg_end_lon"
            ).alias("seg_length_m")
        ).with_row_index("segment_id").select(
            "real_shape_id", "segment_id", "seg_start_lat", "seg_start_lon",
            "seg_end_lat", "seg_end_lon", "seg_start_sequence", "seg_length_m",
        )

        stops_with_segments = stops.select(
            "shape_id", "real_shape_id", "stop_id", "stop_sequence", "stop_lat", "stop_lon"
        ).join(segments, on="real_shape_id", how="inner")

        # Local planar (equirectangular) projection of the stop onto each
        # candidate segment -- only used to pick the *nearest* segment; the
        # final along-shape distance uses the exact haversine segment length.
        cos_lat = (pl.col("seg_start_lat") * (math.pi / 180)).cos()
        dx = (pl.col("seg_end_lon") - pl.col("seg_start_lon")) * cos_lat
        dy = pl.col("seg_end_lat") - pl.col("seg_start_lat")
        px = (pl.col("stop_lon") - pl.col("seg_start_lon")) * cos_lat
        py = pl.col("stop_lat") - pl.col("seg_start_lat")
        seg_len_sq = (dx ** 2 + dy ** 2)
        raw_t = pl.when(seg_len_sq > 0).then((px * dx + py * dy) / seg_len_sq).otherwise(0.0)
        t = raw_t.clip(0.0, 1.0)
        perp_x = px - t * dx
        perp_y = py - t * dy
        perp_dist_deg = (perp_x ** 2 + perp_y ** 2).sqrt()

        stops_with_segments = stops_with_segments.with_columns(
            t.alias("t"), perp_dist_deg.alias("perp_dist_deg")
        )

        best_segment = (
            stops_with_segments.sort("perp_dist_deg")
            .group_by(["shape_id", "stop_id", "stop_sequence"])
            .agg(pl.all().first())
        )

        best_segment = best_segment.with_columns(
            (pl.col("seg_start_sequence") + pl.col("t")).alias("shape_pt_sequence"),
            (pl.col("t") * pl.col("seg_length_m")).alias("dist_from_segment_start"),
            (
                pl.col("seg_start_lat") + pl.col("t") * (pl.col("seg_end_lat") - pl.col("seg_start_lat"))
            ).alias("shape_pt_lat"),
            (
                pl.col("seg_start_lon") + pl.col("t") * (pl.col("seg_end_lon") - pl.col("seg_start_lon"))
            ).alias("shape_pt_lon"),
        )

        return best_segment.select(
            "shape_id", "stop_id", "stop_sequence", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"
        )

    def _generate_shape_dist_traveled_column(self, shapes):
        shapes = shapes.with_columns(
                pl.col("shape_pt_lat").cast(float,strict=False),
                pl.col("shape_pt_lon").cast(float,strict=False),
        ).drop_nulls(["shape_pt_lat","shape_pt_lon"]).filter(
            pl.col("shape_pt_lat").is_finite() &
            pl.col("shape_pt_lon").is_finite()
        )

        # Cumulative haversine distance per shape_id, in EPSG:4326 directly
        # (no UTM reprojection) -- accurate globally regardless of how large
        # an area a feed's shapes span.
        shapes = shapes.sort("shape_id", "shape_pt_sequence").with_columns(
            geo_polars.haversine_distance_m(
                pl.col("shape_pt_lat").shift(1).over("shape_id"),
                pl.col("shape_pt_lon").shift(1).over("shape_id"),
                "shape_pt_lat",
                "shape_pt_lon",
            ).fill_null(0.0).alias("dist_from_prev")
        )

        shapes = shapes.with_columns(
            pl.col("dist_from_prev")
            .cum_sum()
            .over("shape_id")
            .alias("shape_dist_traveled")
        )

        total_distances = shapes.group_by("shape_id").agg(
            pl.col("dist_from_prev").sum().alias("shape_total_distance")
        )

        shapes = shapes.join(total_distances, on="shape_id", how="left").drop(
            ["dist_from_prev"]
        )

        return shapes

    def _get_shapes_gdf(self, shapes: pl.LazyFrame) -> gpd.GeoDataFrame:
        """
        Convert GTFS shapes.txt into a GeoDataFrame of LINESTRING geometries.
        """
        shapes = shapes.with_columns(
                pl.col("shape_pt_lat").cast(float,strict=False),
                pl.col("shape_pt_lon").cast(float,strict=False),
                pl.col("shape_pt_sequence").cast(float,strict=False),
        ).drop_nulls(["shape_pt_lat","shape_pt_lon","shape_pt_sequence"]).filter(
            pl.col("shape_pt_lat").is_finite() &
            pl.col("shape_pt_lon").is_finite()
        )

        grouped = (
            shapes.sort(["shape_id", "shape_pt_sequence"])
            .with_columns(
                (
                    pl.col("shape_pt_lon").cast(pl.Utf8)
                    + " "
                    + pl.col("shape_pt_lat").cast(pl.Utf8)
                ).alias("pt")
            )
            .group_by("shape_id")
            .agg(pl.col("pt").sort_by("shape_pt_sequence").alias("pt"))
            .filter(pl.col("pt").list.len() >= 1)
            .with_columns(
                (
                    pl.when(pl.col("pt").list.len() == 1)
                    .then(
                        pl.concat_str(
                            [pl.lit("POINT("), pl.col("pt").list.join(""), pl.lit(")")]
                        )
                    )
                    .otherwise(
                        pl.concat_str(
                            [
                                pl.lit("LINESTRING("),
                                pl.col("pt").list.join(", "),
                                pl.lit(")"),
                            ]
                        )
                    )
                ).alias("wkt")
            )
        )

        df = grouped.collect().to_pandas()
        df["geometry"] = gpd.GeoSeries.from_wkt(df["wkt"])
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def assign_direction_ids(self, trip_shape_ids_lf: pl.LazyFrame, trips_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Integrates a globally-reconciled `direction_id` (0/1) directly
        into `self.stop_shapes`, alongside a `route_id` and a boolean
        `direction_id_issues` flag, on top of the raw `shape_direction`/
        `shape_direction_backwards` bearings already there.

        Per-stop local clustering (`_widest_gap_split`) only ever produces
        labels that are meaningful *at that one stop*; a route whose shapes
        branch (e.g. one trip pattern continuing straight, another turning
        off) can trivially get inconsistent 0/1 numbering between stops if
        each stop is labelled independently. This reconciles that, per
        `route_id`, by:

        1. Picking the stop with the most distinct `shape_id`s as the
           anchor -- the stop with the most corroborating shapes is the most
           reliable reference for what "0" and "1" mean for this route.
        2. Walking every other stop in ascending order of distinct
           `shape_id` count (fewest first), so well-supported stops resolve
           before sparser, branch-only ones.
        3. At each stop, choosing whichever of {keep the local 0/1 labels
           as-is, flip them} agrees with more already-resolved shape_ids
           there -- then assigning newly-seen shape_ids their (possibly
           flipped) local label as their first, canonical `direction_id`.
        4. Any already-resolved shape_id that disagrees with the chosen
           orientation at this stop is a genuine conflict: its row at this
           one stop is forced to the majority's value (not its own
           previously-established one) and `direction_id_issues` is set to
           `True` there. This is the only way a single `shape_id` ends up
           with rows of both `direction_id` values -- a real branching
           ambiguity, not a bug -- so it's also what the final feed-wide
           warning counts.

        A `shape_id` shared by more than one `route_id` (uncommon, but
        possible) gets one row per `(shape_id, stop_id, route_id)` here,
        since direction is inherently a per-route concept -- so
        `stop_shapes` may end up with more rows than before for such
        shapes.
        """
        shape_route_map = (
            trip_shape_ids_lf.select(["shape_id", "trip_ids"])
            .explode("trip_ids")
            .rename({"trip_ids": "trip_id"})
            .join(trips_lf.select(["trip_id", "route_id"]), on="trip_id", how="left")
            .select(["shape_id", "route_id"])
            .unique()
            .drop_nulls()
            .collect()
        )

        stop_shapes_df = self.stop_shapes.collect()

        base = stop_shapes_df.select(
            ["shape_id", "stop_id", "stop_sequence", "shape_direction", "shape_direction_backwards"]
        )

        if stop_shapes_df.height == 0 or shape_route_map.height == 0:
            self.stop_shapes = stop_shapes_df.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("route_id"),
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_id_issues"),
            ).lazy()
            return self.stop_shapes

        base = base.join(shape_route_map, on="shape_id", how="inner")
        if base.height == 0:
            self.stop_shapes = stop_shapes_df.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("route_id"),
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_id_issues"),
            ).lazy()
            return self.stop_shapes

        base = base.with_columns(_adjusted_bearing_expr().alias("_adjusted_bearing"))

        result_frames = [
            self._assign_direction_ids_for_route(route_df)
            for _, route_df in base.group_by("route_id", maintain_order=True)
        ]
        directions = pl.concat(result_frames, how="vertical_relaxed").drop("_adjusted_bearing")
        self._warn_about_direction_id_issues(directions)

        # Left join: shapes that matched no route (shouldn't normally
        # happen, but e.g. a trip with no route_id) keep their original
        # single row, with nulls/False for the new columns.
        self.stop_shapes = (
            stop_shapes_df.join(
                directions.select(
                    ["shape_id", "stop_id", "stop_sequence", "route_id", "direction_id", "direction_id_issues"]
                ),
                on=["shape_id", "stop_id", "stop_sequence"],
                how="left",
            )
            .with_columns(pl.col("direction_id_issues").fill_null(False))
            .lazy()
        )
        return self.stop_shapes

    def _assign_direction_ids_for_route(self, route_df: pl.DataFrame) -> pl.DataFrame:
        """Implements the anchor + ascending-support-order reconciliation
        described in `assign_direction_ids`, for the shapes of a single
        `route_id`."""
        usable = route_df.filter(
            pl.col("_adjusted_bearing").is_not_null() & pl.col("_adjusted_bearing").is_finite()
        )

        bearings_by_stop: Dict[str, Dict[str, float]] = {}
        for stop_id, sub in usable.group_by("stop_id", maintain_order=True):
            stop_id = stop_id[0] if isinstance(stop_id, tuple) else stop_id
            bearings_by_stop[stop_id] = dict(zip(sub["shape_id"], sub["_adjusted_bearing"]))

        if not bearings_by_stop:
            # No stop in this route had a computable bearing (e.g. every
            # shape here is a single, degenerate point) -- nothing to assign.
            return route_df.with_columns(
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_id_issues"),
            )

        local_bins = {stop_id: _widest_gap_split(bearings) for stop_id, bearings in bearings_by_stop.items()}
        n_shapes_by_stop = {stop_id: len(bearings) for stop_id, bearings in bearings_by_stop.items()}

        anchor_stop = max(n_shapes_by_stop, key=lambda s: (n_shapes_by_stop[s], s))
        other_stops = sorted(
            (s for s in bearings_by_stop if s != anchor_stop),
            key=lambda s: (n_shapes_by_stop[s], s),
        )

        canonical: Dict[str, int] = dict(local_bins[anchor_stop])
        row_direction: Dict[tuple, int] = {
            (anchor_stop, shape_id): direction_id for shape_id, direction_id in canonical.items()
        }
        issue_rows: set = set()  # {(stop_id, shape_id)} that lost a conflict

        for stop_id in other_stops:
            bins = local_bins[stop_id]
            known = {shape_id: canonical[shape_id] for shape_id in bins if shape_id in canonical}

            if known:
                agree_identity = sum(1 for shape_id, gid in known.items() if bins[shape_id] == gid)
                agree_flip = sum(1 for shape_id, gid in known.items() if (1 - bins[shape_id]) == gid)
                flip = agree_flip > agree_identity
            else:
                flip = False

            oriented = {shape_id: (1 - b if flip else b) for shape_id, b in bins.items()}

            for shape_id, oriented_bin in oriented.items():
                if shape_id in canonical:
                    if oriented_bin == canonical[shape_id]:
                        row_direction[(stop_id, shape_id)] = canonical[shape_id]
                    else:
                        # Genuine conflict: this stop's majority orientation
                        # disagrees with the shape's own established value.
                        # The row is forced to the majority (not the shape's
                        # own preference), and flagged.
                        row_direction[(stop_id, shape_id)] = oriented_bin
                        issue_rows.add((stop_id, shape_id))
                else:
                    canonical[shape_id] = oriented_bin
                    row_direction[(stop_id, shape_id)] = oriented_bin

        direction_id_col = []
        issues_col = []
        for row in route_df.iter_rows(named=True):
            key = (row["stop_id"], row["shape_id"])
            direction_id_col.append(row_direction.get(key))
            issues_col.append(key in issue_rows)

        return route_df.with_columns(
            pl.Series("direction_id", direction_id_col, dtype=pl.Int32),
            pl.Series("direction_id_issues", issues_col, dtype=pl.Boolean),
        )

    def _warn_about_direction_id_issues(self, directions: pl.DataFrame) -> None:
        """Feed-wide summary warning: how many distinct shape_ids ended up
        with rows of *both* direction_id values somewhere in their own stop
        sequence (a genuine branching ambiguity `assign_direction_ids`
        couldn't fully resolve), and at how many distinct stops that
        happened -- both as absolute counts and as a percentage of all
        shape_ids/stop_ids assign_direction_ids actually processed."""
        if directions.height == 0:
            return

        total_shape_ids = directions["shape_id"].n_unique()
        total_stop_ids = directions["stop_id"].n_unique()

        issue_rows = directions.filter(pl.col("direction_id_issues"))
        if issue_rows.height == 0:
            return

        n_stops_with_issues = issue_rows["stop_id"].n_unique()
        n_shape_ids_with_issues = issue_rows["shape_id"].n_unique()
        shape_pct = (100.0 * n_shape_ids_with_issues / total_shape_ids) if total_shape_ids else 0.0
        stop_pct = (100.0 * n_stops_with_issues / total_stop_ids) if total_stop_ids else 0.0

        warnings.warn(
            f"direction_id assignment: {n_shape_ids_with_issues} of {total_shape_ids} "
            f"shape_ids ({shape_pct:.1f}%) had unresolved direction conflicts "
            f"(both direction_id values in their own stop sequence), across "
            f"{n_stops_with_issues} of {total_stop_ids} stop(s) ({stop_pct:.1f}%).",
            RuntimeWarning,
        )
