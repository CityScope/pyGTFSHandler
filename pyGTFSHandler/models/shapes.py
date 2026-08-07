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

Direction assignment pipeline (`direction_id` from geometry alone):
---------------------------------------------------------------------
Separately from the polyline/distance handling above, this module also
derives a `direction_id` (0/1) for every `(shape_id, stop_id)` row in
`stop_shapes`, purely from geometry -- no reliance on `trips.txt`'s own
`direction_id` field, which real-world feeds frequently leave null or set
inconsistently. The pipeline, in the order data flows through it:

1.  `_reconcile_fwd_bwd` -- per `(station, shape_id)`, forces that shape's
    raw incoming/outgoing bearings through the stop to be exactly 180
    degrees apart, so a lone noisy reading is never trusted over the
    other, and a shape's own antipodal pair is always available to anchor
    the far side of the compass circle at that station.
2.  `_widest_gap_split` (via `_split_angle`) -- pools every shape's
    reconciled bearings at one station and clusters them into two local
    bins by finding the widest gap on the compass circle and splitting at
    its midpoint. The resulting bin numbers (0 vs 1) are only meaningful
    *at that one station*; nothing yet ties "0" at one station to "0" at
    another.
3.  `Shapes._assign_direction_ids_for_route` -- reconciles those
    independent per-station local bins into one consistent, route-wide
    0/1 numbering per `shape_id`, anchored at the most-corroborated
    station and walked outward to every other station in ascending order
    of support. Flags any row whose final value still disagrees with that
    same shape's own majority value as `direction_conflict=True` -- a
    genuine, unresolvable branching ambiguity, not a bug. See its
    docstring for the full step-by-step algorithm.
4.  `Shapes.assign_direction_ids` -- the public entry point: maps every
    `shape_id` to its `route_id`, groups `stop_shapes` by `route_id`, runs
    step 3 per route, and writes the resulting `route_id`/`direction_id`/
    `direction_conflict` columns back onto `self.stop_shapes`.
5.  `Shapes._warn_about_direction_conflict` -- after every route has been
    processed, emits one feed-wide `RuntimeWarning` summarizing how many
    distinct shapes/stops ended up with unresolved conflicts, so a caller
    isn't silently handed inconsistent data.
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


def _reconcile_fwd_bwd(fwd: Optional[float], bwd: Optional[float], method: str = "both") -> List[float]:
    """Forces a shape's forward/backward bearings through a stop to be exactly 180 degrees apart.

    `method` controls how the necessary rotation is distributed between
    the two raw readings:

    - `"both"` (default): rotates *both* readings toward each other by
      half the discrepancy, in opposite directions -- e.g. forward=30,
      backward=100 are 70 degrees short of the ideal 180 degrees apart,
      so forward moves -35 and backward moves +35, landing at 355/175.
      Neither raw reading is trusted over the other; each is corrected by
      the same amount.
    - `"forward"`: forward is left exactly as read; backward is fully
      replaced by `forward + 180`. Use when the forward bearing is the
      more trustworthy reading at this stop (e.g. it's a normal
      mid-shape point, while the backward reading comes from a noisier
      or more sparsely-spaced segment).
    - `"backward"`: backward is left exactly as read; forward is fully
      replaced by `backward + 180`. The mirror image of `"forward"`.

    At a shape's very first/last stop, only one of `fwd`/`bwd` exists (no
    previous/next point on that side). Rather than returning that single
    reading alone -- which would deny that stop the antipodal pair every
    other stop gets, reintroducing the near-duplicate-points-always-split
    problem right at what's often the anchor stop, since termini tend to
    have the most corroborating shapes -- the missing side is filled in as
    the existing reading plus 180. (`method` has no effect in this case --
    there's only one raw reading to keep either way.)

    Args:
        fwd: The raw forward bearing in degrees from north, or `None`/`NaN`
            if this stop is the shape's last point (no next point to
            compute a forward bearing from).
        bwd: The raw backward bearing in degrees from north, or
            `None`/`NaN` if this stop is the shape's first point.
        method: `"both"`, `"forward"`, or `"backward"` -- see above.

    Returns:
        A 2-element `[forward, backward]` pair, always exactly 180 degrees
        apart. An empty list if neither `fwd` nor `bwd` was available at
        all (e.g. a shape with only one stop total, so there's no
        previous/next point to compute a bearing from either way).
    """
    if method not in ("both", "forward", "backward"):
        raise ValueError(f"method must be 'both', 'forward', or 'backward', got {method!r}")

    fwd_ok = fwd is not None and not math.isnan(fwd) and math.isfinite(fwd)
    bwd_ok = bwd is not None and not math.isnan(bwd) and math.isfinite(bwd)
    if fwd_ok and bwd_ok:
        if method == "forward":
            return [fwd, (fwd + 180) % 360]
        if method == "backward":
            return [(bwd + 180) % 360, bwd]
        # method == "both": how far `bwd - fwd` is from the ideal 180,
        # folded into (-180, 180].
        excess = ((bwd - fwd - 180 + 540) % 360) - 180
        adjustment = excess / 2
        return [(fwd + adjustment) % 360, (bwd - adjustment) % 360]
    elif fwd_ok:
        return [fwd, (fwd + 180) % 360]
    elif bwd_ok:
        return [(bwd + 180) % 360, bwd]
    else:
        return []


def _split_angle(bearings: Dict[str, List[float]]) -> Optional[float]:
    """Computes the widest-gap split boundary `_widest_gap_split` bisects a stop's bearings at.

    Split out on its own so callers that need the boundary itself (e.g.
    `maps.conflict_map`, to show *why* a stop got the direction_id it did)
    can get it without duplicating this computation.

    Args:
        bearings: `{shape_id: [angles, ...]}` -- every shape's antipodal
            bearing pair(s) observed at one stop, in degrees from north,
            as produced by `_reconcile_fwd_bwd`.

    Returns:
        The split angle in degrees from north, such that bin 0 is the
        half-circle `[split_angle, split_angle + 180)` and bin 1 is
        `[split_angle + 180, split_angle + 360)`. `None` when there's at
        most one distinct bearing across all shapes (nothing to split).
    """
    all_points = [round(angle, 9) for angles in bearings.values() for angle in angles]
    unique_bearings = sorted(set(all_points))
    if len(unique_bearings) <= 1:
        return None

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
    return (gap_start + widest_gap / 2) % 360


def _widest_gap_split(bearings: Dict[str, List[float]]) -> Dict[str, int]:
    """Clusters a stop's per-shape bearings into two local direction bins.

    Finds the widest circular gap between the distinct observed bearings
    and splits at its midpoint -- the same "widest separation angle" idea
    `analysis/stops.py` uses, but applied to *deduplicated* per-shape
    bearings rather than per-departure-instance rows. That dedup matters:
    clustering on the raw per-departure rows makes the widest-gap tie-break
    depend on how many scheduled departures each shape happens to have,
    which is an unrelated scheduling detail, not a geometric one, and can
    flip the result for two routes with identical geometry but different
    headways.

    Each shape contributes the (up to) two forward/backward-forced-180
    points `_reconcile_fwd_bwd` produced for it, not a single value: a
    shape's own antipodal pair fills in the far side of the circle from
    its own bearing, so a widest-gap computation over all shapes at a stop
    always has points on both sides of the circle to work with. That's
    what lets it find the genuine gap between two real direction clusters,
    rather than the (irrelevant) tiny gap between two near-identical
    bearings on the same side.

    Args:
        bearings: `{shape_id: [angles, ...]}` -- every shape's antipodal
            bearing pair(s), as produced by `_reconcile_fwd_bwd`, observed
            at one stop, in degrees from north.

    Returns:
        `{shape_id: 0 or 1}`. Bin numbers are only meaningful *within this
        one stop*; they carry no relationship to numbering at any other
        stop, which is exactly why `Shapes._assign_direction_ids_for_route`
        has to reconcile them across stops rather than using them
        directly.
    """
    split_angle = _split_angle(bearings)
    if split_angle is None:
        return {shape_id: 0 for shape_id in bearings}

    def bin_of(angle: float) -> int:
        return 0 if ((angle - split_angle + 360) % 360) < 180 else 1

    result: Dict[str, int] = {}
    for shape_id, angles in bearings.items():
        bins = [bin_of(a) for a in angles]
        # A shape's own forward/backward points are forced 180 apart, so
        # they always land in different bins by construction -- take the
        # forward (first) one, which is what we actually care about.
        result[shape_id] = bins[0]
    return result


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
            this also carries `route_id`, a resolved `direction_id` (0/1,
            constant across every row of a given `shape_id`), and a boolean
            `direction_conflict` flag marking rows where the geometry
            actually disagreed with that shape's reported direction -- see
            that method.
        gdf (gpd.GeoDataFrame): One row per `shape_id`, with a LINESTRING (or
            POINT, if degenerate) geometry built from `lf`'s points.
    """
    def __init__(
        self,
        lf: Optional[pl.LazyFrame] = None,
        stop_shapes: Optional[pl.LazyFrame] = None,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """Wraps already-loaded shape frames, or leaves them empty for `load` to fill in.

        Args:
            lf: Optional pre-loaded shape-points LazyFrame.
            stop_shapes: Optional pre-computed stop-subset LazyFrame (see
                the class docstring).
            gdf: Optional pre-built per-`shape_id` GeoDataFrame.
        """
        self.lf = lf
        self.stop_shapes = stop_shapes
        self.gdf = gdf

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]], None],
        trip_shape_ids_lf: pl.LazyFrame,
        stops_lf: pl.LazyFrame,
        check_files: bool = False,
        min_file_id: int = 0,
    ) -> None:
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
        Each *distinct* `(real_shape_id, stop_id)` pair is joined against
        every segment of that real shape (bounded -- typically dozens of
        points per route, not a feed-wide cross join) and, using a local
        planar approximation (equirectangular projection centered on each
        segment), we compute the projection scalar `t` (clamped to `[0,
        1]`) and perpendicular offset; the nearest segment per stop is the
        row with minimum perpendicular offset. The stop's position within
        the merged sequence is then `segment_index + t` (a fractional key
        sorting it correctly between the segment's two endpoints, and
        correctly relative to other stops on the same segment), and its
        along-shape distance is `segment_start_distance + t *
        segment_length` (using the exact haversine segment length, not the
        planar approximation, which is only used to rank candidate
        segments).

        Deduplicating to `(real_shape_id, stop_id)` before this join matters:
        many *synthetic* `shape_id`s (one per distinct stop-sequence +
        travel-time bucket, see `StopTimes.generate_shape_ids`) commonly
        share the same underlying `real_shape_id` -- e.g. one bus route
        pattern with many travel-time variants. Joining per synthetic
        `shape_id` instead would repeat the identical stop-vs-segment
        geometry once per variant, multiplying the join size (and memory)
        by however many synthetic shapes share that real shape. The
        nearest-segment result depends only on the stop's own coordinates
        and which real shape it's mapped to, never on the synthetic
        `shape_id` or `stop_sequence`, so it's computed once per distinct
        pair and broadcast back to every synthetic shape/stop_sequence that
        needs it afterward.
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

        all_stop_rows = stops.select(
            "shape_id", "real_shape_id", "stop_id", "stop_sequence", "stop_lat", "stop_lon"
        )
        distinct_stops = all_stop_rows.select(
            "real_shape_id", "stop_id", "stop_lat", "stop_lon"
        ).unique(["real_shape_id", "stop_id"])

        stops_with_segments = distinct_stops.join(segments, on="real_shape_id", how="inner")

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

        # `sort(...).group_by(...).agg(pl.all().first())` alone does not
        # reliably keep that sort order through a lazy, parallel group-by
        # aggregation -- on a stop with two (near-)tied nearest segments,
        # which one "wins" as first() could vary between runs, silently
        # changing the stop's inserted position (and everything downstream
        # of it: bearings, direction_id, direction_conflict) on identical
        # input. Filtering to the true minimum first, then breaking any
        # remaining tie deterministically by `segment_id` via an explicit
        # sort + `maintain_order=True` group_by, removes that dependence on
        # incidental execution order.
        min_perp_dist = pl.col("perp_dist_deg").min().over(["real_shape_id", "stop_id"])
        best_segment = (
            stops_with_segments.filter(pl.col("perp_dist_deg") == min_perp_dist)
            .sort(["real_shape_id", "stop_id", "segment_id"])
            .group_by(["real_shape_id", "stop_id"], maintain_order=True)
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
        ).select("real_shape_id", "stop_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon")

        # Broadcast the once-computed nearest-segment position back out to
        # every synthetic `shape_id`/`stop_sequence` that shares this
        # `(real_shape_id, stop_id)` pair.
        return all_stop_rows.select("shape_id", "real_shape_id", "stop_id", "stop_sequence").join(
            best_segment, on=["real_shape_id", "stop_id"], how="inner"
        ).select(
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

    def assign_direction_ids(
        self, trip_shape_ids_lf: pl.LazyFrame, trips_lf: pl.LazyFrame, stops_lf: Optional[pl.LazyFrame] = None,
        method: str = "both",
    ) -> pl.LazyFrame:
        """Integrates a globally-reconciled `direction_id` into `self.stop_shapes`.

        See the module docstring's "Direction assignment pipeline" section
        for how this fits together with `_assign_direction_ids_for_route`
        and the module-level bearing helpers. In short: adds a `route_id`,
        a resolved `direction_id` (0/1), and a boolean `direction_conflict`
        flag directly onto `self.stop_shapes`, on top of the raw
        `shape_direction`/`shape_direction_backwards` bearings already
        there.

        Bearings are gathered and split per `(route_id, parent_station)`,
        not per raw `stop_id` (falling back to `stop_id` itself when
        `stops_lf` isn't given or a stop has no `parent_station`): a
        station's separate platform `stop_id`s (e.g. paired platforms on
        either side of a road) are where GTFS actually records distinct
        rows, but splitting each platform's handful of shapes independently
        starves the widest-gap computation of the corroborating shapes its
        sibling platform sees, right where it matters most -- multi-route
        interchanges. Pooling by station gives every shape passing through
        it, on any of its platforms, a say in that station's split.

        Per-station local clustering (`_widest_gap_split`) only ever
        produces labels that are meaningful *at that one station*; a route
        whose shapes branch (e.g. one trip pattern continuing straight,
        another turning off) can trivially get inconsistent 0/1 numbering
        between stations if each is labelled independently.
        `_assign_direction_ids_for_route` reconciles that per `route_id` --
        see its docstring for the full algorithm.

        A `shape_id` shared by more than one `route_id` (uncommon, but
        possible) gets one row per `(shape_id, stop_id, route_id)` here,
        since direction is inherently a per-route concept -- so
        `self.stop_shapes` may end up with more rows than before for such
        shapes.

        Args:
            trip_shape_ids_lf: Output of `StopTimes.generate_shape_ids`,
                mapping each synthetic `shape_id` to its member `trip_ids`.
            trips_lf: The feed's trips LazyFrame, at least `trip_id` and
                `route_id` columns.
            stops_lf: The feed's stops LazyFrame (`stop_id`,
                `parent_station`), used to pool bearings by station rather
                than by raw `stop_id`. When omitted, every stop is treated
                as its own station.
            method: `"both"`, `"forward"`, or `"backward"` -- passed
                through to `_reconcile_fwd_bwd` for every bearing pair;
                see its docstring for what each does.

        Returns:
            `self.stop_shapes` with `route_id`, `direction_id`, and
            `direction_conflict` columns added -- also stored back onto
            `self.stop_shapes` as a side effect.
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

        if stops_lf is not None and stop_shapes_df.height > 0:
            parent_by_stop = stops_lf.select(["stop_id", "parent_station"]).collect()
            base = base.join(parent_by_stop, on="stop_id", how="left").with_columns(
                pl.col("parent_station").fill_null(pl.col("stop_id")).alias("station")
            ).drop("parent_station")
        else:
            base = base.with_columns(pl.col("stop_id").alias("station"))

        if stop_shapes_df.height == 0 or shape_route_map.height == 0:
            self.stop_shapes = stop_shapes_df.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("route_id"),
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_conflict"),
            ).lazy()
            return self.stop_shapes

        base = base.join(shape_route_map, on="shape_id", how="inner")
        if base.height == 0:
            self.stop_shapes = stop_shapes_df.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("route_id"),
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_conflict"),
            ).lazy()
            return self.stop_shapes

        result_frames = [
            self._assign_direction_ids_for_route(route_df, method=method)
            for _, route_df in base.group_by("route_id", maintain_order=True)
        ]
        directions = pl.concat(result_frames, how="vertical_relaxed")
        self._warn_about_direction_conflict(directions)

        # Left join: shapes that matched no route (shouldn't normally
        # happen, but e.g. a trip with no route_id) keep their original
        # single row, with nulls/False for the new columns.
        self.stop_shapes = (
            stop_shapes_df.join(
                directions.select(
                    ["shape_id", "stop_id", "stop_sequence", "route_id", "direction_id", "direction_conflict"]
                ),
                on=["shape_id", "stop_id", "stop_sequence"],
                how="left",
            )
            .with_columns(pl.col("direction_conflict").fill_null(False))
            .lazy()
        )
        return self.stop_shapes

    def _assign_direction_ids_for_route(self, route_df: pl.DataFrame, method: str = "both") -> pl.DataFrame:
        """Reconciles per-station local direction bins into one route-wide `direction_id`.

        Implements the anchor + ascending-support-order reconciliation
        described in `assign_direction_ids`, for the shapes of a single
        `route_id`. Grouped by `station` (`parent_station`, falling back to
        `stop_id`) throughout, not raw `stop_id` -- see that method's
        docstring for why.

        Concretely, in order:

        1. **Gather bearings, excluding each shape's own first and last
           station.** For every `(station, shape_id)` pair, `_reconcile_
           fwd_bwd` turns that row's raw `shape_direction`/`shape_
           direction_backwards` into a forward/backward-forced-180
           antipodal pair (see that function). A shape's own *first* and
           *last* station in its sequence are skipped here entirely --
           there is no real "direction of travel" through a point where the
           vehicle starts or terminates, so whichever lone reading exists
           there (only one of forward/backward is even computable at a true
           endpoint) is unreliable signal, and letting it into the pool
           would let a degenerate reading distort that station's split for
           every *other* shape serving it too. `bearings_by_station
           [station][shape_id]` ends up holding every shape's antipodal
           pair at every station it passes *through*, never at one it
           starts or ends at. A station that happens to be *only* ever an
           endpoint (for every shape that serves it -- typically a genuine
           route terminus) naturally ends up with zero or few contributing
           shapes here, which is exactly why step 3 below never needs a
           separate "is this an endpoint" check of its own: a pure terminus
           already can't out-compete a real through-station on shape count.

        2. **Split each station locally.** `_widest_gap_split` clusters
           each station's pooled bearings into two local bins, 0 and 1 --
           meaningful only *at that one station*; nothing ties "0" at one
           station to "0" at another yet.

        3. **Pick an anchor.** Reconciling those independent local bins
           into one consistent, route-wide numbering starts from a single
           reference station: the one with the most distinct `shape_id`s
           surviving step 1's endpoint exclusion -- the most corroborated,
           and therefore most reliable, split.

        4. **Walk every other station in ascending order of distinct
           `shape_id` count** (fewest first), deciding at each one whether
           to keep its local 0/1 labels as-is or flip them -- whichever
           agrees with more already-resolved shape_ids there -- then
           recording that (possibly flipped) label as each shape's
           first-seen, canonical `direction_id`.

        5. **Force disagreements to the local majority.** Any
           already-canonical shape_id whose local orientation at a station
           disagrees with its own established value is overwritten to the
           station majority's value there (not left at its own prior
           value) -- but *not* flagged yet; flagging is deferred to step 8,
           after step 7 has had a chance to revise this row further. This
           is the only way one `shape_id` ends up with rows of more than
           one `direction_id` value.

        6. **Back-fill each shape's excluded first/last station** with the
           most frequent `direction_id` among that shape's own other,
           already-resolved stations -- inherited, not compared against
           anything. (The first station is typically already resolved via
           some *other* shape that merely passes through it; only truly
           falls back to "no data at all" -- staying `None` -- when nothing
           else ever pins that station down either.)

        7. **Refine remaining disagreements by shared-edge corroboration.**
           A station forced to the majority in step 5 might share one of
           its own immediate edges -- the (previous_station, this_station)
           or (this_station, next_station) pair from that shape's own
           sequence -- with a *different* shape_id of this same route that
           wasn't itself forced to the majority there. Since that donor
           traverses the exact same physical edge, its direction_id there
           is direct evidence for what this station's direction_id actually
           is; take it. This can only ever change *which* value a
           still-disagreeing row holds, never whether it's flagged --
           that's decided once, globally, in step 8.

        8. **Collapse each shape to a single `direction_id`, flag by final
           self-consistency.** Only now, with every row's value settled
           (steps 5-7 done revising them), is each shape_id's "true" value
           decided: its majority `direction_id` across *all* of its own
           rows (including the step 7-revised ones). Every row of that
           shape_id is then set to that one majority value -- so a single
           `shape_id` never ends up with rows of more than one
           `direction_id` in the output, whatever steps 5-7 computed along
           the way -- and `direction_conflict` is set `True` on exactly the
           rows where the settled (pre-collapse) value disagreed with that
           majority, i.e. where the geometry genuinely couldn't be
           reconciled into the shape's single reported direction. This is
           deliberately a single, final, holistic pass over each shape's
           own settled values -- not an incremental flag/unflag threaded
           through steps 5 and 7 -- so a row's flag reflects whether it's
           *actually* still inconsistent with that shape's own data,
           regardless of which step last touched its value. A row that
           step 7 revises to agree with the shape's own majority elsewhere
           is correctly never flagged; a row step 7 leaves disagreeing (as
           here, where the donor's value doesn't happen to match the
           shape's own majority) is correctly still flagged -- and is
           still reported with the shape's majority `direction_id`, not
           the disagreeing one, since a single shape can only sensibly
           report one direction. The disagreeing (pre-collapse) value
           itself isn't lost: since `direction_id` is always binary (0/1),
           a flagged row's actual local reading is simply `1 -
           direction_id` -- which is what `maps.conflict_map` shows.

        Args:
            route_df: All `stop_shapes` rows for a single `route_id`, with
                at least `shape_id`, `station`, `stop_sequence`,
                `shape_direction`, and `shape_direction_backwards` columns
                (the `station` column is `parent_station`, falling back to
                `stop_id`, as prepared by `assign_direction_ids`).
            method: `"both"`, `"forward"`, or `"backward"` -- passed
                through to `_reconcile_fwd_bwd` (step 1's raw bearing
                inputs) for every `(station, shape_id)` pair.

        Returns:
            `route_df` with `direction_id` (nullable `Int32`) and
            `direction_conflict` (`Boolean`) columns added, one row per
            input row. `direction_id` is `None` for every row when no
            station in this route had a computable bearing at all (e.g.
            every shape here is a single, degenerate point); otherwise
            it's constant across every row of a given `shape_id` -- see
            step 8.
        """
        # See step 1 above.
        first_seq_by_shape: Dict[str, int] = {}
        last_seq_by_shape: Dict[str, int] = {}
        for shape_id, sub_shape in route_df.group_by("shape_id", maintain_order=True):
            shape_id = shape_id[0] if isinstance(shape_id, tuple) else shape_id
            first_seq_by_shape[shape_id] = sub_shape["stop_sequence"].min()
            last_seq_by_shape[shape_id] = sub_shape["stop_sequence"].max()

        bearings_by_station: Dict[str, Dict[str, List[float]]] = {}
        for station, sub in route_df.group_by("station", maintain_order=True):
            station = station[0] if isinstance(station, tuple) else station
            # Sorted by stop_sequence (rather than left as whatever row
            # order `route_df` -- itself downstream of a hash join with no
            # guaranteed row order -- happens to have) so that a shape
            # visiting this station more than once (e.g. a loop route
            # passing the same stop on the way out and back) resolves the
            # `shape_angles[shape_id] = ...` overwrite below deterministically
            # (last-by-stop_sequence wins), instead of incidentally to
            # upstream join/thread ordering.
            sub = sub.sort("stop_sequence")
            shape_angles: Dict[str, List[float]] = {}
            for shape_id, seq, fwd, bwd in zip(
                sub["shape_id"], sub["stop_sequence"], sub["shape_direction"], sub["shape_direction_backwards"]
            ):
                if seq == first_seq_by_shape.get(shape_id) or seq == last_seq_by_shape.get(shape_id):
                    continue
                angles = _reconcile_fwd_bwd(fwd, bwd, method=method)
                if angles:
                    shape_angles[shape_id] = angles
            if shape_angles:
                bearings_by_station[station] = shape_angles

        if not bearings_by_station:
            # No station in this route had a computable bearing (e.g. every
            # shape here is a single, degenerate point) -- nothing to assign.
            return route_df.with_columns(
                pl.lit(None, dtype=pl.Int32).alias("direction_id"),
                pl.lit(False, dtype=pl.Boolean).alias("direction_conflict"),
            )

        local_bins = {station: _widest_gap_split(bearings) for station, bearings in bearings_by_station.items()}
        n_shapes_by_station = {station: len(bearings) for station, bearings in bearings_by_station.items()}

        anchor_station = max(n_shapes_by_station, key=lambda s: (n_shapes_by_station[s], s))
        other_stations = sorted(
            (s for s in bearings_by_station if s != anchor_station),
            key=lambda s: (n_shapes_by_station[s], s),
        )

        canonical: Dict[str, int] = dict(local_bins[anchor_station])
        row_direction: Dict[tuple, int] = {
            (anchor_station, shape_id): direction_id for shape_id, direction_id in canonical.items()
        }
        # {(station, shape_id)} rows step 5 forced to the local majority
        # against that shape's own established value -- candidates for
        # step 7's donor lookup, and for step 8's final flagging (unless
        # step 7 revises them back into agreement first). Not the final
        # `direction_conflict` output; see step 8.
        forced_rows: set = set()

        for station in other_stations:
            bins = local_bins[station]
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
                        row_direction[(station, shape_id)] = canonical[shape_id]
                    else:
                        # Disagreement with this shape's own established
                        # value: the row is forced to the station majority
                        # (not the shape's own preference). Not flagged
                        # here -- see step 8.
                        row_direction[(station, shape_id)] = oriented_bin
                        forced_rows.add((station, shape_id))
                else:
                    canonical[shape_id] = oriented_bin
                    row_direction[(station, shape_id)] = oriented_bin

        # Back-fill each shape's excluded first/last station (step 6) with
        # the most frequent direction_id among that shape's own other,
        # already-resolved stations -- not flagged as an issue, since it
        # was never compared against anything, just inherited. The first
        # station is filled in before the last one so that, for a 2-station
        # shape, the last station's back-fill can see the first station's
        # freshly-inherited value too (rather than finding no data at all
        # for either).
        endpoint_stations_by_shape: Dict[str, List[str]] = {}
        for shape_id, sub_shape in route_df.group_by("shape_id", maintain_order=True):
            shape_id = shape_id[0] if isinstance(shape_id, tuple) else shape_id
            first_row = sub_shape.filter(pl.col("stop_sequence") == first_seq_by_shape[shape_id])
            last_row = sub_shape.filter(pl.col("stop_sequence") == last_seq_by_shape[shape_id])
            endpoint_stations_by_shape[shape_id] = [first_row["station"][0], last_row["station"][0]]

        for shape_id, (first_station, last_station) in endpoint_stations_by_shape.items():
            for endpoint_station in (first_station, last_station):
                if (endpoint_station, shape_id) in row_direction:
                    continue  # e.g. a single-station shape: first == last
                values = [
                    direction_id
                    for (station, sid), direction_id in row_direction.items()
                    if sid == shape_id
                ]
                if values:
                    mode = 0 if values.count(0) >= values.count(1) else 1
                    row_direction[(endpoint_station, shape_id)] = mode

        # Last resolution pass: a still-flagged station might share one of
        # its own immediate edges -- the (previous_station, this_station)
        # or (this_station, next_station) pair from this shape's own
        # sequence -- with a *different* shape_id of this same route that
        # isn't itself flagged there. Since that donor shape traverses the
        # exact same physical edge, its direction_id at this station is
        # direct evidence for what this station's direction_id actually is
        # -- take it, and un-flag this row.
        shape_station_seq: Dict[str, List[str]] = {}
        for shape_id, sub_shape in route_df.group_by("shape_id", maintain_order=True):
            shape_id = shape_id[0] if isinstance(shape_id, tuple) else shape_id
            shape_station_seq[shape_id] = sub_shape.sort("stop_sequence")["station"].to_list()

        def _adjacent_edges(seq: List[str], station: str) -> set:
            idx = seq.index(station)
            edges = set()
            if idx > 0:
                edges.add((seq[idx - 1], station))
            if idx < len(seq) - 1:
                edges.add((station, seq[idx + 1]))
            return edges

        for station, shape_id in sorted(forced_rows):
            seq = shape_station_seq.get(shape_id)
            if not seq or station not in seq:
                continue
            target_edges = _adjacent_edges(seq, station)
            if not target_edges:
                continue

            donor_direction = None
            # Sorted by shape_id (rather than dict/insertion order, which
            # traces back to `route_df`'s row order -- not guaranteed
            # stable across runs of polars' multi-threaded lazy execution)
            # so that when more than one shape_id could donate here, which
            # one wins is a deterministic, reproducible choice rather than
            # incidental to unrelated upstream ordering.
            for donor_shape_id in sorted(shape_station_seq):
                donor_seq = shape_station_seq[donor_shape_id]
                if donor_shape_id == shape_id:
                    continue
                if (station, donor_shape_id) in forced_rows:
                    continue
                if (station, donor_shape_id) not in row_direction:
                    continue
                if station not in donor_seq:
                    continue
                if target_edges & _adjacent_edges(donor_seq, station):
                    donor_direction = row_direction[(station, donor_shape_id)]
                    break

            if donor_direction is not None:
                row_direction[(station, shape_id)] = donor_direction

        # Step 8 (see docstring): each shape's majority direction_id across
        # all of its own settled rows is its one "true" value -- flag
        # exactly the rows that disagreed with it, then collapse every row
        # of that shape_id to the majority, so `direction_id` is always
        # constant per shape_id in the output.
        values_by_shape: Dict[str, List[int]] = {}
        for (_station, shape_id), direction_id in row_direction.items():
            values_by_shape.setdefault(shape_id, []).append(direction_id)

        majority_by_shape = {
            shape_id: (0 if values.count(0) >= values.count(1) else 1)
            for shape_id, values in values_by_shape.items()
        }

        issue_rows = {
            (station, shape_id)
            for (station, shape_id), direction_id in row_direction.items()
            if direction_id != majority_by_shape[shape_id]
        }

        direction_id_col = []
        issues_col = []
        for row in route_df.iter_rows(named=True):
            key = (row["station"], row["shape_id"])
            if key in row_direction:
                # Report the shape's own majority everywhere, not the
                # (possibly disagreeing) per-row value steps 5-7 settled
                # on -- see step 8's docstring for why, and for how to
                # recover the disagreeing reading at a flagged row.
                direction_id_col.append(majority_by_shape[row["shape_id"]])
            else:
                direction_id_col.append(None)
            issues_col.append(key in issue_rows)

        return route_df.with_columns(
            pl.Series("direction_id", direction_id_col, dtype=pl.Int32),
            pl.Series("direction_conflict", issues_col, dtype=pl.Boolean),
        )

    def _warn_about_direction_conflict(self, directions: pl.DataFrame) -> None:
        """Emits a feed-wide summary `RuntimeWarning` about unresolved `direction_id` conflicts.

        Counts how many distinct shape_ids have at least one
        `direction_conflict=True` row -- a stop where the geometry
        genuinely disagreed with that shape's single reported
        `direction_id` (a real branching ambiguity
        `_assign_direction_ids_for_route` couldn't fully reconcile away --
        see its step 8) -- and at how many distinct stops that happened,
        both as absolute counts and as a percentage of all shape_ids/
        stop_ids actually processed. No-ops (emits nothing) when
        `directions` is empty or has no flagged rows.

        Args:
            directions: The concatenated per-route output of
                `_assign_direction_ids_for_route`, with `direction_id`/
                `direction_conflict` already populated.

        Returns:
            None. Emits a `RuntimeWarning` (via `warnings.warn`) as a side
            effect when at least one row is flagged.
        """
        if directions.height == 0:
            return

        total_shape_ids = directions["shape_id"].n_unique()
        total_stop_ids = directions["stop_id"].n_unique()

        issue_rows = directions.filter(pl.col("direction_conflict"))
        if issue_rows.height == 0:
            return

        n_stops_with_issues = issue_rows["stop_id"].n_unique()
        n_shape_ids_with_issues = issue_rows["shape_id"].n_unique()
        shape_pct = (100.0 * n_shape_ids_with_issues / total_shape_ids) if total_shape_ids else 0.0
        stop_pct = (100.0 * n_stops_with_issues / total_stop_ids) if total_stop_ids else 0.0

        warnings.warn(
            f"direction_id assignment: {n_shape_ids_with_issues} of {total_shape_ids} "
            f"shape_ids ({shape_pct:.1f}%) had at least one stop where the geometry "
            f"disagreed with their reported direction_id (direction_conflict=True), "
            f"across {n_stops_with_issues} of {total_stop_ids} stop(s) ({stop_pct:.1f}%).",
            RuntimeWarning,
        )
