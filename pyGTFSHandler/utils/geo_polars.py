# -*- coding: utf-8 -*-
"""Polars-native geometry, distance, and general-purpose polars utilities
shared across `models/`.

Why this module exists: distance and clustering calculations were previously
scattered inline (in `models/stops.py`, `models/shapes.py`) and leaned on
`geopandas`/`shapely`/`sklearn` for things that are actually simple, fast
polars expressions once written out explicitly -- great-circle (haversine)
distance between EPSG:4326 lon/lat points, and grid-bucketing points so nearby
candidate pairs can be found without an all-pairs cross join. Consolidating
these here means `Stops.group_stops` and `Shapes` reuse identical, tested
distance math instead of drifting apart, and keeps `stops.py`/`shapes.py`
from growing past a readable size.

All GTFS coordinates are EPSG:4326 (plain lon/lat degrees), so a closed-form
haversine expression is used directly on those columns -- this is both exact
enough for transit-scale distances (unlike a flat-earth/UTM approximation,
which distorts more the further a feed's stops are from its UTM zone's
central meridian) and avoids any external network/reprojection dependency.

This module also holds the handful of non-geometry polars expression helpers
(`filter_by_id_column`, `mean_angle`, `max_separation_angle`) that don't
belong in any of the other, more specific `utils/` modules -- they're
polars-expression utilities in the same spirit as the distance/grid helpers
above, just not geometric ones.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

import polars as pl

EARTH_RADIUS_M: float = 6371000.0


def haversine_distance_m(
    lat1: str | pl.Expr,
    lon1: str | pl.Expr,
    lat2: str | pl.Expr,
    lon2: str | pl.Expr,
) -> pl.Expr:
    """Builds a polars expression computing great-circle distance in meters.

    Args:
        lat1: Column name or expression for the first point's latitude (degrees).
        lon1: Column name or expression for the first point's longitude (degrees).
        lat2: Column name or expression for the second point's latitude (degrees).
        lon2: Column name or expression for the second point's longitude (degrees).

    Returns:
        pl.Expr: Distance in meters between the two points, evaluated
        row-wise via the standard haversine formula.
    """
    def _col(x: str | pl.Expr) -> pl.Expr:
        return pl.col(x) if isinstance(x, str) else x

    lat1, lon1, lat2, lon2 = _col(lat1), _col(lon1), _col(lat2), _col(lon2)

    lat1_rad = lat1.radians()
    lat2_rad = lat2.radians()
    dlat = (lat2 - lat1).radians()
    dlon = (lon2 - lon1).radians()

    a = (dlat / 2).sin() ** 2 + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin() ** 2
    return 2 * EARTH_RADIUS_M * a.sqrt().arcsin()


def degrees_per_meter(latitude_deg: float) -> tuple[float, float]:
    """Returns (degrees-latitude-per-meter, degrees-longitude-per-meter) at a
    given latitude, for sizing a lon/lat grid cell to a real-world distance.

    Used to bucket points into grid cells approximately `cell_size_m` wide,
    so nearby-candidate-pair search only needs to compare each point against
    the ~9 cells around it instead of an all-pairs cross join. It is
    intentionally an approximation (a flat local reference frame around the
    given latitude) -- fine for the modest cell sizes clustering uses (tens
    to low hundreds of meters); actual distances between candidate pairs are
    still verified with the exact `haversine_distance_m` above.

    Args:
        latitude_deg: Latitude (degrees) to approximate the local scale at.

    Returns:
        tuple[float, float]: (degrees latitude per meter, degrees longitude
        per meter).
    """
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = max(111320.0 * math.cos(math.radians(latitude_deg)), 1e-6)
    return 1.0 / meters_per_degree_lat, 1.0 / meters_per_degree_lon


def grid_cell_columns(
    lat_col: str,
    lon_col: str,
    cell_size_m: float,
    reference_latitude_deg: float,
) -> list[pl.Expr]:
    """Returns `[cell_lat, cell_lon]` integer-bucket expressions for `lat_col`/`lon_col`.

    Args:
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        cell_size_m: Desired grid cell size, in meters.
        reference_latitude_deg: Latitude used to approximate the local
            meters-per-degree scale (see `degrees_per_meter`); typically the
            mean latitude of the points being bucketed.

    Returns:
        list[pl.Expr]: Two expressions aliased `"cell_lat"`/`"cell_lon"`.
    """
    deg_per_m_lat, deg_per_m_lon = degrees_per_meter(reference_latitude_deg)
    cell_size_lat_deg = max(cell_size_m * deg_per_m_lat, 1e-9)
    cell_size_lon_deg = max(cell_size_m * deg_per_m_lon, 1e-9)
    return [
        (pl.col(lat_col) / cell_size_lat_deg).floor().cast(pl.Int64).alias("cell_lat"),
        (pl.col(lon_col) / cell_size_lon_deg).floor().cast(pl.Int64).alias("cell_lon"),
    ]


def connected_components_from_edges(n_nodes: int, edge_index_pairs: list[tuple[int, int]]) -> list[int]:
    """Labels each of `n_nodes` node indices [0, n_nodes) with its connected
    component id, given an undirected edge list.

    This is the one step that isn't a polars primitive -- graph connectivity
    -- so it's delegated to a single vectorized `scipy.sparse.csgraph` call
    over the whole edge list at once (not per-pair, not per-cluster), rather
    than pulling in a full clustering library (e.g. scikit-learn) for what is
    fundamentally just "which points are transitively within the distance
    threshold of one another."

    Args:
        n_nodes: Total number of nodes (points) being clustered.
        edge_index_pairs: List of `(i, j)` node-index pairs known to be
            within the clustering distance of one another.

    Returns:
        list[int]: Component label for each node index `0..n_nodes-1`.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if n_nodes == 0:
        return []

    if not edge_index_pairs:
        return list(range(n_nodes))

    rows = [pair[0] for pair in edge_index_pairs]
    cols = [pair[1] for pair in edge_index_pairs]
    data = [1] * len(rows)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(csgraph=adjacency, directed=False)
    return labels.tolist()


def filter_by_id_column(lf, column, ids: list | None = []):
    """Semi-joins `lf` down to rows whose `column` value is in `ids`.

    IDs read from any single GTFS file are always suffixed with
    `"_<original_value>_file_<n>"` (see `io.read_csv_lazy`) to keep IDs that
    collide across multiple loaded feeds distinct. A caller filtering by a
    plain, un-suffixed id (e.g. `Feed(dir, trip_ids=["T1"])`, exactly as
    documented) would otherwise never match anything, since the column only
    ever contains `"T1_file_0"`-style values -- even for a single feed. To
    support both a plain id and an already-suffixed one (e.g. re-using an id
    obtained from a previous query's output), this matches on the id with
    any trailing `_file_<n>` suffix stripped from both sides.

    Args:
        lf: LazyFrame to filter, or None (returned unchanged).
        column: Name of the id column to filter on.
        ids: List of ids to keep (plain or already `_file_<n>`-suffixed).
            Empty/None means "no filter" (returns `lf` unchanged).

    Returns:
        Optional[pl.LazyFrame]: The filtered LazyFrame, or None if `lf` is None.
    """
    if ids is None:
        ids = []

    if lf is None:
        return None

    if len(ids) > 0:
        unsuffixed_ids = [
            re.sub(r"_file_\d+$", "", str(i)) if i is not None else i for i in ids
        ]
        ids_df = pl.LazyFrame({"__unsuffixed_id": unsuffixed_ids}).unique()
        lf = lf.with_columns(
            pl.col(column).str.replace(r"_file_\d+$", "").alias("__unsuffixed_id")
        ).join(ids_df, on="__unsuffixed_id", how="semi").drop("__unsuffixed_id")

    return lf


def mean_angle(column: str, over: Optional[List[str]] = None) -> pl.Expr:
    """Compute the mean angle in degrees of a column of angles."""
    if over is None:
        mean_cos = pl.col(column).radians().cos().mean()
        mean_sin = pl.col(column).radians().sin().mean()
    else:
        mean_cos = pl.col(column).radians().cos().mean().over(over)
        mean_sin = pl.col(column).radians().sin().mean().over(over)

    return pl.arctan2(mean_sin, mean_cos).degrees().mod(360)


def max_separation_angle(df: pl.DataFrame, column: str) -> pl.Series:
    """
    Compute the maximum separation angle in a list column of angles (degrees).
    Returns a Series of maximum separation angles.
    """
    df = (
        df.with_columns(pl.col(column).list.concat(pl.col(column) + 180).alias(column))
          .with_columns([(pl.col(column) % 360).list.min().alias(f"{column}_min_angle")])
          .with_columns((pl.col(column) % 360 - pl.col(f"{column}_min_angle")).alias(f"{column}_normalized"))
          .with_columns([(pl.col(f"{column}_normalized").list.sort().list.concat(pl.lit([360]))).alias(f"{column}_angle_sorted")])
          .with_columns([pl.col(f"{column}_angle_sorted").list.diff(null_behavior="drop").alias(f"{column}_arc_angle")])
          .with_columns([(pl.col(f"{column}_arc_angle").list.max() / 2 +
                          pl.col(f"{column}_angle_sorted").list.get(pl.col(f"{column}_arc_angle").list.arg_max()) +
                          pl.col(f"{column}_min_angle")).alias(f"{column}_max_separation_angle")])
    )
    return df[f"{column}_max_separation_angle"]
