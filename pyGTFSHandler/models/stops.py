# -*- coding: utf-8 -*-
"""GTFS stops.txt handling: loading, AOI filtering, and proximity clustering.

Why this module exists and how it's organized:
-----------------------------------------------
- **Loading** (`_read_stops`): reads `stops.txt`, validates `stop_lat`/
  `stop_lon` (drops null/non-finite/out-of-range coordinates and warns),
  fills a missing `parent_station` with the stop's own id so every stop
  always has one, and rejects a `stop_id` duplicated *within a single
  source file* (a genuine data error) while leaving the same id reused
  *across* different loaded feeds untouched (that's a multi-feed id-
  collision concern, namespaced by `io.read_csv_lazy`'s `_file_<n>`
  suffixing, not a `stops.txt`-internal one).
- **AOI filtering** (`filter_by_aoi`): a cheap polars bounding-box
  pre-filter, followed by a precise `geopandas`/`shapely` intersection --
  the one place in this file geopandas does real geometric work, since
  arbitrary-polygon containment isn't a reasonable thing to reimplement in
  polars.
- **Clustering** (`group_stops`/`_cluster_by_distance`): groups stops within
  `distance` meters of one another (transitively -- a chain of stops each
  close to the next all land in one cluster even if the two ends are far
  apart) into a shared `parent_station`. This is done entirely in polars +
  `scipy`: bucket stops into a lon/lat grid sized to `distance`
  (`geo_polars.grid_cell_columns`), join each stop against only its own and
  neighboring cells for cheap candidate pairs (never an all-pairs cross
  join), keep pairs within the exact haversine distance
  (`geo_polars.haversine_distance_m`), and take connected components of
  that edge graph (`geo_polars.connected_components_from_edges`) -- no
  scikit-learn, no UTM reprojection.
"""

import polars as pl
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Union, List
from ..utils import gtfs_checker
from ..utils import io
from ..utils import geo_polars
import os
import warnings

class Stops:
    """
    A class to manage GTFS stops using Polars LazyFrames and GeoPandas GeoDataFrames.

    Provides functionality to:
    - Read and optionally filter GTFS stops from one or more directories
    - Filter stops by area of interest (AOI)
    - Group stops spatially and assign `parent_station` values

    Attributes:
        lf (pl.LazyFrame): LazyFrame containing GTFS stops.
        gdf (gpd.GeoDataFrame): GeoDataFrame of stop_id and geometry.
        stop_ids (list[str]): List of stop IDs currently loaded.
        paths (List[Path]): List of GTFS paths (directories).
    """

    def __init__(self,lf=None,gdf=None,stop_ids=None, mean_lon=None, mean_lat=None):
        self.lf = lf
        self.gdf = gdf 
        self.stop_ids = stop_ids
        if (mean_lon is None) or (mean_lat is None):
            if lf is not None: 
                mean_coords = lf.select(
                    [
                        pl.col("stop_lon").mean().alias("mean_lon"),
                        pl.col("stop_lat").mean().alias("mean_lat"),
                    ]
                ).collect()

                mean_lon = mean_coords["mean_lon"][0]
                mean_lat = mean_coords["mean_lat"][0]

        self.mean_lon = mean_lon 
        self.mean_lat = mean_lat 

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        aoi: Union[gpd.GeoDataFrame, gpd.GeoSeries, None] = None,
        stop_group_distance: float = 0,
        stop_ids: Union[List[str], pl.DataFrame | pl.LazyFrame] = None,
        check_files:bool=False,
        min_file_id:int=0,
    ):
        """
        Initialize Stops instance and load GTFS stops from one or more files.

        Args:
            path (str | Path | list): One or more paths to GTFS directories.
            aoi (GeoDataFrame | GeoSeries, optional): Area of interest for spatial filtering.
            stop_ids (list[str], optional): List of stop IDs to include.
        """
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        self.lf = self._read_stops(paths, stop_ids, check_files=check_files, min_file_id=min_file_id)
        
        if aoi is None:
            df = self.lf.select(
                ["stop_id", "parent_station", "stop_lat", "stop_lon"]
            ).collect()
            self.gdf = gpd.GeoDataFrame(
                {
                    "stop_id": df["stop_id"],
                    "parent_station": df["parent_station"],
                },
                geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
                crs="EPSG:4326",
            )
        else:
            self.lf = self.lf.collect().lazy()
            self.lf, self.gdf = self.filter_by_aoi(aoi)
            self.lf = self.lf.collect().lazy()

        if stop_group_distance > 0:
            self.lf = self.lf.collect().lazy()
            self.lf, self.gdf = self.group_stops(stop_group_distance)
            self.lf = self.lf.collect().lazy()

        if (aoi is not None) or (stop_group_distance > 0):
            self.stop_ids = self.lf.select("stop_id").collect()["stop_id"].to_list()
            if (len(self.stop_ids) > 0) and (self.stop_ids[0] is None):
                self.stop_ids = []
        else:
            self.stop_ids = None

        # Compute mean coordinates of stops (assumed self.stops.lf is LazyFrame)
        mean_coords = self.lf.select(
            [
                pl.col("stop_lon").mean().alias("mean_lon"),
                pl.col("stop_lat").mean().alias("mean_lat"),
            ]
        ).collect()

        self.mean_lon = mean_coords["mean_lon"][0]
        self.mean_lat = mean_coords["mean_lat"][0]

    def _read_stops(
        self, paths, stop_ids: Union[List[str], None] = None, check_files=False, min_file_id=0
    ) -> pl.LazyFrame:
        """
        Read GTFS stops.txt files and filter by stop IDs if provided.

        Ensures 'parent_station' column exists across all files.

        Args:
            stop_ids (list[str], optional): Stop IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered and normalized stops LazyFrame.
        """
        stop_paths: List[Path] = []
        file = "stops.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                stop_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                stop_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("stops.txt")
        lf = io.read_csv_list(stop_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)
        if (lf is None) or (lf.select(pl.len()).collect().item() == 0):
            raise Exception(f"No stops.txt file found for any {paths}")
        
        lf = geo_polars.filter_by_id_column(lf, "stop_id", stop_ids)

        if "parent_station" not in lf.collect_schema().names():
            lf = lf.with_columns(pl.lit(None).alias("parent_station"))


        lf = lf.with_columns(
            pl.when(pl.col("parent_station") == "")
            .then(pl.lit(None))
            .otherwise("parent_station")
            .alias("parent_station"),
            pl.when(pl.col("stop_id") == "")
            .then(pl.lit(None))
            .otherwise("stop_id")
            .alias("stop_id")
        ).with_columns(
            pl.when(pl.col("parent_station").is_null())
            .then(pl.col("stop_id"))
            .otherwise("parent_station")
            .alias("parent_station"),
            pl.when(pl.col("stop_id").is_null())
            .then(pl.col("parent_station"))
            .otherwise("stop_id")
            .alias("stop_id")
        )
        lf = lf.filter(pl.col("stop_id").is_not_null() & (pl.col("stop_id") != ""))
        lf = lf.filter(pl.col("stop_lat").is_not_null() & pl.col("stop_lon").is_not_null())
        lf = lf.filter(pl.col("stop_lat").is_finite() & pl.col("stop_lon").is_finite())
        lf = lf.filter(
            pl.col("stop_lat").is_between(-90, 90) & pl.col("stop_lon").is_between(-180, 180)
        )

        # A duplicate `stop_id` *within a single source file* is a genuine
        # data error (the same file can't mean two different physical stops
        # by the same id). Duplicates *across* different loaded feeds are a
        # separate, intentional case (`Feed(gtfs_dirs=[...])` stacking/
        # multi-feed loading) and are handled there, not flagged here.
        duplicate_ids = (
            lf.group_by(["file_id", "stop_id"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .select("stop_id")
            .collect()["stop_id"]
            .to_list()
        )
        if duplicate_ids:
            raise Exception(
                f"stops.txt has duplicate stop_id value(s) within the same file: {duplicate_ids}"
            )

        return lf

    def filter_by_aoi(
        self, aoi: gpd.GeoDataFrame | gpd.GeoSeries
    ) -> tuple[pl.LazyFrame, gpd.GeoDataFrame]:
        """
        Filters stops by a given Area of Interest (AOI).

        Performs:
        1. Bounding box filter on LazyFrame (approximate).
        2. Geometry-based intersection on GeoDataFrame (precise).

        Args:
            aoi (GeoDataFrame | GeoSeries): Area to filter stops within.

        Returns:
            tuple: (filtered LazyFrame, filtered GeoDataFrame)

        Raises:
            ValueError: If no stops are found within AOI.
        """
        aoi = aoi.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = aoi.total_bounds

        filtered_lf = self.lf.filter(
            (pl.col("stop_lon") > minx)
            & (pl.col("stop_lon") < maxx)
            & (pl.col("stop_lat") > miny)
            & (pl.col("stop_lat") < maxy)
        )

        df = filtered_lf.select(
            ["stop_id", "parent_station", "stop_lat", "stop_lon"]
        ).collect()

        gdf = gpd.GeoDataFrame(
            {
                "stop_id": df["stop_id"],
                "parent_station": df["parent_station"],
            },
            geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
            crs="EPSG:4326",
        )

        union_geom = aoi.union_all()
        gdf = gdf[gdf.intersects(union_geom)]

        if gdf.empty:
            raise ValueError("No stops found inside AOI bounds")

        stop_ids_df = pl.from_pandas(gdf[["stop_id"]]).lazy()
        final_lf = filtered_lf.join(stop_ids_df.lazy(), on=["stop_id"], how="semi")

        return final_lf, gdf

    def _cluster_by_distance(self, stop_coords: pl.DataFrame, distance: float) -> pl.DataFrame:
        """Assigns a `cluster` id (int) to each `stop_id` in `stop_coords`,
        grouping stops transitively within `distance` meters of one another.

        See `group_stops` for why this replaces the previous sklearn
        `AgglomerativeClustering`-based approach.

        Args:
            stop_coords: DataFrame with `stop_id`, `stop_lat`, `stop_lon`
                (one row per stop, no nulls, no duplicate `stop_id`).
            distance: Clustering distance threshold, in meters. `<= 0` means
                "no clustering" (every stop its own cluster).

        Returns:
            pl.DataFrame: `stop_id`, `cluster` (int).
        """
        n = stop_coords.height
        if n == 0:
            return pl.DataFrame({"stop_id": [], "cluster": []}, schema={"stop_id": pl.Utf8, "cluster": pl.Int64})
        if n == 1 or distance <= 0:
            return stop_coords.select("stop_id").with_row_index("cluster")

        indexed = stop_coords.with_row_index("idx")
        # Rounded: polars' `.mean()` sums in parallel, and float addition
        # isn't associative, so the exact result can differ in its last
        # bit(s) between runs on identical input. This value only sets the
        # grid's cos(lat) meters-per-degree scale (nothing needs it more
        # precise than ~0.1m), but left unrounded, that run-to-run jitter
        # can flip which grid cell a stop sitting right on a cell boundary
        # falls into -- silently changing cluster membership (and so
        # `parent_station`/`direction_conflict` results) between otherwise
        # identical runs.
        reference_latitude = round(indexed["stop_lat"].mean(), 6)
        cell_exprs = geo_polars.grid_cell_columns(
            "stop_lat", "stop_lon", cell_size_m=distance, reference_latitude_deg=reference_latitude
        )
        indexed = indexed.with_columns(cell_exprs)

        # Candidate pairs: each point against every point sharing its own
        # cell or one of the 8 neighboring cells (never an all-pairs join).
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        neighbor_frames = [
            indexed.select(
                (pl.col("cell_lat") + d_lat).alias("cell_lat"),
                (pl.col("cell_lon") + d_lon).alias("cell_lon"),
                pl.col("idx").alias("idx_b"),
                pl.col("stop_lat").alias("stop_lat_b"),
                pl.col("stop_lon").alias("stop_lon_b"),
            )
            for d_lat, d_lon in neighbor_offsets
        ]
        neighbors = pl.concat(neighbor_frames)

        candidate_pairs = indexed.select("idx", "stop_lat", "stop_lon", "cell_lat", "cell_lon").join(
            neighbors, on=["cell_lat", "cell_lon"], how="inner"
        ).filter(pl.col("idx") < pl.col("idx_b"))

        candidate_pairs = candidate_pairs.with_columns(
            # Rounded to the millimeter: a pair sitting almost exactly on
            # the `distance` threshold can otherwise flip in or out of
            # range between runs from sub-micron floating-point jitter in
            # the trig chain below (not something GTFS coordinate
            # precision could ever resolve anyway) -- silently changing
            # cluster/`parent_station` membership, and everything
            # downstream of it, on identical input.
            geo_polars.haversine_distance_m("stop_lat", "stop_lon", "stop_lat_b", "stop_lon_b").round(3).alias("dist_m")
        ).filter(pl.col("dist_m") <= distance)

        edges = list(zip(candidate_pairs["idx"].to_list(), candidate_pairs["idx_b"].to_list()))
        labels = geo_polars.connected_components_from_edges(n, edges)

        return indexed.select("idx", "stop_id").with_columns(
            pl.Series("cluster", [labels[i] for i in indexed["idx"].to_list()])
        ).select("stop_id", "cluster")

    def group_stops(self, distance: float):
        """
        Groups nearby stops by spatial proximity and assigns consistent parent_station values.

        Args:
            distance (float): Max distance (in meters) for grouping.

        Raises:
            ValueError: If projection is missing or invalid.
        """
        # Polars-native clustering: bucket stops into a lon/lat grid sized to
        # `distance`, self-join each stop against its own cell plus the 8
        # neighboring cells to get cheap candidate pairs (never an all-pairs
        # cross join), keep pairs whose exact haversine distance is within
        # `distance`, and take connected components of that edge graph --
        # this is a *transitive* grouping (a chain of stops each within
        # `distance` of the next all end up in one cluster, even if the two
        # ends are farther apart than `distance`), unlike the "complete"
        # linkage clustering this replaces. No sklearn, no UTM projection.
        stop_coords = (
            self.lf.select(["stop_id", "stop_lat", "stop_lon"])
            .filter(
                pl.col("stop_id").is_not_null()
                & pl.col("stop_lat").is_not_null()
                & pl.col("stop_lon").is_not_null()
            )
            .unique("stop_id")
            .collect()
        )

        cluster_labels = self._cluster_by_distance(stop_coords, distance)

        # This assumes parent_station info comes from self.lf (not self.gdf)
        parent_station_df = self.lf.select(["stop_id", "parent_station"])
        stop_ids_df = cluster_labels.lazy()

        cluster_df = stop_ids_df.join(parent_station_df, on=["stop_id"], how="left")

        # Merge clusters with common parent_station
        cluster_df = cluster_df.join(
            cluster_df.filter(
                pl.col("parent_station").is_not_null()
                & (pl.col("parent_station") != "")
            )
            .group_by("parent_station")
            .agg(pl.col("cluster").min().alias("merged_cluster")),
            on="parent_station",
            how="left",
        )

        cluster_df = cluster_df.with_columns(
            [
                pl.when(pl.col("merged_cluster").is_not_null())
                .then(pl.col("merged_cluster"))
                .otherwise(pl.col("cluster"))
                .alias("final_cluster"),
                pl.when(
                    pl.col("parent_station").is_not_null()
                    & (pl.col("parent_station") != "")
                )
                .then(pl.col("parent_station"))
                .otherwise(pl.col("stop_id"))
                .alias("fallback_id"),
            ]
        ).drop(["cluster", "merged_cluster"])

        # Assign one parent_station per cluster
        cluster_df = (
            cluster_df.group_by("final_cluster")
            .agg(
                [
                    pl.col("stop_id"),
                    pl.col("fallback_id").first().alias("parent_station"),
                ]
            )
            .with_columns(
                pl.col("parent_station")
                .cum_count()
                .over("parent_station")
                .alias("suffix_count"),
                pl.col("parent_station").is_duplicated().alias("is_dup"),
            )
            .explode(["stop_id"])
            .with_columns(
                (
                    pl.when(pl.col("is_dup"))
                    .then(
                        pl.col("parent_station")
                        + "_duplicated_"
                        + pl.col("suffix_count").cast(pl.Utf8)
                    )
                    .otherwise(pl.col("parent_station"))
                ).alias("parent_station")
            )
            .drop(["suffix_count", "is_dup", "final_cluster"])
            .with_columns(
                pl.when(
                    pl.col("parent_station").is_null()
                ).then(
                    pl.lit(None))
                .otherwise(
                    pl.col("parent_station")
                ).alias("parent_station")
            )
            .with_columns(
                pl.when(
                    pl.col("parent_station").is_null()
                ).then(
                    pl.col("stop_id"))
                .otherwise(
                    pl.col("parent_station")
                ).alias("parent_station")
            )
        )

        cluster_df = cluster_df.collect()

        # Update gdf and lf
        gdf = self.gdf.drop(columns=["parent_station"]).merge(
            cluster_df.to_pandas(), on=["stop_id"]
        )
        lf = self.lf.drop("parent_station").join(
            cluster_df.lazy(), on=["stop_id"], how="left"
        )

        return lf, gdf

    def reload_stops_lf(self, path, stop_ids=None):
        if isinstance(path, (str, Path)):
            paths = [Path(path)]
        else:
            paths = [Path(p) for p in path]

        stop_paths: List[Path] = []
        file = "stops.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                stop_paths.append(None)
            else:
                stop_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("stops.txt")
        stops = io.read_csv_list(stop_paths, schema_overrides=schema_dict, check_files=True)
        stops = stops.filter(
            pl.col("stop_lat").is_not_null() & pl.col("stop_lon").is_not_null()
            & pl.col("stop_lat").is_finite() & pl.col("stop_lon").is_finite()
            & pl.col("stop_lat").is_between(-90, 90) & pl.col("stop_lon").is_between(-180, 180)
        )

        if isinstance(stop_ids, list):
            stop_ids_lf = pl.LazyFrame({"stop_id": stop_ids})

            # Select matching stop_times with just needed columns
            stops = stops.join(stop_ids_lf, on="stop_id", how="semi")
        elif stop_ids is not None:
            if isinstance(stop_ids, pl.DataFrame):
                stop_ids = stop_ids.lazy()

            columns = stop_ids.collect_schema().names()

            stops = stops.join(stop_ids, on=columns, how="semi")

        if "parent_station" in stops.collect_schema().names():
            stops = stops.with_columns(
                (
                    pl.when(pl.col("parent_station").is_null())
                    .then(pl.col("stop_id"))
                    .otherwise(pl.col("parent_station"))
                ).alias("parent_station")
            )
        else:
            stops = stops.with_columns(pl.col("stop_id").alias("parent_station"))

        stops = (
            stops.join(
                self.lf.select("stop_id", "parent_station").rename(
                    {"parent_station": "parent_station_right"}
                ),
                on=["stop_id"],
                how="left",
            )
            .with_columns(
                (
                    pl.when(pl.col("parent_station_right").is_null())
                    .then(pl.col("parent_station"))
                    .otherwise(pl.col("parent_station_right"))
                ).alias("parent_station")
            )
            .drop("parent_station_right")
        )

        stops = stops.with_columns(
            pl.when(pl.col("parent_station") == "")
            .then(pl.lit(None))
            .otherwise("parent_station")
            .alias("parent_station"),
            pl.when(pl.col("stop_id") == "")
            .then(pl.lit(None))
            .otherwise("stop_id")
            .alias("stop_id")
        ).with_columns(
            pl.when(pl.col("parent_station").is_null())
            .then(pl.col("stop_id"))
            .otherwise("parent_station")
            .alias("parent_station"),
            pl.when(pl.col("stop_id").is_null())
            .then(pl.col("parent_station"))
            .otherwise("stop_id")
            .alias("stop_id")
        )
        stops = stops.filter(pl.col("stop_id").is_not_null() & (pl.col("stop_id") != ""))
        stops = stops.filter(pl.col("stop_lat").is_not_null() & pl.col("stop_lon").is_not_null())
        
        self.lf = stops
        return None
