import polars as pl
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Union, List
from .. import utils, gtfs_checker
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

        self.lf = self.__read_stops(paths, stop_ids, check_files=check_files, min_file_id=min_file_id)

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

    def __read_stops(
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
            new_p = gtfs_checker.search_file(p, file=file)
            if new_p is None:
                stop_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                stop_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("stops.txt")
        lf = utils.read_csv_list(stop_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id)

        lf = utils.filter_by_id_column(lf, "stop_id", stop_ids)

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

    def group_stops(self, distance: float):
        """
        Groups nearby stops by spatial proximity and assigns consistent parent_station values.

        Args:
            distance (float): Max distance (in meters) for grouping.

        Raises:
            ValueError: If projection is missing or invalid.
        """
        from sklearn.cluster import AgglomerativeClustering

        gdf = self.gdf.copy()

        if not gdf.crs or not gdf.crs.is_projected:
            gdf = gdf.to_crs(gdf.estimate_utm_crs())

        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y

        if len(gdf) == 1:
            gdf["cluster"] = 0
        elif len(gdf) == 0:
            gdf["cluster"] = pd.Series(dtype=int)
        else:
            gdf["cluster"] = (
                AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance,
                    metric="euclidean",
                    linkage="complete",
                )
                .fit(gdf[["x", "y"]])
                .labels_
            )

        # This assumes parent_station info comes from self.lf (not self.gdf)
        parent_station_df = self.lf.select(["stop_id", "parent_station"])
        stop_ids_df = pl.from_pandas(gdf[["stop_id", "cluster"]]).lazy()

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
            new_p = gtfs_checker.search_file(p, file=file)
            if new_p is None:
                stop_paths.append(None)
            else:
                stop_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("stops.txt")
        stops = utils.read_csv_list(stop_paths, schema_overrides=schema_dict, check_files=True)

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

        self.lf = stops
        return None
