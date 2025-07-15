import polars as pl
import geopandas as gpd
from pathlib import Path
import utils


class Stops:
    """
    A class to manage GTFS stops using Polars LazyFrames and GeoPandas GeoDataFrames.

    Provides functionality to:
    - Read and optionally filter GTFS stops
    - Filter stops by area of interest (AOI)
    - Group stops spatially and assign `parent_station` values

    Attributes:
        lf (pl.LazyFrame): LazyFrame containing GTFS stops.
        gdf (gpd.GeoDataFrame): GeoDataFrame of stop_id and geometry.
        stop_ids (list[str]): List of stop IDs currently loaded.
        path (Path): Path to GTFS 'stops.txt' file.
    """

    def __init__(
        self,
        path: str | Path,
        aoi: gpd.GeoDataFrame | gpd.GeoSeries = None,
        stop_ids: list[str] = None,
    ):
        """
        Initialize Stops instance and load GTFS stops.

        Args:
            path (str | Path): Path to the GTFS directory.
            aoi (GeoDataFrame | GeoSeries, optional): Area of interest for spatial filtering.
            stop_ids (list[str], optional): List of stop IDs to include.
        """
        self.path = Path(path) / "stops.txt"
        self.lf = self.__read_stops(stop_ids)

        if aoi:
            self.lf, self.gdf = self.__filter_by_aoi(aoi)
        else:
            df = self.lf.select(["stop_id", "stop_lat", "stop_lon"]).collect()
            self.gdf = gpd.GeoDataFrame(
                {"stop_id": df["stop_id"]},
                geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
                crs="EPSG:4326",
            )

        self.stop_ids = self.lf.select("stop_id").collect()["stop_id"].to_list()

    def __read_stops(self, stop_ids: list[str] = None) -> pl.LazyFrame:
        """
        Read GTFS stops.txt and filter by stop IDs if provided.

        Ensures 'parent_station' column exists.

        Args:
            stop_ids (list[str], optional): Stop IDs to filter by.

        Returns:
            pl.LazyFrame: Filtered and normalized stops LazyFrame.
        """
        schema_dict = utils.get_df_schema_dict(self.path)

        lf = utils.read_csv_list(self.path, schema_overrides=schema_dict)

        if stop_ids:
            stop_ids_df = pl.DataFrame({"stop_id": stop_ids})
            lf = lf.join(stop_ids_df.lazy(), on="stop_id", how="inner")

        if "parent_station" not in lf.schema:
            lf = lf.with_columns(pl.lit(None).alias("parent_station"))

        return lf

    def __filter_by_aoi(
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

        df = filtered_lf.select(["stop_id", "stop_lat", "stop_lon"]).collect()

        gdf = gpd.GeoDataFrame(
            {"stop_id": df["stop_id"]},
            geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
            crs="EPSG:4326",
        )

        union_geom = aoi.unary_union
        gdf = gdf[gdf.intersects(union_geom)]

        if gdf.empty:
            raise ValueError("No stops found inside AOI bounds")

        stop_ids_df = pl.DataFrame({"stop_id": gdf["stop_id"].to_list()})
        final_lf = filtered_lf.join(stop_ids_df.lazy(), on="stop_id", how="inner")

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
        parent_station_df = self.lf.select(["stop_id", "parent_station"]).collect()

        cluster_df = pl.DataFrame(
            {"stop_id": gdf["stop_id"].to_list(), "cluster": gdf["cluster"].to_list()}
        ).join(parent_station_df, on="stop_id", how="left")

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
            .explode("stop_id")
            .drop("final_cluster")
        )

        # Update gdf and lf
        self.gdf = self.gdf.merge(cluster_df.to_pandas(), on="stop_id")
        self.lf = self.lf.drop("parent_station").join(
            cluster_df.lazy(), on="stop_id", how="left"
        )
