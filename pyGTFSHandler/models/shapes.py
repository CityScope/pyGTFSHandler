import polars as pl
from pathlib import Path
from typing import Union, List
import geopandas as gpd
from shapely import wkt
import math

"TODO: read shapes file and use it when available"
"TODO: define directions clustering shape ids by similar directionality in linestrings by consistent with routes or maybe one direction per stop or somethig"

TRIP_ROUND_TIME = 120


class Shapes:
    """
    Manage GTFS shapes.txt data using Polars LazyFrames.

    """

    def __init__(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        trip_shape_ids_lf,
        stops_lf,
    ):
        """
        Initialize the Shapes instance.

        """
        if isinstance(path, (str, Path)):
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]

        self.lf = self.__generate_shapes_file(stops_lf, trip_shape_ids_lf)
        self.lf = self.lf.collect().lazy()
        self.gdf = self.__get_shapes_gdf(self.lf)

    def __generate_shapes_file(self, stops_lf, trip_shape_ids_lf):
        trip_shape_ids_lf = (
            trip_shape_ids_lf.select(["shape_id", "stop_ids", "stop_sequence"])
            .explode(["stop_ids", "stop_sequence"])
            .rename({"stop_ids": "stop_id"})
        )

        trip_shape_ids_lf = trip_shape_ids_lf.join(
            stops_lf.select("stop_id", "stop_lat", "stop_lon"), on="stop_id", how="left"
        )

        # Prepare stops_lf as shape points with null sequence
        shapes = (
            trip_shape_ids_lf.select(
                [
                    "shape_id",
                    "stop_sequence",
                    "stop_lat",
                    "stop_lon",
                ]
            )
            .rename(
                {
                    "stop_lat": "shape_pt_lat",
                    "stop_lon": "shape_pt_lon",
                }
            )
            .with_columns(pl.col("stop_sequence").alias("shape_pt_sequence"))
        )

        shapes = self.__generate_shape_dist_traveled_column(shapes)
        return shapes

    def __generate_shape_dist_traveled_column(self, shapes):
        R = 6371000  # Earth radius in meters
        deg2rad = math.pi / 180

        shapes = shapes.with_columns(
            [
                # Convert lat/lon to radians
                (pl.col("shape_pt_lat") * deg2rad).alias("lat_rad"),
                (pl.col("shape_pt_lon") * deg2rad).alias("lon_rad"),
            ]
        )

        # Calculate cumulative distance per shape_id without intermediate columns
        shapes = (
            shapes.sort("shape_pt_sequence")
            .with_columns(
                [
                    (
                        (
                            2
                            * pl.arctan2(
                                (
                                    (
                                        (
                                            (
                                                pl.col("lat_rad")
                                                - pl.col("lat_rad")
                                                .shift(1)
                                                .over("shape_id")
                                            )
                                            / 2
                                        ).sin()
                                        ** 2
                                    )
                                    + (
                                        pl.col("lat_rad").shift(1).over("shape_id")
                                    ).cos()
                                    * (pl.col("lat_rad")).cos()
                                    * (
                                        (
                                            (
                                                pl.col("lon_rad")
                                                - pl.col("lon_rad")
                                                .shift(1)
                                                .over("shape_id")
                                            )
                                            / 2
                                        ).sin()
                                        ** 2
                                    )
                                ).sqrt(),
                                (
                                    1
                                    - (
                                        (
                                            (
                                                (
                                                    pl.col("lat_rad")
                                                    - pl.col("lat_rad")
                                                    .shift(1)
                                                    .over("shape_id")
                                                )
                                                / 2
                                            ).sin()
                                            ** 2
                                        )
                                        + (
                                            pl.col("lat_rad").shift(1).over("shape_id")
                                        ).cos()
                                        * (pl.col("lat_rad")).cos()
                                        * (
                                            (
                                                (
                                                    pl.col("lon_rad")
                                                    - pl.col("lon_rad")
                                                    .shift(1)
                                                    .over("shape_id")
                                                )
                                                / 2
                                            ).sin()
                                            ** 2
                                        )
                                    )
                                ).sqrt(),
                            )
                        )
                        * R
                    )
                    .fill_null(
                        0
                    )  # first row per shape_id has no previous point, so distance=0
                    .alias("dist_from_prev")
                ]
            )
            .drop(["lat_rad", "lon_rad"])
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

    def __get_shapes_gdf(self, shapes):
        return None
        # Create WKT LINESTRINGs grouped by shape_id
        shapes_gdf = (
            shapes.group_by("shape_id")
            .agg(
                [
                    pl.format(
                        "LINESTRING({})",
                        pl.concat_str(
                            pl.col("shape_pt_lon")
                            .sort_by("shape_pt_sequence")
                            .cast(str)
                            + " "
                            + pl.col("shape_pt_lat")
                            .sort_by("shape_pt_sequence")
                            .cast(str),
                            separator=", ",
                        ),
                    ).alias("geometry")
                ]
            )
            .collect()
            .to_pandas()
        )

        # Convert WKT strings to shapely geometries
        shapes_gdf["geometry"] = shapes_gdf["geometry"].apply(wkt.loads)

        # Convert to GeoDataFrame and assign CRS (EPSG:4326)
        shapes_gdf = gpd.GeoDataFrame(shapes_gdf, geometry="geometry", crs="EPSG:4326")
        return shapes_gdf
