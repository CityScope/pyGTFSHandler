import polars as pl
from pathlib import Path
from typing import Union, List
import geopandas as gpd
import math

"TODO: read shapes file and use it when available"
"TODO: finish shape gdf"
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
        check_files:bool=False
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
        self.stop_shapes = self.lf.filter(pl.col("stop_id").is_not_null())
        self.stop_shapes = self.__generate_shape_direction_column(self.stop_shapes)
        self.stop_shapes = self.stop_shapes.collect().lazy()
        self.gdf = self.__get_shapes_gdf(self.lf)

    def __generate_shape_direction_column(self, stop_shapes):
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
            # .with_columns(
            #     [
            #         ((pl.col("shape_direction") / round_factor).round(0) * round_factor).alias(
            #             "shape_direction"
            #         ),
            #         ((pl.col("shape_direction_backwards") / round_factor).round(0) * round_factor).alias(
            #             "shape_direction_backwards"
            #         ),
            #     ]
            # )
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
                "lat1_rad",
                "lat2_rad_rev",
                "y_rev",
                "x_rev",
            )
        )

        return stop_shapes

    def __generate_shapes_file(self, stops_lf, trip_shape_ids_lf):
        trip_shape_ids_lf = (
            trip_shape_ids_lf.select(["shape_id", "stop_ids", "stop_sequence"])
            .explode(["stop_ids", "stop_sequence"])
            .rename({"stop_ids": "stop_id"})
        )

        trip_shape_ids_lf = trip_shape_ids_lf.join(
            stops_lf.select("stop_id", "stop_lat", "stop_lon"),
            on=["stop_id"],
            how="left",
        )

        # Prepare stops_lf as shape points with null sequence
        shapes = (
            trip_shape_ids_lf.select(
                [
                    "shape_id",
                    "stop_id",
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

    def __get_shapes_gdf(self, shapes: pl.LazyFrame) -> gpd.GeoDataFrame:
        """
        Convert GTFS shapes.txt into a GeoDataFrame of LINESTRING geometries.
        """
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
            .with_columns(
                (
                    pl.when(pl.col("pt").list.len() == 1)
                    .then(
                        pl.concat_str(
                            [pl.lit("Point("), pl.col("pt").list.join(""), pl.lit(")")]
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
