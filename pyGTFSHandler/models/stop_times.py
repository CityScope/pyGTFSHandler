import polars as pl
from pathlib import Path
import utils


class StopTimes:
    """
    Manage GTFS stop_times.txt data using Polars LazyFrames.

    Features:
    - Filter by stop_ids and/or trip_ids
    - Normalize time strings
    - Convert time to/from seconds since midnight
    - Interpolate missing times, optionally using shape_dist_traveled for more accurate interpolation
    - Return a fully processed LazyFrame ready for analysis
    """

    def __init__(
        self, path: str | Path, stop_ids: list[str] = None, trip_ids: list[str] = None
    ):
        """
        Initialize the StopTimes instance.

        Args:
            path (str | Path): Path to the directory containing stop_times.txt.
            stop_ids (list[str], optional): List of stop IDs to filter on.
            trip_ids (list[str], optional): List of trip IDs to filter on.
        """
        self.path = Path(path) / "stop_times.txt"
        self.lf = self.__read_stop_times(stop_ids, trip_ids)

    @staticmethod
    def __interpolate_times_with_shape_dist(stop_times: pl.LazyFrame) -> pl.LazyFrame:
        """
        Interpolate missing times using proportional interpolation based on shape_dist_traveled.

        Args:
            stop_times (pl.LazyFrame): Input LazyFrame with stop times and shape_dist_traveled.

        Returns:
            pl.LazyFrame: LazyFrame with interpolated departure_time_secs and arrival_time_secs.
        """
        # Sort by trip_id and shape_dist_traveled
        stop_times = stop_times.sort(["trip_id", "shape_dist_traveled"])

        # Forward and backward fills for times and shape distances over trip_id groups
        dep_ffill = pl.col("departure_time_secs").forward_fill().over("trip_id")
        dep_bfill = pl.col("departure_time_secs").backward_fill().over("trip_id")
        arr_ffill = pl.col("arrival_time_secs").forward_fill().over("trip_id")
        arr_bfill = pl.col("arrival_time_secs").backward_fill().over("trip_id")

        dist_ffill = pl.col("shape_dist_traveled").forward_fill().over("trip_id")
        dist_bfill = pl.col("shape_dist_traveled").backward_fill().over("trip_id")

        # Proportional interpolation formulas
        dep_interp = dep_ffill + (dep_bfill - dep_ffill) * (
            (pl.col("shape_dist_traveled") - dist_ffill) / (dist_bfill - dist_ffill)
        )
        arr_interp = arr_ffill + (arr_bfill - arr_ffill) * (
            (pl.col("shape_dist_traveled") - dist_ffill) / (dist_bfill - dist_ffill)
        )

        # Replace null times with interpolated values
        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("departure_time_secs").is_null())
                .then(dep_interp)
                .otherwise(pl.col("departure_time_secs"))
                .alias("departure_time_secs"),
                pl.when(pl.col("arrival_time_secs").is_null())
                .then(arr_interp)
                .otherwise(pl.col("arrival_time_secs"))
                .alias("arrival_time_secs"),
            ]
        )

        return stop_times

    @staticmethod
    def __interpolate_times(stop_times: pl.LazyFrame) -> pl.LazyFrame:
        """
        Interpolate missing times linearly ordered by stop_sequence within each trip.

        Args:
            stop_times (pl.LazyFrame): Input LazyFrame with stop times.

        Returns:
            pl.LazyFrame: LazyFrame with interpolated departure_time_secs and arrival_time_secs.
        """
        return stop_times.with_columns(
            [
                pl.col("departure_time_secs")
                .fill_null(strategy="linear")
                .over("trip_id")
                .sort_by("stop_sequence")
                .alias("departure_time_secs"),
                pl.col("arrival_time_secs")
                .fill_null(strategy="linear")
                .over("trip_id")
                .sort_by("stop_sequence")
                .alias("arrival_time_secs"),
            ]
        )

    def __read_stop_times(
        self, stop_ids: list[str] = None, trip_ids: list[str] = None
    ) -> pl.LazyFrame:
        """
        Read and preprocess stop_times.txt into a Polars LazyFrame.

        This includes:
        - Optional filtering by stop_id and trip_id
        - Normalizing time strings
        - Computing seconds from midnight for times
        - Interpolating missing time values
        - Formatting seconds back into HH:MM:SS strings
        - Handling 24+ hour times

        Args:
            stop_ids (list[str], optional): Filter stop_times by these stop_ids.
            trip_ids (list[str], optional): Filter stop_times by these trip_ids.

        Returns:
            pl.LazyFrame: Preprocessed stop_times data.
        """
        schema_dict = utils.get_df_schema_dict(self.path)
        stop_times = utils.read_csv_list(self.path, schema_overrides=schema_dict)

        # Filter by stop_ids if provided
        if stop_ids:
            stop_ids_df = pl.DataFrame({"stop_id": stop_ids})
            stop_times = stop_times.join(stop_ids_df.lazy(), on="stop_id", how="inner")

        # Filter by trip_ids if provided
        if trip_ids:
            trip_ids_df = pl.DataFrame({"trip_id": trip_ids})
            stop_times = stop_times.join(trip_ids_df.lazy(), on="trip_id", how="inner")

        # Normalize time strings to HH:MM:SS
        stop_times = stop_times.with_columns(
            [
                ("0" + pl.col("arrival_time").cast(str))
                .str.slice(-8, 8)
                .alias("arrival_time"),
                ("0" + pl.col("departure_time").cast(str))
                .str.slice(-8, 8)
                .alias("departure_time"),
            ]
        )

        # Fill missing times by copying from the other time if one is null
        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("departure_time").is_null())
                .then(pl.col("arrival_time"))
                .otherwise(pl.col("departure_time"))
                .alias("departure_time"),
                pl.when(pl.col("arrival_time").is_null())
                .then(pl.col("departure_time"))
                .otherwise(pl.col("arrival_time"))
                .alias("arrival_time"),
            ]
        )

        # Convert times to seconds since midnight
        stop_times = stop_times.with_columns(
            [
                (
                    pl.col("departure_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("departure_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("departure_time").str.slice(6, 2).cast(int)
                ).alias("departure_time_secs"),
                (
                    pl.col("arrival_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("arrival_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("arrival_time").str.slice(6, 2).cast(int)
                ).alias("arrival_time_secs"),
            ]
        ).sort("trip_id", "stop_sequence")

        # Decide interpolation strategy based on shape_dist_traveled presence and validity
        if "shape_dist_traveled" in stop_times.collect_schema().names():
            # Check for nulls or negative values in shape_dist_traveled
            check_df = (
                stop_times.filter(
                    pl.col("shape_dist_traveled").is_null()
                    | (pl.col("shape_dist_traveled") < 0)
                )
                .select(pl.col("shape_dist_traveled"))
                .limit(1)  # Only need to find one problematic row to fallback
                .collect()
            )
            if check_df.height > 0:
                # Problematic shape_dist_traveled, fallback to default interpolation
                stop_times = self.__interpolate_times(stop_times)
            else:
                # Use shape_dist_traveled based proportional interpolation
                stop_times = self.__interpolate_times_with_shape_dist(stop_times)
        else:
            # No shape_dist_traveled column, use default interpolation
            stop_times = self.__interpolate_times(stop_times)

        # Convert seconds back to HH:MM:SS strings
        stop_times = stop_times.with_columns(
            [
                self.to_hhmmss("departure_time_secs", "departure_time"),
                self.to_hhmmss("arrival_time_secs", "arrival_time"),
            ]
        )

        # Handle times greater than 24 hours by wrapping around
        stop_times = stop_times.with_columns(
            pl.when(pl.col("departure_time_secs") >= 24 * 3600)
            .then(pl.col("departure_time_secs") - 24 * 3600)
            .otherwise(pl.col("departure_time_secs"))
            .alias("departure_time_secs_24")
        )

        return stop_times

    def to_hhmmss(self, field: str, new_field: str) -> pl.Expr:
        """
        Convert seconds since midnight to hh:mm:ss formatted string.

        Args:
            field (str): Column name containing seconds.
            new_field (str): Name of the resulting hh:mm:ss string column.

        Returns:
            pl.Expr: Polars expression for hh:mm:ss conversion.
        """
        seconds_expr = pl.col(field)
        hours = ("0" + (seconds_expr // 3600).cast(pl.Int32).cast(str)).str.slice(-2, 2)
        minutes = (
            "0" + ((seconds_expr % 3600) // 60).cast(pl.Int32).cast(str)
        ).str.slice(-2, 2)
        seconds = ("0" + (seconds_expr % 60).cast(pl.Int32).cast(str)).str.slice(-2, 2)
        return (hours + ":" + minutes + ":" + seconds).alias(new_field)
