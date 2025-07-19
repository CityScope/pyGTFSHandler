import polars as pl
from pathlib import Path
from typing import Union, List, Tuple, Optional
import utils
from datetime import datetime, time
from functools import reduce
import warnings
import operator


class StopTimes:
    """
    Manage GTFS stop_times.txt data using Polars LazyFrames.

    Features:
    - Filter by stop_ids and/or trip_ids
    - Normalize time strings
    - Convert time to/from seconds since midnight
    - Interpolate missing times
    - Return a fully processed LazyFrame ready for analysis
    """

    def __init__(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        start_time: datetime = None,
        end_time: datetime = None,
        stop_ids: List[str] = None,
        trip_ids: List[str] = None,
    ):
        """
        Initialize the StopTimes instance.

        Args:
            path (str | Path | list[str | Path]): One or more paths to directories containing stop_times.txt files.
            stop_ids (list[str], optional): List of stop IDs to filter on.
            trip_ids (list[str], optional): List of trip IDs to filter on.
        """
        if isinstance(path, (str, Path)):
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]

        self.lf = self.__read_stop_times(stop_ids, trip_ids)
        self.frequencies = self.__read_frequencies(trip_ids)
        if start_time and end_time:
            self.lf, self.frequencies = self.filter_by_time_bounds(start_time, end_time)

        # if (start_time and end_time) or stop_ids or trip_ids:
        #    self.trip_ids = self.lf.select("trip_id").unique().collect()["trip_id"].to_list()
        # else:
        #    self.trip_ids = trip_ids

    def __read_stop_times(
        self, stop_ids: List[str] = None, trip_ids: List[str] = None
    ) -> pl.LazyFrame:
        """
        Read and preprocess stop_times.txt files into a Polars LazyFrame.

        This includes:
        - Optional filtering by stop_id and trip_id
        - Normalizing time strings
        - Filling missing times
        - Ensuring times are formatted to HH:MM:SS

        Args:
            stop_ids (list[str], optional): Filter stop_times by these stop_ids.
            trip_ids (list[str], optional): Filter stop_times by these trip_ids.

        Returns:
            pl.LazyFrame: Preprocessed stop_times data.
        """
        stop_times_paths = [
            p / "stop_times.txt" for p in self.paths if (p / "stop_times.txt").exists()
        ]
        if not stop_times_paths:
            raise FileNotFoundError("No stop_times.txt files found in given paths.")

        schema_dict = utils.get_df_schema_dict(stop_times_paths[0])
        stop_times = utils.read_csv_list(stop_times_paths, schema_overrides=schema_dict)

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
        )

        stop_times = stop_times.with_columns(
            pl.when(
                (pl.col("departure_time_secs") >= 86400)
                | (pl.col("arrival_time_secs") >= 86400)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("next_day")
        )

        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("departure_time_secs") >= 86400)
                .then(pl.col("departure_time_secs") - 86400)
                .otherwise(pl.col("departure_time_secs"))
                .alias("departure_time_secs"),
                pl.when(pl.col("arrival_time_secs") >= 86400)
                .then(pl.col("arrival_time_secs") - 86400)
                .otherwise(pl.col("arrival_time_secs"))
                .alias("arrival_time_secs"),
            ]
        )

        return stop_times

    def __read_frequencies(
        self, trip_ids: Optional[List[str]] = None
    ) -> Optional[pl.LazyFrame]:
        """
        Reads and processes GTFS `frequencies.txt` files from all available paths.

        - Parses `start_time` and `end_time` to seconds since midnight.
        - Converts suspiciously low `headway_secs` (likely in minutes) to seconds.
        - Handles trips that span midnight by splitting them into two intervals.
        - Optionally filters by specific trip IDs.

        Args:
            trip_ids (Optional[List[str]]): List of `trip_id`s to filter by. If None, no filtering is applied.

        Returns:
            Optional[pl.LazyFrame]: A LazyFrame of parsed and cleaned frequencies, or None if no file is found.
        """
        # Locate all available frequencies.txt files
        frequencies_paths = [
            p / "frequencies.txt"
            for p in self.paths
            if (p / "frequencies.txt").exists()
        ]
        if not frequencies_paths:
            return None

        # Read schema and load all frequencies.csv files into one LazyFrame
        schema_dict = utils.get_df_schema_dict(frequencies_paths[0])
        frequencies = utils.read_csv_list(
            frequencies_paths, schema_overrides=schema_dict
        )

        # Filter to only specified trip_ids, if provided
        if trip_ids:
            trip_ids_df = pl.DataFrame({"trip_id": trip_ids})
            frequencies = frequencies.join(
                trip_ids_df.lazy(), on="trip_id", how="inner"
            )

        # Normalize time strings to HH:MM:SS
        frequencies = frequencies.with_columns(
            [
                ("0" + pl.col("start_time").cast(str))
                .str.slice(-8, 8)
                .alias("start_time"),
                ("0" + pl.col("end_time").cast(str)).str.slice(-8, 8).alias("end_time"),
            ]
        )

        # Parse start_time and end_time into seconds since midnight
        frequencies = frequencies.with_columns(
            [
                (
                    pl.col("start_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("start_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("start_time").str.slice(6, 2).cast(int)
                ).alias("start_time_secs"),
                (
                    pl.col("end_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("end_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("end_time").str.slice(6, 2).cast(int)
                ).alias("end_time_secs"),
            ]
        )

        # Identify gtfs_name groups where any headway_secs is suspiciously small (< 20)
        suspicious_names = (
            frequencies.filter(pl.col("headway_secs") < 20)
            .select("gtfs_name")
            .unique()
            .collect()
            .get_column("gtfs_name")
            .to_list()
        )

        # Warn if any such groups are found
        if suspicious_names:
            warnings.warn(
                f"GTFS files with possibly incorrect 'headway_secs' (likely in minutes instead of seconds): {suspicious_names}"
            )

        # Convert headway_secs to seconds for suspicious groups
        frequencies = frequencies.with_columns(
            pl.when(pl.col("gtfs_name").is_in(suspicious_names))
            .then(pl.col("headway_secs") * 60)
            .otherwise(pl.col("headway_secs"))
            .alias("headway_secs")
        )

        frequencies = frequencies.with_columns(
            pl.when(
                (pl.col("start_time_secs") >= 86400)
                & (pl.col("end_time_secs") >= 86400)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("next_day")
        )

        frequencies = frequencies.with_columns(
            [
                pl.when(
                    (pl.col("start_time_secs") >= 86400)
                    & (pl.col("end_time_secs") >= 86400)
                )
                .then(pl.col("start_time_secs") - 86400)
                .otherwise(pl.col("start_time_secs"))
                .alias("start_time_secs"),
                pl.when(
                    (pl.col("start_time_secs") >= 86400)
                    & (pl.col("end_time_secs") >= 86400)
                )
                .then(pl.col("end_time_secs") - 86400)
                .otherwise(pl.col("end_time_secs"))
                .alias("end_time_secs"),
            ]
        )

        # Clamp end_time_secs to 23:59:59 (86399) if it equals 86400 (i.e. 24:00:00)
        frequencies = frequencies.with_columns(
            pl.when(pl.col("end_time_secs") == 86400)
            .then(86399)
            .otherwise(pl.col("end_time_secs"))
            .alias("end_time_secs")
        )

        # Identify trips that cross midnight (end_time_secs < start_time_secs or invalid > 86400)
        spans_midnight = frequencies.filter(
            (pl.col("end_time_secs") < pl.col("start_time_secs"))
            | (pl.col("end_time_secs") >= 86400)
        )

        # First part: From original start to midnight
        first_half = spans_midnight.with_columns(pl.lit(86399).alias("end_time_secs"))

        # Second part: From midnight to original end
        second_half = spans_midnight.with_columns(
            pl.lit(0).alias("start_time_secs"), pl.lit(True).alias("next_day")
        )

        # Duplicate and merge the midnight-splitting intervals
        duplicated_rows = pl.concat([first_half, second_half], how="vertical_relaxed")

        # Keep trips that do not cross midnight and are valid
        normal_rows = frequencies.filter(
            (pl.col("end_time_secs") >= pl.col("start_time_secs"))
            & (pl.col("end_time_secs") < 86400)
        )

        # Final cleaned LazyFrame
        frequencies = pl.concat([normal_rows, duplicated_rows], how="vertical_relaxed")

        return frequencies

    def filter_by_time_bounds(
        self, start_time: datetime, end_time: datetime
    ) -> tuple[pl.LazyFrame | None, pl.LazyFrame]:
        """
        Filter stop_times by a time interval between start_time and end_time.

        Handles intervals that cross midnight (e.g., 22:00 to 03:00 next day).

        Args:
            start_time (datetime): Start datetime of the filter interval.
            end_time (datetime): End datetime of the filter interval.

        Returns:
            tuple:
                filtered_stop_times (pl.LazyFrame): Filtered stop_times LazyFrame within the time interval,
                                                    including trips with valid frequencies overlapping the interval.
                filtered_frequencies (pl.LazyFrame | None): Frequencies overlapping the time interval or None if frequencies data is missing.
        """

        def time_to_seconds(t: time) -> int:
            return t.hour * 3600 + t.minute * 60 + t.second

        start_secs = time_to_seconds(start_time.time())
        end_secs = time_to_seconds(end_time.time())

        # If frequencies is None, return stop_times filtered only by arrival_time_secs
        if self.frequencies is None:
            if start_time.date() == end_time.date():
                filtered_stop_times = self.lf.filter(
                    (pl.col("arrival_time_secs") >= start_secs)
                    & (pl.col("arrival_time_secs") <= end_secs)
                )
            else:
                raise ValueError("Start and end datetime must be on the same date")

            return filtered_stop_times, None

        # Frequencies is available: proceed
        if start_time.date() == end_time.date():
            filtered_frequencies = self.frequencies.filter(
                (pl.col("start_time_secs") <= end_secs)
                & (pl.col("end_time_secs") >= start_secs)
            )

            filtered_frequencies = filtered_frequencies.with_columns(
                [
                    pl.when(pl.col("end_time_secs") > end_secs)
                    .then(end_secs)
                    .otherwise(pl.col("end_time_secs"))
                    .alias("end_time_secs"),
                    pl.when(pl.col("start_time_secs") < start_secs)
                    .then(start_secs)
                    .otherwise(pl.col("start_time_secs"))
                    .alias("start_time_secs"),
                ]
            )

            valid_trip_ids = filtered_frequencies.select("trip_id").unique()

            filtered_stop_times = self.lf.filter(
                (
                    (pl.col("arrival_time_secs") >= start_secs)
                    & (pl.col("arrival_time_secs") <= end_secs)
                )
                | (pl.col("trip_id").is_in(valid_trip_ids))
            )

        else:
            raise ValueError("Start and end datetime must be on the same date")

        return filtered_stop_times, filtered_frequencies

    def filter_by_multi_time_bounds(
        self, time_bounds: List[Tuple[datetime, datetime]]
    ) -> Tuple[Optional[pl.LazyFrame], Optional[pl.LazyFrame]]:
        """
        Filter stop_times and frequencies lazily by multiple time intervals.

        Each interval must be on the same date (no cross-midnight support here).

        Args:
            time_bounds: List of (start_datetime, end_datetime) tuples.

        Returns:
            Tuple of filtered_stop_times and filtered_frequencies LazyFrames.
        """

        def time_to_seconds(t: time) -> int:
            return t.hour * 3600 + t.minute * 60 + t.second

        # Validate intervals and extract start_secs and end_secs
        intervals = []
        for start_dt, end_dt in time_bounds:
            if start_dt.date() != end_dt.date():
                raise ValueError("Each interval must be within the same date")
            intervals.append(
                (time_to_seconds(start_dt.time()), time_to_seconds(end_dt.time()))
            )

        if self.frequencies is None:
            # No frequencies: only filter stop_times arrival_time_secs for any interval

            # Build a single filter expression for stop_times combining all intervals with OR
            arrival_filters = [
                (pl.col("arrival_time_secs") >= start_secs)
                & (pl.col("arrival_time_secs") <= end_secs)
                for start_secs, end_secs in intervals
            ]
            combined_arrival_filter = reduce(operator.or_, arrival_filters)

            filtered_stop_times = self.lf.filter(combined_arrival_filter)
            return filtered_stop_times, None

        # Frequencies exist: build combined filter for frequencies with OR

        freq_filters = [
            (pl.col("start_time_secs") <= end_secs)
            & (pl.col("end_time_secs") >= start_secs)
            for start_secs, end_secs in intervals
        ]
        combined_freq_filter = reduce(operator.or_, freq_filters)

        filtered_frequencies = self.frequencies.filter(combined_freq_filter)

        # For clipping frequencies start_time_secs and end_time_secs, create expressions that clamp
        # each frequency interval by the min start_secs and max end_secs it overlaps with.

        # For each interval, create expressions for clipped start and end times:
        start_clips = [
            pl.when(pl.col("start_time_secs") < start_secs)
            .then(start_secs)
            .otherwise(pl.col("start_time_secs"))
            for start_secs, _ in intervals
        ]
        # Min of all start_clips (the maximum among all start_secs)
        clipped_start = reduce(
            lambda acc, expr: pl.when(expr > acc).then(expr).otherwise(acc),
            start_clips,
            pl.lit(0),
        )

        end_clips = [
            pl.when(pl.col("end_time_secs") > end_secs)
            .then(end_secs)
            .otherwise(pl.col("end_time_secs"))
            for _, end_secs in intervals
        ]
        # Max of all end_clips (the minimum among all end_secs)
        clipped_end = reduce(
            lambda acc, expr: pl.when(expr < acc).then(expr).otherwise(acc),
            end_clips,
            pl.lit(86399),
        )

        # Apply clipping columns
        filtered_frequencies = filtered_frequencies.with_columns(
            [
                clipped_start.alias("start_time_secs"),
                clipped_end.alias("end_time_secs"),
            ]
        )

        # Get unique trip_ids from filtered frequencies
        valid_trip_ids = filtered_frequencies.select("trip_id").unique()

        # Build combined stop_times arrival_time_secs filter (OR all intervals)
        arrival_filters = [
            (pl.col("arrival_time_secs") >= start_secs)
            & (pl.col("arrival_time_secs") <= end_secs)
            for start_secs, end_secs in intervals
        ]
        combined_arrival_filter = reduce(operator.or_, arrival_filters)

        # Filter stop_times by arrival_time OR trip_id in valid_trip_ids
        filtered_stop_times = self.lf.filter(
            combined_arrival_filter | pl.col("trip_id").is_in(valid_trip_ids)
        )

        return filtered_stop_times, filtered_frequencies

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

    def fill_times(self):
        stop_times = self.lf
        # Convert times to seconds since midnight

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
