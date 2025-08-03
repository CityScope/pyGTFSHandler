# -*- coding: utf-8 -*-
import polars as pl
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from .. import utils
from datetime import datetime, time
import warnings

"""
GTFS StopTimes Data Processing Module

This module provides the `StopTimes` class, a powerful tool for processing and analyzing
General Transit Feed Specification (GTFS) data, specifically focusing on `stop_times.txt`
and its relationship with `frequencies.txt`.

What is performed by this module:
---------------------------------
The `StopTimes` class is designed to load, clean, and enrich GTFS stop time data
using the high-performance Polars library for lazy evaluation. Its primary capabilities include:

1.  **Data Loading**: It reads one or more `stop_times.txt` files from specified GTFS directories.
    It can also optionally load corresponding `frequencies.txt` files.

2.  **Filtering**: The data can be filtered at initialization by `stop_ids`, `trip_ids`, or
    a specific time window (`start_time`, `end_time`). More complex time-window filtering can
    also be applied after initialization using the `filter_by_time_range` and
    `filter_by_multi_time_bounds` methods.

3.  **Data Cleaning and Correction**:
    - **Time Normalization**: It standardizes time strings into a consistent 'HH:MM:SS' format
      and then converts them into "seconds since midnight". This normalizes all times to a 0-24 hour range (0-86399 seconds).
      Services that pass midnight are explicitly marked in the `next_day` column as `True`.
    - **Sequence Correction**: If the `stop_sequence` column is missing or contains nulls
      for a trip, it reconstructs a valid, zero-indexed sequence based on the original
      file order to ensure trip integrity.
    - **Missing Time Interpolation**: It handles null `arrival_time` or `departure_time` values
      using a simple forward-fill strategy within a trip. A more complex interpolation, which
      can leverage `shapes.txt`, is available in the main `Feed` class.
    - **Headway Correction**: It detects a common data quality issue in `frequencies.txt` where
      `headway_secs` is incorrectly provided in minutes. It identifies these cases, converts
      the values to seconds, and issues a warning.

4.  **Feature Enrichment**:
    - **Travel Time Calculation**: It computes the cumulative travel time along a trip's
      path (`shape_time_traveled`) and the total trip duration (`shape_total_travel_time`).
    - **Midnight Crossing Detection**: It robustly identifies trips that span past midnight,
      both from explicit GTFS time notation (e.g., '25:30:00') and by detecting when
      time decreases between consecutive stops.

5.  **Frequency-Based Trip Expansion**: This module expands frequency definitions into explicit trips,
    particularly for services that may cross midnight. When a trip's duration could cause it to end
    on the next calendar day, its `next_day` status would vary for each departure time, which cannot
    be expressed by a single frequency rule. Therefore, the module generates explicit, time-shifted
    trip records for these cases. Original template trip IDs are stored in the `orig_trip_id` column,
    and new unique IDs are generated. These new trips are prepared for integration with the main
    trips data in the `Feed` class.

6.  **Shape Generation**: The `generate_shape_ids` method generates new, canonical `shape_id`s by
    grouping trips. A "shape" is defined as all trips that share the exact same sequence of stops
    and have a total travel duration within a 2-minute tolerance (configured by `TRIP_ROUND_TIME`).
    This is useful for analyzing route patterns.

The final output of the class is a clean, comprehensive Polars LazyFrame (`self.lf`) containing
all stop times, including those generated from frequencies, ready for advanced analysis.
"""

"TODO: Check if stop_times already has n_trips in the time bounds delete the frequency and warn. "
"Check that n_trips in frequency has in stop times no other trip_ids from the same route"

# A constant used for rounding trip travel times when generating shape_ids.
# It groups trips with travel times within a 5-minute (300s) window.
TRIP_ROUND_TIME: int = 600
SECS_PER_DAY: int = 86400


class StopTimes:
    """
    Manages and processes GTFS stop_times.txt and frequencies.txt data.

    This class provides a comprehensive pipeline for reading, cleaning, and enriching
    stop time data from GTFS feeds. It leverages Polars LazyFrames for efficient,
    memory-friendly processing.

    Key Features:
        - Reads and combines data from multiple GTFS sources.
        - Filters data by `stop_ids`, `trip_ids`, and time windows.
        - Normalizes GTFS times to seconds within a 0-24 hour day, using a `next_day`
          flag for services crossing midnight.
        - Corrects invalid or missing `stop_sequence` values.
        - Interpolates missing arrival/departure times (a more complex interpolation
          can be done together with the shapes file in the feed class).
        - Calculates cumulative and total travel times for each trip.
        - Expands frequency-based services into explicit stop times, creating new
          trip records for analysis.
        - Generates canonical shape IDs by grouping trips with identical stop
          patterns and similar travel times.

    Attributes:
        paths (List[Path]): A list of Path objects to the GTFS directories.
        lf (pl.LazyFrame): The main LazyFrame containing the processed stop times data.
        fixed_times (bool): A flag indicating if any stop times were interpolated.
        frequencies (Optional[pl.LazyFrame]): A LazyFrame for processed frequencies data,
                                               or None if not present.
    """

    def __init__(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        trips,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        stop_ids: Optional[List[str]] = None,
        trip_ids: Optional[List[str]] = None,
    ):
        """
        Initializes the StopTimes instance and runs the processing pipeline.

        Args:
            path (Union[str, Path, List[Union[str, Path]]]):
                A single path or a list of paths to GTFS directories.
            start_time (Optional[datetime]):
                The start of a time window for initial filtering. If provided,
                `end_time` must also be provided.
            end_time (Optional[datetime]):
                The end of a time window for initial filtering. If provided,
                `start_time` must also be provided.
            stop_ids (Optional[List[str]]):
                A list of `stop_id`s to filter the data upon loading. Only stop times
                associated with these stops will be processed.
            trip_ids (Optional[List[str]]):
                A list of `trip_id`s to filter the data upon loading. Only stop times
                associated with these trips will be processed.
        """
        # Standardize input path(s) to a list of Path objects
        if isinstance(path, (str, Path)):
            paths: List[Path] = [Path(path)]
        else:
            paths: List[Path] = [Path(p) for p in path]

        # --- Main Processing Pipeline ---
        self.lf: pl.LazyFrame = self.__read_stop_times(paths, trip_ids)
        self.lf = self.__correct_sequence(self.lf)
        self.lf, self.fixed_times = self.__fix_nulls_easy(self.lf)
        self.lf = self.__normalize_times(self.lf)

        if stop_ids:
            self.lf = self.__filter_by_stop_id(self.lf, stop_ids)
            self.lf = self.__correct_sequence(self.lf)
            self.lf = self.lf.collect().lazy()

        if self.fixed_times:
            warnings.warn("Some departure times are null and have been interpolated")

        self.frequencies: Optional[pl.LazyFrame] = self.__read_frequencies(
            paths, trip_ids
        )

        if self.frequencies is not None:
            self.lf = self.lf.collect().lazy()

            self.frequencies = self.frequencies.join(self.lf, on="trip_id", how="semi")
            self.frequencies = self.__filter_repeated_frequencies_with_trips(
                self.frequencies, trips
            )
            self.frequencies = self.frequencies.collect().lazy()
            self.frequencies = self._frequencies_midnight_crossing(self.frequencies)
            self.lf, self.frequencies = self.__check_frequencies_in_stop_times(
                self.lf, self.frequencies
            )

            self.lf = self.lf.collect().lazy()
            self.frequencies = self.frequencies.collect().lazy()

        self.lf = self._add_shape_time_and_midnight_crossing(self.lf)

        if self.frequencies is not None:
            self.frequencies = self._add_departure_time_to_frequencies(
                self.lf, self.frequencies
            )

        if start_time and end_time:
            self.lf = self.lf.collect().lazy()
            self.frequencies = self.frequencies.collect().lazy()

            self.lf, self.frequencies = self.filter_by_time_range(
                start_time, end_time, strict=False
            )

            self.frequencies = self.frequencies.collect().lazy()

        # Eagerly evaluate to checkpoint the result and optimize subsequent queries
        self.lf = self.lf.collect().lazy()

        if self.frequencies is not None:
            self.frequencies = self.__fix_headway(self.frequencies)
            self.lf, self.frequencies = self._midnight_frequencies_to_stop_times(
                self.lf, self.frequencies
            )
            if start_time and end_time:
                self.frequencies = self.frequencies.filter(
                    (pl.col("start_time") < utils.time_to_seconds(end_time))
                    & (pl.col("end_time") > utils.time_to_seconds(start_time))
                )

            # Eagerly evaluate frequencies to checkpoint results
            self.frequencies = self.frequencies.collect().lazy()

            self.frequencies = self._add_frequencies_n_trips(self.frequencies)

            # Recalculate travel times as new trips may have been added
            self.lf = self._add_shape_time_and_midnight_crossing(self.lf)
            self.lf = self.lf.collect().lazy()

            if "orig_trip_id" in self.lf.collect_schema().names():
                unique_trip_ids: pl.LazyFrame = (
                    self.lf.select(["trip_id", "orig_trip_id"])
                    .unique()
                    .rename({"trip_id": "new_trip_id"})
                )

                # Join the trips table with this mapping. This duplicates the original
                # trip's data (route_id, service_id, etc.) for each new generated trip_id.
                trips = (
                    trips.join(
                        unique_trip_ids,
                        left_on="trip_id",
                        right_on="orig_trip_id",
                        how="right",  # Right join ensures all new trip_ids from stop_times are included.
                    )
                    .with_columns(pl.col("new_trip_id").alias("trip_id"))
                    .drop("new_trip_id", "orig_trip_id")
                )

                self.lf = self.lf.drop("orig_trip_id")

        self.trips_lf = pl.concat(
            [
                trips.with_columns(pl.lit(False).alias("next_day")),
                trips.with_columns(
                    (pl.col("service_id") + "_night").alias("service_id"),
                    pl.lit(True).alias("next_day"),
                ),
            ]
        )

    def __read_stop_times(
        self, paths, trip_ids: Optional[List[str]] = None
    ) -> pl.LazyFrame:
        """
        Reads and preprocesses `stop_times.txt` files into a Polars LazyFrame.

        This private method performs the initial loading and cleaning, which includes:
        - Locating and reading all `stop_times.txt` files.
        - Optionally filtering by `stop_id` and/or `trip_id` for efficiency.
        - Normalizing time strings to 'HH:MM:SS' format.
        - Filling null `arrival_time` or `departure_time` with the value from the other.
        - Converting time strings to integer seconds since midnight.

        Args:
            stop_ids (Optional[List[str]]): A list of stop IDs to filter by.
            trip_ids (Optional[List[str]]): A list of trip IDs to filter by.

        Returns:
            pl.LazyFrame: A LazyFrame containing preprocessed stop_times data with
                          times converted to seconds since midnight.

        Raises:
            FileNotFoundError: If no `stop_times.txt` files are found in the given paths.
        """
        stop_times_paths: List[Path] = [
            p / "stop_times.txt" for p in paths if (p / "stop_times.txt").exists()
        ]
        if not stop_times_paths:
            raise FileNotFoundError("No stop_times.txt files found in given paths.")

        schema_dict: Dict[str, pl.DataType] = utils.get_df_schema_dict(
            stop_times_paths[0]
        )
        stop_times: pl.LazyFrame = utils.read_csv_list(
            stop_times_paths, schema_overrides=schema_dict
        )

        if trip_ids:
            trip_ids_df = pl.LazyFrame({"trip_id": trip_ids})
            stop_times = stop_times.join(trip_ids_df, on="trip_id", how="semi")

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
                ).alias("departure_time"),
                (
                    pl.col("arrival_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("arrival_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("arrival_time").str.slice(6, 2).cast(int)
                ).alias("arrival_time"),
                pl.lit(False).alias("next_day"),
            ]
        )
        return stop_times

    def __correct_sequence(self, stop_times: pl.LazyFrame) -> pl.LazyFrame:
        """
        Ensures all trips have a valid and continuous `stop_sequence`.

        If `stop_sequence` is missing or contains null values for any part of a trip,
        this method replaces it with a new, zero-indexed sequence based on the
        original row order within that trip.

        Args:
            stop_times (pl.LazyFrame): The input LazyFrame of stop times.

        Returns:
            pl.LazyFrame: A LazyFrame with a corrected `stop_sequence` column.
        """
        if "stop_sequence" not in stop_times.collect_schema().names():
            stop_times = stop_times.with_columns(
                pl.lit(None).cast(pl.Int64).alias("stop_sequence")
            )

        stop_times = stop_times.with_row_index("original_idx")

        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("stop_sequence").is_null().any().over("trip_id"))
                .then(None)
                .otherwise(pl.col("stop_sequence"))
                .alias("stop_sequence")
            ]
        ).with_columns(
            [
                pl.when(pl.col("stop_sequence").is_nan().any().over("trip_id"))
                .then(None)
                .otherwise(pl.col("stop_sequence"))
                .alias("stop_sequence")
            ]
        )

        stop_times = (
            stop_times.sort("trip_id", "stop_sequence", "original_idx")
            .with_columns(
                (pl.arange(0, pl.count()).over("trip_id")).alias("stop_sequence")
            )
            .drop("original_idx")
        )

        return stop_times

    def __normalize_times(self, stop_times):
        stop_times = stop_times.with_columns(
            pl.when(
                (pl.col("departure_time") >= SECS_PER_DAY)
                | (pl.col("arrival_time") >= SECS_PER_DAY)
            )
            .then(pl.lit(True))
            .otherwise(pl.col("next_day"))
            .alias("next_day")
        )

        stop_times = stop_times.with_columns(
            (pl.col("departure_time") % SECS_PER_DAY).alias("departure_time"),
            (pl.col("arrival_time") % SECS_PER_DAY).alias("arrival_time"),
        )
        return stop_times

    def __filter_by_stop_id(
        self, stop_times: pl.LazyFrame, stop_ids: list[str]
    ) -> pl.LazyFrame:
        # Create stop_ids as a lazy frame
        stop_ids_lf = pl.LazyFrame({"stop_id": stop_ids})

        # Select matching stop_times with just needed columns
        stop_times_filter = stop_times.select(
            ["trip_id", "stop_id", "stop_sequence"]
        ).join(stop_ids_lf, on="stop_id", how="semi")

        # Use a window function to compute max(stop_sequence) per trip,
        # then increment it and keep only one row per trip
        next_stop = (
            stop_times_filter.with_columns(
                pl.max("stop_sequence").over("trip_id").alias("max_seq")
            )
            .filter(pl.col("stop_sequence") == pl.col("max_seq"))
            .with_columns((pl.col("stop_sequence") + 1).alias("stop_sequence"))
            .select(["trip_id", "stop_sequence"])
        )

        # Append the new stop row to stop_times
        stop_times_filter = pl.concat(
            [stop_times_filter.select(["trip_id", "stop_sequence"]), next_stop]
        )

        # Filter stop_times by matching trip_id and stop_sequence
        stop_times = stop_times.join(
            stop_times_filter, on=["trip_id", "stop_sequence"], how="semi"
        )
        return stop_times

    def __fix_nulls_easy(self, stop_times: pl.LazyFrame) -> Tuple[pl.LazyFrame, bool]:
        """
        Interpolates null departure and arrival times using linear interpolation.

        This method handles cases where intermediate stops in a trip have null times.
        It flags rows where times were interpolated.

        Args:
            stop_times (pl.LazyFrame): The input LazyFrame of stop times.

        Returns:
            Tuple[pl.LazyFrame, bool]: A tuple containing:
                - The LazyFrame with times interpolated.
                - A boolean, `True` if any times were fixed, `False` otherwise.
        """
        has_nulls_expr = (
            pl.col("departure_time").is_null().any()
            | pl.col("departure_time").is_nan().any()
        )
        has_nulls: bool = stop_times.select(has_nulls_expr).collect().item()

        if has_nulls:
            stop_times = stop_times.sort(["trip_id", "stop_sequence"]).with_columns(
                (
                    pl.col("departure_time").is_null()
                    | pl.col("departure_time").is_nan()
                ).alias("fixed_time"),
                # Linear interpolation per trip
                pl.col("departure_time")
                .interpolate(method="linear")
                .over("trip_id")
                .alias("departure_time"),
            )

            stop_times = stop_times.with_columns(
                [
                    pl.when(
                        pl.col("arrival_time").is_null()
                        | pl.col("arrival_time").is_nan()
                    )
                    .then(pl.col("departure_time"))
                    .otherwise(pl.col("arrival_time"))
                    .alias("arrival_time"),
                ]
            )
            return stop_times, True
        else:
            stop_times = stop_times.with_columns(pl.lit(False).alias("fixed_time"))
            return stop_times, False

    def _add_shape_time_and_midnight_crossing(
        self, stop_times: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Calculates travel time and detects midnight crossings for each trip.

        This method enriches the stop times data with:
        - `next_day`: A boolean flag, true if the stop occurs after midnight.
        - `shape_time_delta`: Time in seconds between a stop and the previous one.
        - `shape_time_traveled`: Cumulative time in seconds from the start of the trip.
        - `shape_total_travel_time`: The total duration of the trip in seconds.

        It handles midnight crossings by checking for GTFS times > 24:00:00 and also
        by detecting when time decreases between consecutive stops.

        Args:
            stop_times (pl.LazyFrame): The input LazyFrame of stop times.

        Returns:
            pl.LazyFrame: The enriched LazyFrame with new time-related columns.
        """
        stop_times = (
            stop_times.sort(["trip_id", "stop_sequence"])
            .with_columns(
                [
                    pl.col("departure_time")
                    .shift(1)
                    .over("trip_id")
                    .alias("prev_departure_time")
                ]
            )
            .with_columns(
                [
                    (pl.col("departure_time") - pl.col("prev_departure_time"))
                    .fill_null(0)
                    .alias("shape_time_delta")
                ]
            )
            .with_columns(
                [
                    (
                        ((pl.col("shape_time_delta") < 0).cum_sum().over("trip_id") > 0)
                        | pl.col("next_day")
                    ).alias("next_day")
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("shape_time_delta") < 0)
                    .then(pl.col("shape_time_delta") + SECS_PER_DAY)
                    .otherwise(pl.col("shape_time_delta"))
                    .alias("shape_time_delta")
                ]
            )
            .with_columns(
                [
                    pl.col("shape_time_delta")
                    .cum_sum()
                    .over("trip_id")
                    .alias("shape_time_traveled")
                ]
            )
            # .drop("prev_departure_time")#,"shape_time_delta")
        )

        total_travel_times = stop_times.group_by("trip_id").agg(
            pl.col("shape_time_traveled").max().alias("shape_total_travel_time")
        )

        if "shape_total_travel_time" in stop_times.collect_schema().names():
            stop_times = stop_times.drop("shape_total_travel_time")

        stop_times = stop_times.join(total_travel_times, on="trip_id", how="left")

        return stop_times

    def __read_frequencies(
        self, paths, trip_ids: Optional[List[str]] = None
    ) -> Optional[pl.LazyFrame]:
        """
        Reads and processes GTFS `frequencies.txt` files from all available paths.

        - Parses `start_time` and `end_time` to seconds since midnight.
        - Optionally filters by specific trip IDs.

        Args:
            trip_ids (Optional[List[str]]): A list of `trip_id`s to filter by.
                                             If None, no filtering is applied.

        Returns:
            Optional[pl.LazyFrame]: A LazyFrame of parsed frequencies,
                                    or None if no `frequencies.txt` file is found.
        """
        frequencies_paths: List[Path] = [
            p / "frequencies.txt" for p in paths if (p / "frequencies.txt").exists()
        ]
        if not frequencies_paths:
            return None

        schema_dict: Dict[str, pl.DataType] = utils.get_df_schema_dict(
            frequencies_paths[0]
        )
        frequencies: pl.LazyFrame = utils.read_csv_list(
            frequencies_paths, schema_overrides=schema_dict
        )

        if trip_ids:
            trip_ids_df = pl.LazyFrame({"trip_id": trip_ids})
            frequencies = frequencies.join(trip_ids_df, on="trip_id", how="semi")

        frequencies = frequencies.with_columns(
            [
                ("0" + pl.col("start_time").cast(str))
                .str.slice(-8, 8)
                .alias("start_time"),
                ("0" + pl.col("end_time").cast(str)).str.slice(-8, 8).alias("end_time"),
            ]
        )

        frequencies = frequencies.with_columns(
            [
                (
                    pl.col("start_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("start_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("start_time").str.slice(6, 2).cast(int)
                ).alias("start_time"),
                (
                    pl.col("end_time").str.slice(0, 2).cast(int) * 3600
                    + pl.col("end_time").str.slice(3, 2).cast(int) * 60
                    + pl.col("end_time").str.slice(6, 2).cast(int)
                ).alias("end_time"),
                pl.lit(False).alias("next_day"),
                pl.col("trip_id").alias("orig_trip_id"),
            ]
        )

        return frequencies

    def __filter_repeated_frequencies_with_trips(self, frequencies, trips):
        cols = frequencies.collect_schema().names()
        frequencies = frequencies.join(
            trips.select(
                ["trip_id", "service_id", "trip_headsign", "direction_id", "shape_id"]
            ),
            on="trip_id",
            how="inner",
        ).unique(
            [
                "service_id",
                "trip_headsign",
                "direction_id",
                "shape_id",
                "start_time",
                "end_time",
                "headway_secs",
            ]
        )

        return frequencies.select(cols)

    def __check_frequencies_in_stop_times(self, stop_times, frequencies):
        frequencies = frequencies.join(
            stop_times,
            left_on="orig_trip_id",
            right_on="trip_id",
            how="semi",
        )

        stop_times_cols = stop_times.collect_schema().names()

        # Build frequencies with orig_trip_id and suffixed trip_id if duplicated
        frequencies = (
            frequencies.with_columns(
                [
                    pl.col("trip_id").count().over("trip_id").alias("trip_count"),
                    pl.col("trip_id").cum_count().over("trip_id").alias("suffix_index"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("trip_count") == 1)
                    .then(pl.col("trip_id"))
                    .otherwise(
                        pl.col("trip_id")
                        + "_frequency_"
                        + pl.col("suffix_index").cast(str)
                    )
                    .alias("trip_id")
                ]
            )
            .drop(["trip_count", "suffix_index"])
        )

        # Split stop_times:
        # 1. Those with trip_ids in frequencies (to be duplicated)
        # 2. Those without (to be preserved)

        # Join and replicate matching stop_times
        stop_times_matched = frequencies.join(
            stop_times.with_columns(pl.col("trip_id").alias("orig_trip_id")),
            on="orig_trip_id",
            how="inner",
        ).select(stop_times_cols + ["orig_trip_id"])

        # Preserve unmatched stop_times
        stop_times_unmatched = (
            stop_times.join(
                frequencies.select("orig_trip_id").unique(),
                left_on="trip_id",
                right_on="orig_trip_id",
                how="anti",
            )
            .with_columns([pl.col("trip_id").alias("orig_trip_id")])
            .select(stop_times_cols + ["orig_trip_id"])
        )

        # Final stop_times = matched + unmatched
        stop_times = pl.concat([stop_times_matched, stop_times_unmatched])

        return stop_times, frequencies

    def __fix_headway(self, frequencies: pl.LazyFrame) -> pl.LazyFrame:
        """
        Corrects `headway_secs` values that are likely in minutes instead of seconds.

        Some GTFS feeds incorrectly state headway in minutes. This method identifies
        feeds where any headway is suspiciously low (e.g., < 20) and multiplies all
        headways for that feed by 60.

        Args:
            frequencies (pl.LazyFrame): The frequencies LazyFrame.

        Returns:
            pl.LazyFrame: Frequencies LazyFrame with corrected `headway_secs`.
        """
        suspicious_names: List[str] = (
            frequencies.filter(pl.col("headway_secs") < 20)
            .select("gtfs_name")
            .unique()
            .collect()
            .get_column("gtfs_name")
            .to_list()
        )

        if suspicious_names:
            warnings.warn(
                f"GTFS files with possibly incorrect 'headway_secs' (likely in minutes instead of seconds): {suspicious_names}"
            )

        frequencies = frequencies.with_columns(
            pl.when(pl.col("gtfs_name").is_in(suspicious_names))
            .then(pl.col("headway_secs") * 60)
            .otherwise(pl.col("headway_secs"))
            .alias("headway_secs")
        )

        return frequencies

    def _frequencies_midnight_crossing(self, frequencies: pl.LazyFrame) -> pl.LazyFrame:
        """
        Handles frequency entries that span midnight.

        A frequency definition like 22:00 to 02:00 is split into two separate entries:
        1. 22:00 to 23:59:59 on the current day.
        2. 00:00 to 02:00 on the next day (marked with `next_day` = True).

        Args:
            frequencies (pl.LazyFrame): The frequencies LazyFrame.

        Returns:
            pl.LazyFrame: A LazyFrame with midnight-spanning frequencies properly split.
        """
        frequencies = frequencies.with_columns(
            pl.when(
                (pl.col("start_time") >= SECS_PER_DAY)
                & (pl.col("end_time") >= SECS_PER_DAY)
            )
            .then(pl.lit(True))
            .otherwise(pl.col("next_day"))
            .alias("next_day")
        )

        frequencies = frequencies.with_columns(
            [
                pl.when(
                    (pl.col("start_time") >= SECS_PER_DAY)
                    & (pl.col("end_time") >= SECS_PER_DAY)
                )
                .then(pl.col("start_time") % SECS_PER_DAY)
                .otherwise(pl.col("start_time"))
                .alias("start_time"),
                pl.when(
                    (pl.col("start_time") >= SECS_PER_DAY)
                    & (pl.col("end_time") >= SECS_PER_DAY)
                )
                .then(pl.col("end_time") % SECS_PER_DAY)
                .otherwise(pl.col("end_time"))
                .alias("end_time"),
            ]
        )

        frequencies = frequencies.with_columns(
            pl.when(pl.col("end_time") == SECS_PER_DAY)
            .then(SECS_PER_DAY - 1)
            .otherwise(pl.col("end_time"))
            .alias("end_time")
        )

        spans_midnight = frequencies.filter(
            (pl.col("end_time") < pl.col("start_time"))
            | (pl.col("end_time") >= SECS_PER_DAY)
        )

        first_half = spans_midnight.with_columns(pl.lit(SECS_PER_DAY).alias("end_time"))
        second_half = spans_midnight.with_columns(
            pl.lit(0).alias("start_time"),
            (pl.col("end_time") % SECS_PER_DAY).alias("end_time"),
            pl.lit(True).alias("next_day"),
            (
                pl.concat_str(
                    pl.col("trip_id"),
                    pl.lit("_night"),
                )
            ).alias("trip_id"),
        )

        duplicated_rows = pl.concat([first_half, second_half], how="vertical_relaxed")
        normal_rows = frequencies.filter(
            (pl.col("end_time") >= pl.col("start_time"))
            & (pl.col("end_time") < SECS_PER_DAY)
        )

        frequencies = pl.concat([normal_rows, duplicated_rows], how="vertical_relaxed")

        return frequencies

    def _add_frequencies_n_trips(self, frequencies: pl.LazyFrame) -> pl.LazyFrame:
        """
        Calculates the number of trips generated by each frequency entry.

        Args:
            frequencies (pl.LazyFrame): The frequencies LazyFrame.

        Returns:
            pl.LazyFrame: Frequencies LazyFrame with an added `n_trips` column.
        """
        frequencies = frequencies.with_columns(
            ((pl.col("end_time") - pl.col("start_time")) / pl.col("headway_secs"))
            .ceil()
            .cast(pl.UInt32)
            .alias("n_trips")
        ).filter(pl.col("n_trips") > 0)

        return frequencies

    def _add_departure_time_to_frequencies(
        self, stop_times: pl.LazyFrame, frequencies: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Adds trip-specific details to corresponding frequency entries.

        This method joins the frequencies table with the first stop of each
        trip template to get the base `departure_time` and the total travel time.
        This information is essential for later expanding the frequency definitions
        into explicit trip schedules.

        Args:
            stop_times (pl.LazyFrame): The main stop_times LazyFrame.
            frequencies (pl.LazyFrame): The frequencies LazyFrame to be enriched.

        Returns:
            pl.LazyFrame: The frequencies LazyFrame, now containing `departure_time`
                          and `shape_total_travel_time` for each trip template.
        """
        frequencies = frequencies.join(
            stop_times.filter(pl.col("stop_sequence") == 0).select(
                ["trip_id", "shape_total_travel_time", "departure_time"]
            ),
            on="trip_id",
            how="left",
        ).rename({"departure_time": "first_departure_time"})
        return frequencies

    def _midnight_frequencies_to_stop_times(
        self, stop_times: pl.LazyFrame, frequencies: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Expands frequency-based trips into explicit stop time records.

        This method is crucial for handling services defined in `frequencies.txt`.
        It specifically targets frequency definitions where a trip's duration could
        cause it to end after midnight. In such cases, the `next_day` status would
        change depending on the departure time, which cannot be represented by a
        single, simple frequency rule.

        The method works by:
        1. Identifying frequency entries for trips that might cross midnight.
        2. Calculating the series of departure time offsets (`delta_time`) needed to
           generate each individual trip within the frequency window.
        3. Creating new, time-shifted copies of the base trip's stop times for each offset.
        4. Assigning new, unique `trip_id`s to these generated trips, while preserving
           the original template ID in `orig_trip_id`.
        5. Adjusting the frequency window to exclude the now-explicitly-generated trips,
           preventing double counting.

        Args:
            stop_times (pl.LazyFrame): The main stop_times LazyFrame, containing trip templates.
            frequencies (pl.LazyFrame): The frequencies LazyFrame, which will be expanded.

        Returns:
            Tuple[pl.LazyFrame, pl.LazyFrame]: A tuple containing:
                - The updated stop_times LazyFrame with the newly generated trips.
                - The updated frequencies LazyFrame with adjusted time windows.
        """
        midnight_frequencies = frequencies.filter(
            pl.col("end_time") + pl.col("shape_total_travel_time") >= SECS_PER_DAY
        ).with_columns(
            [
                ((SECS_PER_DAY - pl.col("shape_total_travel_time")) - 1).alias(
                    "new_end_time"
                )
            ]
        )

        delta_times = midnight_frequencies.with_columns(
            [
                (
                    (
                        (pl.col("new_end_time") - pl.col("first_departure_time"))
                        / pl.col("headway_secs")
                    ).ceil()
                    * pl.col("headway_secs")
                    + pl.col("first_departure_time")
                ).alias("aligned_start")
            ]
        ).with_columns(
            [
                pl.int_ranges(
                    pl.col("aligned_start") - pl.col("first_departure_time"),
                    pl.col("end_time") + 1 - pl.col("first_departure_time"),
                    pl.col("headway_secs"),
                )
                .alias("delta_time")
                .list.eval(pl.element().filter(pl.element() != 0).append(0))
            ]
        )

        stop_times = (
            stop_times.join(
                delta_times.select(["trip_id", "delta_time", "headway_secs"]),
                on="trip_id",
                how="left",
            )
            .explode("delta_time")
            .with_columns(pl.col("delta_time").fill_null(0).alias("delta_time"))
            .with_columns(
                (pl.col("arrival_time") + pl.col("delta_time")).alias("arrival_time"),
                (pl.col("departure_time") + pl.col("delta_time")).alias(
                    "departure_time"
                ),
                (
                    pl.when(pl.col("delta_time") != 0)
                    .then(
                        pl.concat_str(
                            pl.col("trip_id"),
                            pl.lit("_"),
                            (pl.col("delta_time") / pl.col("headway_secs"))
                            .ceil()
                            .cast(int),
                        )
                    )
                    .otherwise(pl.col("trip_id"))
                ).alias("trip_id"),
            )
            .with_columns(
                pl.when(pl.col("delta_time") > 0)
                .then(pl.lit(False))
                .otherwise(pl.col("next_day"))
                .alias("next_day")
            )
            .drop("delta_time", "headway_secs")
        )

        frequencies = (
            frequencies.join(
                midnight_frequencies.select(["trip_id", "new_end_time"]).unique(
                    "trip_id"
                ),
                on="trip_id",
                how="left",
            )
            .with_columns(
                [
                    pl.when(pl.col("new_end_time").is_not_null())
                    .then(pl.col("new_end_time"))
                    .otherwise(pl.col("end_time"))
                    .alias("end_time")
                ]
            )
            .drop("new_end_time")
            .filter(
                (pl.col("start_time") + pl.col("headway_secs")) < pl.col("end_time")
            )
        )

        return stop_times, frequencies

    def generate_shape_ids(self) -> pl.LazyFrame:
        """
        Groups trips by stop sequence and travel time to create canonical shape IDs.

        This method generates new `shape_id`s. A shape is defined as a group of trips
        that share the exact same sequence of stops and have a total travel duration
        within a configurable tolerance (`TRIP_ROUND_TIME`, e.g., 2 minutes).

        This is useful for identifying all trips that follow the same physical path
        with a similar duration, effectively creating a `shape_id` when one is not
        provided or is inconsistent in the GTFS data.

        Returns:
            pl.LazyFrame: A LazyFrame where each row represents a unique "shape",
                          containing the generated `shape_id`, a list of all `trip_ids`
                          belonging to it, the common `stop_ids` list, and the
                          `stop_sequence`.
        """
        trip_sequences = (
            self.lf.sort(["trip_id", "stop_sequence"])
            .group_by("trip_id")
            .agg(
                [
                    pl.col("stop_id").sort_by("stop_sequence").alias("stop_ids"),
                    pl.col("stop_sequence").sort().alias("stop_sequence"),
                    pl.col("shape_total_travel_time")
                    .first()
                    .alias("shape_total_travel_time"),
                    (
                        pl.col("shape_total_travel_time").first()
                        // TRIP_ROUND_TIME
                        * TRIP_ROUND_TIME
                    ).alias("shape_total_travel_time_rounded"),
                ]
            )
        )

        grouped = trip_sequences.group_by(
            ["stop_ids", "shape_total_travel_time_rounded"]
        ).agg(
            [
                pl.col("trip_id").unique().alias("trip_ids"),
                pl.col("stop_sequence").first(),
                pl.col("trip_id").min().alias("shape_id"),
            ]
        )

        return grouped

    def filter_by_time_range(
        self,
        start_time: datetime | time,
        end_time: datetime | time,
        strict: bool = True,
    ) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]:
        """
        Filters stop_times and frequencies by a single time interval.

        It handles intervals on the same date. If `frequencies` data is available,
        it ensures that entire trips belonging to an overlapping frequency definition
        are kept, even if not all their stops fall within the time bounds.

        Args:
            start_time (datetime|time): Start datetime of the filter interval.
            end_time (datetime|time): End datetime of the filter interval.

        Returns:
            Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]: A tuple containing:
                - `filtered_stop_times`: LazyFrame filtered to the time interval.
                - `filtered_frequencies`: LazyFrame of frequencies overlapping the interval,
                  or None if no frequency data exists.

        Raises:
            ValueError: If `start_time` and `end_time` are not on the same date.
        """
        start_secs = utils.time_to_seconds(start_time)
        end_secs = utils.time_to_seconds(end_time)

        if self.frequencies is None:
            if start_time.date() != end_time.date():
                raise ValueError("Start and end datetime must be on the same date")

            if strict:
                filtered_stop_times = self.lf.filter(
                    (pl.col("arrival_time") >= start_secs)
                    & (pl.col("arrival_time") <= end_secs)
                )
            else:
                filtered_stop_times = self.lf.join(
                    self.lf.filter(
                        (pl.col("arrival_time") >= start_secs)
                        & (pl.col("arrival_time") <= end_secs)
                    ),
                    on="trip_id",
                    how="semi",
                )

            return filtered_stop_times, None

        if strict:
            filtered_frequencies = self.frequencies.filter(
                (pl.col("start_time") < end_secs) & (pl.col("end_time") > start_secs)
            )

            filtered_frequencies = filtered_frequencies.with_columns(
                [
                    pl.when(pl.col("end_time") > end_secs)
                    .then(end_secs)
                    .otherwise(pl.col("end_time"))
                    .alias("end_time"),
                    pl.when(pl.col("start_time") < start_secs)
                    .then(start_secs)
                    .otherwise(pl.col("start_time"))
                    .alias("start_time"),
                ]
            )

        else:
            filtered_frequencies = self.frequencies.filter(
                (pl.col("start_time") < end_secs) & (pl.col("end_time") > start_secs)
            )

            filtered_frequencies = filtered_frequencies.with_columns(
                [
                    pl.when(pl.col("end_time") > end_secs)
                    .then(end_secs)
                    .otherwise(pl.col("end_time"))
                    .alias("end_time"),
                    pl.when(
                        (pl.col("start_time") - pl.col("shape_total_travel_time"))
                        < (start_secs - pl.col("shape_total_travel_time"))
                    )
                    .then(
                        pl.when((start_secs - pl.col("shape_total_travel_time")) < 0)
                        .then(0)
                        .otherwise(start_secs - pl.col("shape_total_travel_time"))
                    )
                    .otherwise(pl.col("start_time"))
                    .alias("start_time"),
                ]
            )

        if strict:
            filtered_stop_times = pl.concat(
                [
                    self.lf.join(filtered_frequencies, on="trip_id", how="semi"),
                    self.lf.join(filtered_frequencies, on="trip_id", how="anti").filter(
                        (pl.col("arrival_time") >= start_secs)
                        & (pl.col("arrival_time") <= end_secs)
                    ),
                ]
            )
        else:
            valid_trip_ids = pl.concat(
                [
                    filtered_frequencies.select("trip_id"),
                    self.lf.join(filtered_frequencies, on="trip_id", how="anti")
                    .filter(
                        (pl.col("arrival_time") >= start_secs)
                        & (pl.col("arrival_time") <= end_secs)
                    )
                    .select("trip_id"),
                ]
            )

            filtered_stop_times = self.lf.join(valid_trip_ids, on="trip_id", how="semi")

        filtered_stop_times = self.__correct_sequence(filtered_stop_times)
        filtered_stop_times = self._add_shape_time_and_midnight_crossing(
            filtered_stop_times
        )

        return filtered_stop_times, filtered_frequencies

    def to_hhmmss(self, field: str, new_field: str) -> pl.Expr:
        """
        Creates a Polars expression to convert seconds since midnight to a HH:MM:SS string.

        Args:
            field (str): The name of the column containing seconds (integer).
            new_field (str): The desired name for the new HH:MM:SS string column.

        Returns:
            pl.Expr: A Polars expression that performs the conversion. Can be used in
                     a `.with_columns()` statement.
        """
        seconds_expr = pl.col(field)
        hours = (seconds_expr // 3600).cast(pl.Int32)
        minutes = ((seconds_expr % 3600) // 60).cast(pl.Int32)
        seconds = (seconds_expr % 60).cast(pl.Int32)
        return (pl.format("{:02}:{:02}:{:02}", hours, minutes, seconds)).alias(
            new_field
        )
