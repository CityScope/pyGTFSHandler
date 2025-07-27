# -*- coding: utf-8 -*-
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
    also be applied after initialization using the `filter_by_time_bounds` and
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

import polars as pl
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from utils import read_csv_list, get_df_schema_dict
from datetime import datetime, time
from functools import reduce
import warnings
import operator

# A constant used for rounding trip travel times when generating shape_ids.
# It groups trips with travel times within a 2-minute (120s) window.
TRIP_ROUND_TIME: int = 120


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
            self.paths: List[Path] = [Path(path)]
        else:
            self.paths: List[Path] = [Path(p) for p in path]

        # --- Main Processing Pipeline ---
        self.lf: pl.LazyFrame = self.__read_stop_times(stop_ids, trip_ids)
        self.lf = self.__correct_sequence(self.lf)
        self.lf, self.fixed_times = self.__fix_nulls_easy(self.lf)

        if self.fixed_times:
            warnings.warn("Some departure times are null and have been interpolated")

        self.lf = self._add_shape_time_and_midnight_crossing(self.lf)
        # Eagerly evaluate to checkpoint the result and optimize subsequent queries
        self.lf = self.lf.collect().lazy()

        self.frequencies: Optional[pl.LazyFrame] = self.__read_frequencies(trip_ids)
        if self.frequencies is not None:
            self.frequencies = self.__fix_headway(self.frequencies)
            self.frequencies = self._frequencies_midnight_crossing(self.frequencies)
            self.frequencies = self._add_departure_time_to_frequencies(
                self.lf, self.frequencies
            )
            self.lf, self.frequencies = self._midnight_frequencies_to_stop_times(
                self.lf, self.frequencies
            )

            # Eagerly evaluate frequencies to checkpoint results
            self.frequencies = self.frequencies.collect().lazy()
            self.frequencies = self._add_frequencies_n_trips(self.frequencies)

            # Recalculate travel times as new trips may have been added
            self.lf = self._add_shape_time_and_midnight_crossing(self.lf)
            self.lf = self.lf.collect().lazy()

        if start_time and end_time:
            self.lf, self.frequencies = self.filter_by_time_bounds(start_time, end_time)

    def __read_stop_times(
        self, stop_ids: Optional[List[str]] = None, trip_ids: Optional[List[str]] = None
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
            p / "stop_times.txt" for p in self.paths if (p / "stop_times.txt").exists()
        ]
        if not stop_times_paths:
            raise FileNotFoundError("No stop_times.txt files found in given paths.")

        schema_dict: Dict[str, pl.DataType] = get_df_schema_dict(stop_times_paths[0])
        stop_times: pl.LazyFrame = read_csv_list(
            stop_times_paths, schema_overrides=schema_dict
        )

        if stop_ids:
            stop_ids_df = pl.DataFrame({"stop_id": stop_ids})
            stop_times = stop_times.join(stop_ids_df.lazy(), on="stop_id", how="inner")

        if trip_ids:
            trip_ids_df = pl.DataFrame({"trip_id": trip_ids})
            stop_times = stop_times.join(trip_ids_df.lazy(), on="trip_id", how="inner")

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
        )

        stop_times = (
            stop_times.sort("trip_id", "stop_sequence", "original_idx")
            .with_columns(
                (pl.arange(0, pl.count()).over("trip_id")).alias("stop_sequence")
            )
            .drop("original_idx")
        )

        return stop_times

    def __fix_nulls_easy(self, stop_times: pl.LazyFrame) -> Tuple[pl.LazyFrame, bool]:
        """
        Interpolates null departure and arrival times using a forward-fill strategy.

        This method handles cases where intermediate stops in a trip have null times.
        It flags rows where times were interpolated.

        Args:
            stop_times (pl.LazyFrame): The input LazyFrame of stop times.

        Returns:
            Tuple[pl.LazyFrame, bool]: A tuple containing:
                - The LazyFrame with times interpolated.
                - A boolean, `True` if any times were fixed, `False` otherwise.
        """
        has_nulls_expr = pl.col("departure_time").is_null().any()
        has_nulls: bool = stop_times.select(has_nulls_expr).collect().item()

        if has_nulls:
            stop_times = stop_times.sort(["trip_id", "stop_sequence"]).with_columns(
                (pl.col("departure_time").is_null()).alias("fixed_time"),
                pl.col("departure_time")
                .forward_fill()
                .over("trip_id")
                .alias("departure_time"),
            )

            stop_times = stop_times.with_columns(
                [
                    pl.when(pl.col("arrival_time").is_null())
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
        stop_times = stop_times.with_columns(
            pl.when(
                (pl.col("departure_time") >= 86400) | (pl.col("arrival_time") >= 86400)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("next_day")
        )

        stop_times = stop_times.with_columns(
            [
                pl.when(pl.col("departure_time") >= 86400)
                .then(pl.col("departure_time") - 86400)
                .otherwise(pl.col("departure_time"))
                .alias("departure_time"),
                pl.when(pl.col("arrival_time") >= 86400)
                .then(pl.col("arrival_time") - 86400)
                .otherwise(pl.col("arrival_time"))
                .alias("arrival_time"),
            ]
        )

        stop_times = (
            stop_times.sort(["trip_id", "stop_sequence"])
            .with_columns(
                [
                    pl.col("departure_time")
                    .shift(1)
                    .over("trip_id")
                    .fill_null(strategy="forward")
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
                    .then(pl.col("shape_time_delta") + 86400)
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
            .drop("prev_departure_time")
        )

        total_travel_times = stop_times.group_by("trip_id").agg(
            pl.col("shape_time_traveled").max().alias("shape_total_travel_time")
        )

        stop_times = stop_times.join(total_travel_times, on="trip_id", how="left")

        return stop_times

    def __read_frequencies(
        self, trip_ids: Optional[List[str]] = None
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
            p / "frequencies.txt"
            for p in self.paths
            if (p / "frequencies.txt").exists()
        ]
        if not frequencies_paths:
            return None

        schema_dict: Dict[str, pl.DataType] = get_df_schema_dict(frequencies_paths[0])
        frequencies: pl.LazyFrame = read_csv_list(
            frequencies_paths, schema_overrides=schema_dict
        )

        if trip_ids:
            trip_ids_df = pl.DataFrame({"trip_id": trip_ids})
            frequencies = frequencies.join(
                trip_ids_df.lazy(), on="trip_id", how="inner"
            )

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
            ]
        )

        return frequencies

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
            pl.when((pl.col("start_time") >= 86400) & (pl.col("end_time") >= 86400))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("next_day")
        )

        frequencies = frequencies.with_columns(
            [
                pl.when((pl.col("start_time") >= 86400) & (pl.col("end_time") >= 86400))
                .then(pl.col("start_time") - 86400)
                .otherwise(pl.col("start_time"))
                .alias("start_time"),
                pl.when((pl.col("start_time") >= 86400) & (pl.col("end_time") >= 86400))
                .then(pl.col("end_time") - 86400)
                .otherwise(pl.col("end_time"))
                .alias("end_time"),
            ]
        )

        frequencies = frequencies.with_columns(
            pl.when(pl.col("end_time") == 86400)
            .then(86399)
            .otherwise(pl.col("end_time"))
            .alias("end_time")
        )

        spans_midnight = frequencies.filter(
            (pl.col("end_time") < pl.col("start_time")) | (pl.col("end_time") >= 86400)
        )

        first_half = spans_midnight.with_columns(pl.lit(86399).alias("end_time"))
        second_half = spans_midnight.with_columns(
            pl.lit(0).alias("start_time"), pl.lit(True).alias("next_day")
        )

        duplicated_rows = pl.concat([first_half, second_half], how="vertical_relaxed")
        normal_rows = frequencies.filter(
            (pl.col("end_time") >= pl.col("start_time")) & (pl.col("end_time") < 86400)
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
            .floor()
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
        )
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
            pl.col("end_time") + pl.col("shape_total_travel_time") >= 86400
        ).with_columns(
            [((86400 - pl.col("shape_total_travel_time")) - 1).alias("new_end_time")]
        )

        delta_times = (
            midnight_frequencies.with_columns(
                [
                    (
                        (
                            (pl.col("new_end_time") - pl.col("departure_time"))
                            / pl.col("headway_secs")
                        ).floor()
                        * pl.col("headway_secs")
                        + pl.col("departure_time")
                    ).alias("aligned_start")
                ]
            )
            .with_columns(
                [
                    (
                        pl.concat_list(
                            [
                                pl.lit([0]),
                                pl.int_ranges(
                                    pl.col("aligned_start"),
                                    pl.col("end_time") + 1,
                                    pl.col("headway_secs"),
                                ),
                            ]
                        )
                    ).alias("delta_time")
                ]
            )
            .explode("delta_time")
        )

        updated_stop_times = (
            stop_times.join(
                delta_times.select(["trip_id", "delta_time"]), on="trip_id", how="left"
            )
            .with_columns(pl.col("delta_time").fill_null(pl.lit(0)))
            .with_columns(
                (pl.col("arrival_time") + pl.col("delta_time")).alias("arrival_time"),
                (pl.col("departure_time") + pl.col("delta_time")).alias(
                    "departure_time"
                ),
                pl.when(pl.col("delta_time") > 0)
                .then(
                    pl.col("trip_id").cast(pl.Utf8)
                    + "_"
                    + pl.col("delta_time").cast(pl.Utf8)
                )
                .otherwise(pl.col("trip_id"))
                .alias("trip_id"),
                pl.col("trip_id").alias("orig_trip_id"),
            )
            .with_columns(
                pl.when(pl.col("delta_time") > 0)
                .then(pl.lit(False))
                .otherwise(pl.col("next_day"))
                .alias("next_day")
            )
            .drop("delta_time")
        )

        updated_frequencies = (
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
            .filter(pl.col("start_time") + pl.col("headway_secs") < pl.col("end_time"))
        )

        return updated_stop_times, updated_frequencies

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

    def filter_by_time_bounds(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]:
        """
        Filters stop_times and frequencies by a single time interval.

        It handles intervals on the same date. If `frequencies` data is available,
        it ensures that entire trips belonging to an overlapping frequency definition
        are kept, even if not all their stops fall within the time bounds.

        Args:
            start_time (datetime): Start datetime of the filter interval.
            end_time (datetime): End datetime of the filter interval.

        Returns:
            Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]: A tuple containing:
                - `filtered_stop_times`: LazyFrame filtered to the time interval.
                - `filtered_frequencies`: LazyFrame of frequencies overlapping the interval,
                  or None if no frequency data exists.

        Raises:
            ValueError: If `start_time` and `end_time` are not on the same date.
        """

        def time_to_seconds(t: time) -> int:
            return t.hour * 3600 + t.minute * 60 + t.second

        start_secs = time_to_seconds(start_time.time())
        end_secs = time_to_seconds(end_time.time())

        if self.frequencies is None:
            if start_time.date() != end_time.date():
                raise ValueError("Start and end datetime must be on the same date")

            filtered_stop_times = self.lf.filter(
                (pl.col("arrival_time") >= start_secs)
                & (pl.col("arrival_time") <= end_secs)
            )
            return filtered_stop_times, None

        if start_time.date() != end_time.date():
            raise ValueError("Start and end datetime must be on the same date")

        filtered_frequencies = self.frequencies.filter(
            (pl.col("start_time") <= end_secs) & (pl.col("end_time") >= start_secs)
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

        valid_trip_ids = filtered_frequencies.select("trip_id")

        filtered_stop_times = self.lf.filter(
            (
                (pl.col("arrival_time") >= start_secs)
                & (pl.col("arrival_time") <= end_secs)
            )
            | (pl.col("trip_id").is_in(valid_trip_ids))
        )

        return filtered_stop_times, filtered_frequencies

    def filter_by_multi_time_bounds(
        self, time_bounds: List[Tuple[datetime, datetime]]
    ) -> Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]:
        """
        Filters stop_times and frequencies lazily by multiple time intervals.

        Each interval in the list must be within the same calendar date.

        Args:
            time_bounds (List[Tuple[datetime, datetime]]): A list of
                (start_datetime, end_datetime) tuples defining the filter windows.

        Returns:
            Tuple[pl.LazyFrame, Optional[pl.LazyFrame]]: A tuple containing:
                - `filtered_stop_times`: LazyFrame filtered to the union of all intervals.
                - `filtered_frequencies`: LazyFrame of frequencies overlapping any interval,
                  or None if no frequency data exists.

        Raises:
            ValueError: If any time interval tuple has start and end on different dates.
        """

        def time_to_seconds(t: time) -> int:
            return t.hour * 3600 + t.minute * 60 + t.second

        intervals = []
        for start_dt, end_dt in time_bounds:
            if start_dt.date() != end_dt.date():
                raise ValueError("Each interval must be within the same date")
            intervals.append(
                (time_to_seconds(start_dt.time()), time_to_seconds(end_dt.time()))
            )

        if self.frequencies is None:
            arrival_filters = [
                (pl.col("arrival_time") >= start_secs)
                & (pl.col("arrival_time") <= end_secs)
                for start_secs, end_secs in intervals
            ]
            combined_arrival_filter = reduce(operator.or_, arrival_filters)

            filtered_stop_times = self.lf.filter(combined_arrival_filter)
            return filtered_stop_times, None

        freq_filters = [
            (pl.col("start_time") <= end_secs) & (pl.col("end_time") >= start_secs)
            for start_secs, end_secs in intervals
        ]
        combined_freq_filter = reduce(operator.or_, freq_filters)

        filtered_frequencies = self.frequencies.filter(combined_freq_filter)

        start_clips = [
            pl.when(pl.col("start_time") < start_secs)
            .then(start_secs)
            .otherwise(pl.col("start_time"))
            for start_secs, _ in intervals
        ]
        clipped_start = reduce(
            lambda acc, expr: pl.when(expr > acc).then(expr).otherwise(acc),
            start_clips,
            pl.lit(0),
        )

        end_clips = [
            pl.when(pl.col("end_time") > end_secs)
            .then(end_secs)
            .otherwise(pl.col("end_time"))
            for _, end_secs in intervals
        ]
        clipped_end = reduce(
            lambda acc, expr: pl.when(expr < acc).then(expr).otherwise(acc),
            end_clips,
            pl.lit(86399),
        )

        filtered_frequencies = filtered_frequencies.with_columns(
            [
                clipped_start.alias("start_time"),
                clipped_end.alias("end_time"),
            ]
        )

        valid_trip_ids = filtered_frequencies.select("trip_id")

        arrival_filters = [
            (pl.col("arrival_time") >= start_secs)
            & (pl.col("arrival_time") <= end_secs)
            for start_secs, end_secs in intervals
        ]
        combined_arrival_filter = reduce(operator.or_, arrival_filters)

        filtered_stop_times = self.lf.filter(
            combined_arrival_filter | pl.col("trip_id").is_in(valid_trip_ids)
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
