# -*- coding: utf-8 -*-
import polars as pl
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
from ..utils import gtfs_checker
from ..utils import io
from datetime import datetime, time
import warnings
import os

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

from ..utils import time_parsing
from ..utils.time_parsing import normalize_time_expr, SECS_PER_DAY
from ..utils import geo_polars
from .frequencies import FrequenciesMixin

# A constant used for rounding trip travel times when generating shape_ids.
# It groups trips with travel times within a 5-minute (300s) window.
TRIP_ROUND_TIME: int = 300


class StopTimes(FrequenciesMixin):
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
    def __init__(self,lf=None,frequencies=None,fixed_times=None,trips_lf=None) -> None:
        self.lf = lf 
        self.frequencies = frequencies 
        self.fixed_times = fixed_times
        self.trips_lf = trips_lf

    def load(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        trips,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        stop_ids: Optional[List[str] | pl.DataFrame | pl.LazyFrame] = None,
        trip_ids: Optional[List[str]] = None,
        check_files:bool=False,
        min_file_id=0
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
        self.lf: pl.LazyFrame = self._read_stop_times(paths, trip_ids, check_files=check_files, min_file_id=min_file_id)

        # Referential integrity: drop stop_times.txt rows whose trip_id has
        # no matching row in trips.txt (a dangling reference) *before* any
        # frequency-driven trip_id expansion happens below -- doing this
        # check later, against the post-expansion trip_ids, would wrongly
        # flag every legitimate frequency-generated instance as orphaned.
        known_trip_ids = trips.select("trip_id").unique()
        orphaned_trip_ids = (
            self.lf.select("trip_id").unique().join(known_trip_ids, on="trip_id", how="anti").collect()
        )
        self.lf = self.lf.join(known_trip_ids, on="trip_id", how="semi")
        if orphaned_trip_ids.height > 0:
            warnings.warn(
                f"{orphaned_trip_ids.height} trip_id(s) in stop_times.txt "
                f"do not exist in trips.txt; their stop_times were dropped."
            )

        self.lf = self._correct_sequence(self.lf)
        self.lf, self.fixed_times = self._fix_nulls_easy(self.lf)
        self.lf = self._normalize_times(self.lf)

        if stop_ids is not None:
            self.lf = self._filter_by_stop_id(self.lf, stop_ids)
            self.lf = self._correct_sequence(self.lf)
            self.lf = self.lf.collect().lazy()

        if self.fixed_times:
            warnings.warn("Some departure times are null and have been interpolated")

        self.frequencies: Optional[pl.LazyFrame] = self._read_frequencies(
            paths, trip_ids, check_files=check_files, min_file_id=min_file_id
        )

        if self.frequencies is not None:
            self.lf = self.lf.collect().lazy()

            self.frequencies = self.frequencies.join(self.lf, on="trip_id", how="semi")
            if check_files:
                self.frequencies = self._filter_repeated_frequencies_with_trips(
                    self.frequencies, trips
                )
            self.frequencies = self.frequencies.collect().lazy()
            self.frequencies = self._frequencies_midnight_crossing(self.frequencies)
            if check_files:
                self.lf, self.frequencies = self._check_frequencies_in_stop_times(
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
            if self.frequencies is not None:
                self.frequencies = self.frequencies.collect().lazy()

            self.lf, self.frequencies = self.filter_by_time_range(
                start_time, end_time, strict=False
            )

            if self.frequencies is not None:
                self.frequencies = self.frequencies.collect().lazy()

        # Eagerly evaluate to checkpoint the result and optimize subsequent queries
        self.lf = self.lf.collect().lazy()

        if self.frequencies is not None:
            self.frequencies = self._fix_headway(self.frequencies)
            self.lf, self.frequencies = self._midnight_frequencies_to_stop_times(
                self.lf, self.frequencies
            )
            if start_time and end_time:
                self.frequencies = self.frequencies.filter(
                    (pl.col("start_time") < time_parsing.time_to_seconds(end_time))
                    & (pl.col("end_time") > time_parsing.time_to_seconds(start_time))
                )

            # Eagerly evaluate frequencies to checkpoint results
            self.frequencies = self.frequencies.collect().lazy()

            # Reconcile each trip's chain of windows against headway_secs
            # after every load-time filter (initial start/end window,
            # midnight-crossing split) has already shrunk/adjusted them --
            # always run, regardless of check_files, so no window's fractional
            # trailing period leaks into an adjacent window.
            self.frequencies = self._reconcile_frequency_windows(
                self.frequencies, check_files=check_files
            )
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

        # `day_offset` on `self.lf` already resolves which real calendar day
        # each stop belongs to (see `_normalize_times`), so trips no longer
        # need to be duplicated as a separate "_night" service_id/trip_id
        # variant the way the old single-bit `next_day` scheme required.
        self.trips_lf = trips

    def _read_stop_times(
        self, paths, trip_ids: Optional[List[str]] = None, check_files=False, min_file_id=0
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
        stop_times_paths: List[Path] = []
        file = "stop_times.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                stop_times_paths.append(None)
                warnings.warn(f"File {file} does not exist in {p}", UserWarning)
            else:
                stop_times_paths.append(new_p)

        schema_dict, _ = gtfs_checker.get_df_schema_dict("stop_times.txt")
        stop_times: pl.LazyFrame = io.read_csv_list(
            stop_times_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id
        )
        if (stop_times is None) or (stop_times.select(pl.len()).collect().item() == 0):
            raise Exception(f"No stop_times.txt file found for any {paths}")
        

        stop_times = stop_times.with_columns(
            pl.when(pl.col("arrival_time").str.strip_chars().eq(""))
            .then(None)
            .otherwise(pl.col("arrival_time")).alias("arrival_time"),
            pl.when(pl.col("departure_time").str.strip_chars().eq(""))
            .then(None)
            .otherwise(pl.col("departure_time")).alias("departure_time")
        )

        stop_times = geo_polars.filter_by_id_column(stop_times, "trip_id", trip_ids)
        
        stop_times = stop_times.with_columns([
            normalize_time_expr("arrival_time").alias("arrival_time"),
            normalize_time_expr("departure_time").alias("departure_time"),
        ])

        # Checkpoint here (a single `.collect()`, then re-wrapped as lazy):
        # `stop_times.txt` is typically the single largest file in a feed,
        # and every subsequent step in this method -- and, without this, in
        # `_correct_sequence`/`_fix_nulls_easy`/`_normalize_times` right
        # after it -- runs its own small "probe" query (row counts, null
        # counts) that would otherwise silently re-run the *entire* CSV
        # scan + time-string parsing from scratch each time, since none of
        # those earlier steps had materialized anything yet. Materializing
        # once here means every later probe/transform works off the same
        # in-memory Arrow data instead of re-reading and re-parsing the file
        # repeatedly.
        stop_times = stop_times.collect().lazy()

        # Lazily count rows with invalid (null) times
        null_count_expr = (
            ((pl.col("departure_time") == pl.lit("None")) | (pl.col("arrival_time") == pl.lit("None")))
            .sum()
            .alias("num_invalid_rows")
        )

        null_count_df = stop_times.select(null_count_expr).collect()
        num_invalid_rows = null_count_df.item()  # extract scalar

        # Warn (in Python, after the lazy step)
        if num_invalid_rows > 0:
            warnings.warn(f"{num_invalid_rows} rows dropped due to invalid time values in stop_times.", UserWarning)


        # Only drop rows with the "None" sentinel (a genuinely malformed,
        # unparseable time string) -- a real null (blank field, meant to be
        # interpolated later by `_fix_nulls_easy`) must survive here. Using
        # `!= "None"` alone would also drop every null row, since `null !=
        # "None"` evaluates to null, which `.filter()` treats as "exclude" --
        # silently discarding the stop entirely instead of leaving a gap to
        # interpolate.
        stop_times = stop_times.filter(
            pl.col("arrival_time").is_null() | (pl.col("arrival_time") != pl.lit("None"))
        )

        stop_times = stop_times.filter(
            pl.col("departure_time").is_null() | (pl.col("departure_time") != pl.lit("None"))
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

        # Safely cast times to seconds
        stop_times = stop_times.with_columns([
            (
                (pl.col("departure_time").str.slice(0, 2).cast(pl.Int32, strict=False) * 3600)
                + (pl.col("departure_time").str.slice(3, 2).cast(pl.Int32, strict=False) * 60)
                + (pl.col("departure_time").str.slice(6, 2).cast(pl.Int32, strict=False))
            ).alias("departure_time"),

            (
                (pl.col("arrival_time").str.slice(0, 2).cast(pl.Int32, strict=False) * 3600)
                + (pl.col("arrival_time").str.slice(3, 2).cast(pl.Int32, strict=False) * 60)
                + (pl.col("arrival_time").str.slice(6, 2).cast(pl.Int32, strict=False))
            ).alias("arrival_time"),

            pl.lit(False).alias("next_day"),
        ])

        return stop_times

    def _correct_sequence(self, stop_times: pl.LazyFrame) -> pl.LazyFrame:
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

    def _normalize_times(self, stop_times: pl.LazyFrame) -> pl.LazyFrame:
        """
        Resolves each trip's `arrival_time`/`departure_time` into a `day_offset`
        (0, 1, 2, ...) and a 0-24h `time_of_day`, regardless of whether the
        source feed encoded overnight trips explicitly (e.g. ``25:10:00``,
        parsed by `_read_stop_times` as the literal, unbounded 90600 seconds)
        or implicitly (silently wrapping back past midnight with no ``>=24``
        marker at all, e.g. ``23:58:00`` followed by ``00:10:00``).

        How it works: within each trip (ordered by `stop_sequence`), the raw,
        unbounded seconds value already encodes any explicit day count
        (``50:10:00`` parses to 180600s, i.e. day_offset 2, without any
        correction needed) and is therefore always monotonically
        non-decreasing along an explicitly-encoded trip. What it can't encode
        is an *implicit* wrap: a raw value that is smaller than the previous
        stop's raw value, because the source feed reused `00:xx:xx`-style
        clock time instead of continuing past `24:00:00`. Each such decrease
        means exactly one more real day has passed, so a single cumulative
        count of "did the raw value decrease versus the previous stop"
        (`.cum_sum().over("trip_id")`, ordered by `stop_sequence`) gives the
        exact number of *additional* days to add on top of whatever the raw
        value already encoded explicitly -- one cheap, non-iterative pass,
        correct for feeds that mix both styles across different trips (or
        even rely on explicit encoding for part of a trip and implicit
        wraparound for the rest), since explicitly-encoded stops simply never
        trigger the decrease condition in the first place.

        After this correction, `day_offset = corrected_seconds // 86400` and
        `time_of_day = corrected_seconds % 86400` are exact. `arrival_time`/
        `departure_time` are then reduced back to the 0-24h `time_of_day`
        range (matching this codebase's existing "seconds within a day"
        convention used throughout `Feed`), while `day_offset` is the new,
        unbounded source of truth for which real calendar day a stop belongs
        to (replacing the old single-bit `next_day` flag).

        Args:
            stop_times: LazyFrame with raw, unbounded-seconds `arrival_time`/
                `departure_time` columns (as produced by `_read_stop_times`).

        Returns:
            pl.LazyFrame: `stop_times` with `arrival_time`/`departure_time`
            reduced to 0-24h seconds, plus new `day_offset` (Int64) and
            `time_of_day` (Int64, == the new `departure_time`) columns, and
            `next_day` kept (as `day_offset > 0`) for backward compatibility
            with code that only needs a same-day/next-day boolean.
        """
        stop_times = stop_times.sort(["trip_id", "stop_sequence"])

        for column in ("departure_time", "arrival_time"):
            # Nesting two `.over("trip_id")` window expressions in a single
            # expression (one for `previous_raw`, another wrapping the
            # cum_sum that depends on it) is rejected by newer polars
            # ("window expression not allowed in aggregation"). Materialize
            # the shifted column first so the cum_sum's `.over()` only wraps
            # a plain column reference.
            stop_times = stop_times.with_columns(
                pl.col(column).shift(1).over("trip_id").alias(f"_previous_{column}")
            )
            raw = pl.col(column)
            previous_raw = pl.col(f"_previous_{column}")
            implicit_wraps = (
                (previous_raw.is_not_null() & (raw < previous_raw))
                .cast(pl.Int64)
                .cum_sum()
                .over("trip_id")
            )
            stop_times = stop_times.with_columns(
                (raw + implicit_wraps * SECS_PER_DAY).alias(column)
            ).drop(f"_previous_{column}")

        stop_times = stop_times.with_columns(
            (pl.col("departure_time") // SECS_PER_DAY).alias("day_offset"),
        )

        stop_times = stop_times.with_columns(
            (pl.col("departure_time") % SECS_PER_DAY).alias("departure_time"),
            (pl.col("arrival_time") % SECS_PER_DAY).alias("arrival_time"),
        )

        stop_times = stop_times.with_columns(
            pl.col("departure_time").alias("time_of_day"),
            (pl.col("day_offset") > 0).alias("next_day"),
        )
        return stop_times

    def _filter_by_stop_id(self, stop_times: pl.LazyFrame, stop_ids) -> pl.LazyFrame:
        # Create stop_ids as a lazy frame

        if isinstance(stop_ids, list):
            stop_ids_lf = pl.LazyFrame({"stop_id": stop_ids})

            # Select matching stop_times with just needed columns
            stop_times_filter = stop_times.select(
                ["trip_id", "stop_id", "stop_sequence"]
            ).join(stop_ids_lf, on=["stop_id"], how="semi")
        elif stop_ids is not None:
            if isinstance(stop_ids, pl.DataFrame):
                stop_ids = stop_ids.lazy()

            columns = stop_ids.collect_schema().names()

            stop_times_filter = stop_times.select(
                ["trip_id", "stop_sequence", *columns]
            ).join(stop_ids, on=columns, how="semi")

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

    def _fix_nulls_easy(self, stop_times: pl.LazyFrame) -> Tuple[pl.LazyFrame, bool]:
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
                .round(0)  # round to nearest integer
                .cast(int)  # ensure integer type
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

            stop_times = stop_times.filter(pl.col("departure_time").is_not_null())
            stop_times = stop_times.filter(pl.col("departure_time").is_not_nan())

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
        # `departure_time` is the 0-24h time_of_day; `day_offset` (from
        # `_normalize_times`) already fully resolves which real day each stop
        # falls on, whether the source feed used explicit `>=24:00:00` times
        # or silently wrapped past midnight. Reconstructing the absolute,
        # monotonically non-decreasing seconds value from the two is enough
        # to compute travel time correctly, with no further decrease-detection
        # needed here (that used to be this function's job, before
        # `day_offset` existed).
        stop_times = stop_times.sort(["trip_id", "stop_sequence"]).with_columns(
            (pl.col("departure_time") + pl.col("day_offset") * SECS_PER_DAY).alias(
                "absolute_departure_time"
            )
        )

        stop_times = (
            stop_times.with_columns(
                [
                    pl.col("absolute_departure_time")
                    .shift(1)
                    .over("trip_id")
                    .alias("prev_departure_time")
                ]
            )
            .with_columns(
                [
                    (pl.col("absolute_departure_time") - pl.col("prev_departure_time"))
                    .fill_null(0)
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
            .drop("absolute_departure_time")
        )

        total_travel_times = stop_times.group_by("trip_id").agg(
            pl.col("shape_time_traveled").max().alias("shape_total_travel_time")
        )

        if "shape_total_travel_time" in stop_times.collect_schema().names():
            stop_times = stop_times.drop("shape_total_travel_time")

        stop_times = stop_times.join(total_travel_times, on="trip_id", how="left")

        return stop_times

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
        start_secs = time_parsing.time_to_seconds(start_time)
        end_secs = time_parsing.time_to_seconds(end_time)

        if self.frequencies is None:
            if (
                isinstance(start_time, datetime)
                and isinstance(end_time, datetime)
                and start_time.date() != end_time.date()
            ):
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

        filtered_stop_times = self._correct_sequence(filtered_stop_times)
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

