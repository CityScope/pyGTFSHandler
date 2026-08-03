# -*- coding: utf-8 -*-
"""GTFS frequencies.txt handling: reading, headway correction, and expanding
frequency-defined trips (including ones that cross midnight) into concrete
stop_times rows.

Split out of models/stop_times.py to keep that file within a readable size.
Mixed into `StopTimes` (`class StopTimes(FrequenciesMixin)`), so `self` here
is always a `StopTimes` instance -- these are not usable as a standalone
class. `normalize_time_expr` and `SECS_PER_DAY` are imported from
`stop_times` rather than duplicated, so the two modules share one exact
definition of "how a GTFS time string is parsed."
"""

import polars as pl
from typing import List, Optional, Tuple
import warnings

from ..utils import gtfs_checker
from ..utils import io
from ..utils.time_parsing import normalize_time_expr, SECS_PER_DAY
from ..utils import geo_polars


class FrequenciesMixin:
    """Mixin providing `StopTimes`'s `frequencies.txt` reading and
    frequency-to-stop_times expansion logic."""

    def _read_frequencies(
        self, paths, trip_ids: Optional[List[str]] = None, check_files=False, min_file_id=0
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
        frequencies_paths: List[Path] = []
        file = "frequencies.txt"
        for p in paths:
            new_p = io.search_file(p, file=file)
            if new_p is None:
                frequencies_paths.append(None)
            else:
                frequencies_paths.append(new_p)

        if len(frequencies_paths) == 0:
            return None

        schema_dict, _ = gtfs_checker.get_df_schema_dict("frequencies.txt")
        frequencies: pl.LazyFrame = io.read_csv_list(
            frequencies_paths, schema_overrides=schema_dict, check_files=check_files, min_file_id=min_file_id
        )

        if (frequencies is None) or (frequencies.select(pl.len()).collect().item() == 0):
            return None 

        frequencies = geo_polars.filter_by_id_column(frequencies, "trip_id", trip_ids)

        frequencies = frequencies.filter(
            (~pl.col("start_time").is_null()) & (pl.col("start_time") != "")
        )
        frequencies = frequencies.filter(
            (~pl.col("end_time").is_null()) & (pl.col("end_time") != "")
        )
        
        frequencies = frequencies.with_columns([
            normalize_time_expr("start_time").alias("start_time"),
            normalize_time_expr("end_time").alias("end_time"),
        ])

        # Lazily count rows with invalid (null) times
        null_count_expr = (
            ((pl.col("start_time") == pl.lit("None")) | (pl.col("end_time") == pl.lit("None")))
            .sum()
            .alias("num_invalid_rows")
        )

        null_count_df = frequencies.select(null_count_expr).collect()
        num_invalid_rows = null_count_df.item()  # extract scalar

        # Warn (in Python, after the lazy step)
        if num_invalid_rows > 0:
            warnings.warn(f"{num_invalid_rows} rows dropped due to invalid time values in frequencies.", UserWarning)


        frequencies = frequencies.filter(
            pl.col("start_time") != pl.lit("None")
        )

        frequencies = frequencies.filter(
            pl.col("end_time") != pl.lit("None")
        )
        # frequencies = frequencies.with_columns(
        #     [
        #         ("0" + pl.col("start_time").cast(str))
        #         .str.slice(-8, 8)
        #         .alias("start_time"),
        #         ("0" + pl.col("end_time").cast(str)).str.slice(-8, 8).alias("end_time"),
        #     ]
        # )

        # Safely cast times to seconds
        frequencies = frequencies.with_columns([
            (
                (pl.col("start_time").str.slice(0, 2).cast(pl.Int32, strict=False) * 3600)
                + (pl.col("start_time").str.slice(3, 2).cast(pl.Int32, strict=False) * 60)
                + (pl.col("start_time").str.slice(6, 2).cast(pl.Int32, strict=False))
            ).alias("start_time"),

            (
                (pl.col("end_time").str.slice(0, 2).cast(pl.Int32, strict=False) * 3600)
                + (pl.col("end_time").str.slice(3, 2).cast(pl.Int32, strict=False) * 60)
                + (pl.col("end_time").str.slice(6, 2).cast(pl.Int32, strict=False))
            ).alias("end_time"),

            pl.lit(False).alias("next_day"),
            pl.col("trip_id").alias("orig_trip_id"),
        ])

        # Lazily count rows with invalid times
        null_count_expr = (
            (pl.col("start_time").is_null() | pl.col("end_time").is_null())
            .sum()
            .alias("num_invalid_rows")
        )

        null_count_df = frequencies.select(null_count_expr).collect()
        num_invalid_rows = null_count_df.item()  # scalar count

        # Drop invalid rows
        frequencies = frequencies.filter(
            pl.col("start_time").is_not_null() & pl.col("end_time").is_not_null()
        )

        # Warn if any rows were removed
        if num_invalid_rows > 0:
            warnings.warn(f"{num_invalid_rows} rows dropped due to invalid start/end times.", UserWarning)

        return frequencies

    def _filter_repeated_frequencies_with_trips(self, frequencies, trips):
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

    def _check_frequencies_in_stop_times(self, stop_times, frequencies):
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

    def _reconcile_frequency_windows(
        self, frequencies: pl.LazyFrame, check_files: bool = False
    ) -> pl.LazyFrame:
        """
        Walks each trip's chain of `frequencies.txt` windows (all rows sharing
        a `trip_id`, sorted by `start_time` -- i.e. one calendar day's worth of
        windows for that trip; a midnight-spanning window has already been
        split into two separate `trip_id`s by `_frequencies_midnight_crossing`
        before this runs, so each chain here never itself crosses midnight)
        and reconciles each window's `end_time` against its own `headway_secs`,
        propagating the effect onto the following window so the whole day's
        chain fits together with no overlaps and no unreachable trailing
        period.

        For every window except the last one of the trip's chain:
        - If `(end_time - start_time)` is already an exact multiple of
          `headway_secs`, nothing is changed.
        - Otherwise, `end_time` is pushed UP to the next exact multiple
          (`start_time + ceil((end_time - start_time) / headway_secs) *
          headway_secs`). No trip is ever generated exactly at `end_time`
          (departure generation elsewhere already treats `end_time` as an
          exclusive bound), so this does not create a new trip at the new
          `end_time` -- it only moves the boundary out to the next clean grid
          point past it, without changing which departures exist.
        - Because that can now push `end_time` past the *next* window's
          `start_time`, the next window is reconciled in turn: if the new
          `end_time` is beyond the next window's `start_time`, that next
          window's `start_time` is pulled forward to match; if the new
          `end_time` reaches or passes the next window's `end_time` too, that
          next window is entirely subsumed and dropped, and reconciliation
          continues against the window after it.
        - The last window of a trip's chain is never touched (neither its
          `start_time` nor `end_time`), so a day's service never appears to
          extend past its originally declared end.

        Args:
            frequencies (pl.LazyFrame): The frequencies LazyFrame.
            check_files (bool): Unused; kept for call-site compatibility.

        Returns:
            pl.LazyFrame: Frequencies LazyFrame with reconciled windows.
        """
        import math

        df = frequencies.collect()
        if df.height == 0:
            return df.lazy()

        # Group by the ORIGINAL trip_id, not `trip_id` itself: by this point
        # `trip_id` may already have been suffixed per-row (e.g.
        # `_check_frequencies_in_stop_times`'s `_frequency_<n>` suffix, which
        # runs whenever `check_files=True`, the default), which would make
        # every row its own singleton group and defeat the whole point of
        # walking each trip's chain of windows together.
        chain_key = "orig_trip_id" if "orig_trip_id" in df.columns else "trip_id"

        schema = df.schema
        end_time_changes = 0
        start_time_shifts = 0
        dropped_rows = 0

        out_rows: List[dict] = []
        for group in df.partition_by(chain_key, maintain_order=True):
            recs = group.sort("start_time").to_dicts()
            n = len(recs)
            idx = 0
            while idx < n:
                row = recs[idx]
                is_last = idx == n - 1
                if not is_last:
                    start = row["start_time"]
                    end = row["end_time"]
                    headway = row["headway_secs"]
                    width = end - start
                    if headway and (width % headway != 0):
                        new_end = start + math.ceil(width / headway) * headway
                    else:
                        new_end = end

                    if new_end != end:
                        end_time_changes += 1
                        row["end_time"] = new_end

                    j = idx + 1
                    while j < n and new_end > recs[j]["start_time"]:
                        if new_end >= recs[j]["end_time"]:
                            recs.pop(j)
                            n -= 1
                            dropped_rows += 1
                        else:
                            recs[j]["start_time"] = new_end
                            start_time_shifts += 1
                            break

                out_rows.append(row)
                idx += 1

        if end_time_changes or start_time_shifts or dropped_rows:
            summary = (
                f"Reconciled frequencies.txt windows against headway_secs: "
                f"{end_time_changes} end_time(s) adjusted, "
                f"{start_time_shifts} start_time(s) pulled forward, "
                f"{dropped_rows} window(s) fully covered by a preceding window dropped."
            )
            warnings.warn(summary)

        result = pl.DataFrame(out_rows, schema=schema)
        return result.lazy()

    def _fix_headway(self, frequencies: pl.LazyFrame) -> pl.LazyFrame:
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
                # Each shifted instance starts its own day_offset count from 0
                # (it is a fresh trip departing at `delta_time` past the
                # template's first stop); if adding `delta_time` pushed this
                # stop's time past 24h, recover that as a day_offset increment
                # exactly like `_normalize_times` does for ordinary trips.
                pl.when(pl.col("delta_time") > 0)
                .then(pl.col("departure_time") // SECS_PER_DAY)
                .otherwise(pl.col("day_offset"))
                .alias("day_offset")
            )
            .with_columns(
                (pl.col("arrival_time") % SECS_PER_DAY).alias("arrival_time"),
                (pl.col("departure_time") % SECS_PER_DAY).alias("departure_time"),
            )
            .with_columns((pl.col("day_offset") > 0).alias("next_day"))
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

