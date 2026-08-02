# -*- coding: utf-8 -*-
"""Time-related helpers: GTFS time-string parsing, `datetime.time` ->
seconds-since-midnight conversion, and along-trip time/distance
interpolation.

Why this module exists and how it's organized:
-----------------------------------------------
- **`normalize_time_expr`**: split into its own concern (rather than living
  in `stop_times.py`, where it originated) purely to avoid a circular
  import -- both `stop_times.py` (`StopTimes`) and `frequencies.py`
  (`FrequenciesMixin`, mixed into `StopTimes`) need the exact same "parse a
  GTFS `HH:MM:SS`-ish string" logic, and `frequencies.py` cannot import it
  back out of `stop_times.py` without `stop_times.py` also needing to
  import `frequencies.py` (for `FrequenciesMixin`) at the same time.
- **`time_to_seconds`**: the `datetime.time`/`datetime.datetime` ->
  seconds-since-midnight conversion `Feed`'s constructor and filtering
  methods (`analysis/filtering.py`) use for `start_time`/`end_time` args.
- **`time_displacement`**: given a `Feed`-style LazyFrame with per-stop
  `shape_time_traveled`/`shape_dist_traveled`, finds -- for each trip -- the
  position (in distance) reached after `secs_disp` seconds have elapsed
  from each stop, via a forward/backward `join_asof` against the trip's own
  stops. Used by `analysis/stops.py`'s speed calculations to locate where a
  vehicle would be some fixed time before/after each stop.
"""

from datetime import datetime, time
from typing import Union

import polars as pl

SECS_PER_DAY: int = 86400


def time_to_seconds(t: Union[datetime, time]) -> int:
    """Convert datetime/time object to seconds since midnight."""
    if isinstance(t, datetime):
        t = t.time()
    return t.hour * 3600 + t.minute * 60 + t.second


def normalize_time_expr(col: str) -> pl.Expr:
    """Normalizes a GTFS-style time column into a clean `"HH:MM:SS"` string.

    Args:
        col: Name of the column to normalize (e.g. `"arrival_time"`,
            `"start_time"`).

    Returns:
        pl.Expr: `null` for empty/null input, `"None"` (the literal string,
        used as a sentinel by callers to distinguish "malformed" from "was
        genuinely blank") for anything that doesn't match `H(:MM(:SS))`,
        otherwise the zero-padded `"HH:MM:SS"` form. Hours are not clamped
        to 24, so GTFS's `>=24:00:00` overnight convention round-trips
        through this function unchanged.
    """
    # Cast to Utf8 and remove unwanted characters
    cleaned = pl.col(col).cast(pl.Utf8).str.replace_all(r'[^0-9:]', '')

    # Split into exactly 3 parts → struct
    parts_struct = cleaned.str.split_exact(":", 3)

    # Fields safely
    h = parts_struct.struct.field("field_0").fill_null("0").cast(pl.Utf8).str.zfill(2)
    m = parts_struct.struct.field("field_1").fill_null("0").cast(pl.Utf8).str.zfill(2)
    s = parts_struct.struct.field("field_2").fill_null("0").cast(pl.Utf8).str.zfill(2)

    # Valid pattern: 1-2 digits, optional :1-2 digits, optional :1-2 digits
    is_valid = cleaned.str.contains(r'^\d{1,2}(:\d{1,2}){0,2}$')

    # Handle empty strings or nulls
    is_empty_or_null = pl.col(col).is_null() | (pl.col(col) == "")

    # Combine parts safely
    result = pl.when(is_empty_or_null).then(pl.lit(None)) \
               .when(is_valid).then(h + ":" + m + ":" + s) \
               .otherwise(pl.lit("None"))

    return result


def time_displacement(gtfs_lf,secs_disp):
    gtfs_lf = gtfs_lf.with_columns(
        (pl.col("shape_time_traveled") + secs_disp).alias("target_time")
    )
    gtfs_lf = gtfs_lf.with_columns(
        pl.when(
            pl.col("target_time") < 0
        ).then(
            pl.lit(0)
        ).otherwise(
            pl.col("target_time")
        ).alias("target_time"),
    )
    gtfs_lf = gtfs_lf.with_columns(
        pl.when(
            pl.col("target_time") > pl.col("shape_total_travel_time")
        ).then(
            pl.col("shape_total_travel_time")
        ).otherwise(
            pl.col("target_time")
        ).alias("target_time"),
    )
    gtfs_lf = gtfs_lf.sort(["trip_id", "shape_time_traveled"])
    gtfs_lf = gtfs_lf.join_asof(
        gtfs_lf.select([
            "trip_id",
            pl.col("shape_time_traveled").alias("t_lb"),
            pl.col("shape_dist_traveled").alias("d_lb"),
        ]),
        left_on="target_time",
        right_on="t_lb",
        by="trip_id",
        strategy="backward",
    ).sort(["trip_id", "shape_time_traveled"])

    gtfs_lf = gtfs_lf.join_asof(
        gtfs_lf.select([
            "trip_id",
            pl.col("shape_time_traveled").alias("t_ub"),
            pl.col("shape_dist_traveled").alias("d_ub"),
        ]),
        left_on="target_time",
        right_on="t_ub",
        by="trip_id",
        strategy="forward",
    )

    gtfs_lf = gtfs_lf.with_columns(
        pl.when(pl.col("d_lb").is_null())
            .then(
                pl.lit(None)
            ).otherwise(
                pl.col("t_lb")
            ).alias("t_lb"),
        pl.when(pl.col("t_lb").is_null())
            .then(
                pl.lit(None)
            ).otherwise(
                pl.col("d_lb")
            ).alias("d_lb"),
        pl.when(pl.col("d_ub").is_null())
            .then(
                pl.lit(None)
            ).otherwise(
                pl.col("t_ub")
            ).alias("t_ub"),
        pl.when(pl.col("t_ub").is_null())
            .then(
                pl.lit(None)
            ).otherwise(
                pl.col("d_ub")
            ).alias("d_ub"),
    )

    gtfs_lf = gtfs_lf.with_columns(
        pl.when(
            pl.col("target_time") == 0
        ).then(
            pl.col("shape_time_traveled")
        ).otherwise(
            pl.col("t_ub")
        ).alias("t_ub"),
        pl.when(
            pl.col("target_time") == pl.col("shape_total_travel_time")
        ).then(
            pl.col("shape_time_traveled")
        ).otherwise(
            pl.col("t_lb")
        ).alias("t_lb"),
        pl.when(
            pl.col("target_time") == 0
        ).then(
            pl.col("shape_dist_traveled")
        ).otherwise(
            pl.col("d_ub")
        ).alias("d_ub"),
        pl.when(
            pl.col("target_time") == pl.col("shape_total_travel_time")
        ).then(
            pl.col("shape_dist_traveled")
        ).otherwise(
            pl.col("d_lb")
        ).alias("d_lb"),
    )
    
    gtfs_lf = gtfs_lf.with_columns(
        pl.when(pl.col("d_lb").is_null())
            .then(
                pl.col("shape_dist_traveled")
            ).otherwise(
                pl.col("d_lb")
            ).alias("d_lb"),
        pl.when(pl.col("t_lb").is_null())
            .then(
                pl.col("shape_time_traveled")
            ).otherwise(
                pl.col("t_lb")
            ).alias("t_lb"),
        pl.when(pl.col("d_ub").is_null())
            .then(
                pl.col("shape_dist_traveled")
            ).otherwise(
                pl.col("d_ub")
            ).alias("d_ub"),
        pl.when(pl.col("t_ub").is_null())
            .then(
                pl.col("shape_time_traveled")
            ).otherwise(
                pl.col("t_ub")
            ).alias("t_ub"),
    )

    gtfs_lf = gtfs_lf.with_columns(
        # Calculate "time"
        (
            pl.when(pl.col("t_lb").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("t_ub").is_null())
                    .then(pl.lit(None))
                    .otherwise(pl.col("target_time") - pl.col("shape_time_traveled"))
            )
        ).abs().alias("time_weight"),
        
        # Calculate "distance"
        (
            pl.when(pl.col("d_lb").is_null())
            .then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("d_ub").is_null())
                    .then(pl.lit(None))
                    .otherwise(
                        (
                            pl.col("d_lb") + 
                            (
                                (pl.col("d_ub") - pl.col("d_lb")) / 
                                (pl.col("t_ub") - pl.col("t_lb"))
                            ) * 
                            (
                                pl.col("target_time") - pl.col("t_lb")
                            )
                        ) - pl.col("shape_dist_traveled")
                    )
            )
        ).abs().alias("distance_weight")
    )#.drop(["d_lb","d_ub","t_lb","t_ub", "target_time"])

    gtfs_lf = gtfs_lf.with_columns(
        pl.when(
            pl.col("time_weight").is_infinite() | pl.col("time_weight").is_nan()
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col("time_weight")
        ).alias("time_weight"),

        pl.when(
            pl.col("distance_weight").is_infinite() | pl.col("distance_weight").is_nan()
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col("distance_weight")
        ).alias("distance_weight"),
    )

    gtfs_lf = gtfs_lf.with_columns(
        pl.when(
            pl.col("time_weight").is_null()
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col("distance_weight")
        ).alias("distance_weight"),
        pl.when(
            pl.col("distance_weight").is_null()
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col("time_weight")
        ).alias("time_weight")
    )

    return gtfs_lf
